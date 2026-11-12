"""Evidence-conditioned graph transitions; no labels, index writes, or new model."""
from collections import defaultdict
import math

from .graph_flow import _positive, _two_steps, EVIDENCE_MODES

MODES = EVIDENCE_MODES


async def score_evidence_flow(seeds, weights, provenance, vectors, query, mode):
    """Return projected scores, directed flows, work counts and diagnostics.

    Distinct supporting chunks receive outgoing budgets before their supporting
    edges do. Thus adding another relation citing one chunk does not create a
    second chunk vote. See P3_EVIDENCE_FLOW_PROTOCOL.md for the fixed equations.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown evidence flow mode: {mode!r}")
    candidates = sorted({c for ids in provenance.values() for c in ids})
    if mode == "evidence_uniform":
        affinity = {c: 1.0 for c in candidates}
    else:
        if not callable(getattr(vectors, "score_ids", None)):
            raise ValueError("P3 evidence flow requires chunk score_ids support")
        rows = await vectors.score_ids(query, candidates)
        if {row["id"] for row in rows} != set(candidates) or len(rows) != len(candidates):
            raise ValueError("P3 evidence scorer must return exactly one score per candidate chunk")
        affinity = {row["id"]: _positive(row["distance"]) for row in rows}
    extras = {"chunk_vector_reads": 0 if mode == "evidence_uniform" else len(candidates),
              "evidence_component_updates": 0, "transition_edges": 0}
    if mode == "evidence_dense":
        total = math.fsum(affinity.values())
        scores = {c: a / total for c, a in sorted(affinity.items()) if a > 0} if total else {}
        return scores, {}, {}, {"first_step_updates": 0, "second_step_updates": 0,
                                "paths_enumerated": 0}, extras

    substochastic = mode in ("evidence_substochastic", "evidence_lift", "evidence_lift_enumerated")
    if substochastic:
        total = math.fsum(affinity.values())
        affinity = {c: a / total if total else 0.0 for c, a in affinity.items()}

    # Incidence-weight normalization is local to each source node, not global
    # document frequency. Every (node, chunk) has exactly one outgoing budget.
    denominator_terms = defaultdict(list)
    for (u, v), relation in sorted(weights.items()):
        for c in provenance[(u, v)]:
            denominator_terms[(u, c)].append(relation)
            denominator_terms[(v, c)].append(relation)
    denominators = {key: math.fsum(values) for key, values in denominator_terms.items()}
    adjacency, components = defaultdict(dict), {}
    for (u, v), relation in sorted(weights.items()):
        for source, target in ((u, v), (v, u)):
            values = {}
            for c in provenance[(u, v)]:
                denominator = 1.0 if mode == "evidence_sum" else denominators[(source, c)]
                value = affinity[c] * relation / denominator
                if value > 0:
                    values[c] = value
                extras["evidence_component_updates"] += 1
            weight = math.fsum(values.values())
            if weight > 0:
                adjacency[source][target] = weight
                components[(source, target)] = values
    adjacency = {u: dict(sorted(neighbors.items())) for u, neighbors in sorted(adjacency.items())}
    extras["transition_edges"] = sum(len(n) for n in adjacency.values())
    if substochastic and any(math.fsum(n.values()) > 1 + 1e-9 for n in adjacency.values()):
        raise ArithmeticError("Evidence transition row exceeded its substochastic budget")
    kernel_mode = ("enumerated_raw" if mode == "evidence_lift_enumerated" else "no_normalization") if substochastic else (
        "enumerated" if mode == "evidence_enumerated" else "full")

    def project(matrix, parts):
        first, second, work = _two_steps(matrix, seeds, kernel_mode)
        contributions = defaultdict(list)
        for flow in (first, second):
            for (u, v), mass in sorted(flow.items()):
                for c, value in parts[(u, v)].items():
                    contributions[c].append((mass / 2.0) * (value / matrix[u][v]))
        return {c: math.fsum(v) for c, v in sorted(contributions.items())}, first, second, work

    scores, first, second, work = project(adjacency, components)
    if mode in ("evidence_lift", "evidence_lift_enumerated"):
        background_parts, background_matrix = {}, defaultdict(dict)
        for (u, v), relation in sorted(weights.items()):
            for source, target in ((u, v), (v, u)):
                values = {c: relation / denominators[(source,c)] / len(candidates) for c in provenance[(u,v)]}
                weight = math.fsum(values.values())
                if weight > 0:
                    background_parts[(source,target)] = values
                    background_matrix[source][target] = weight
                extras["evidence_component_updates"] += len(values)
        background_matrix = {u:dict(sorted(n.items())) for u,n in sorted(background_matrix.items())}
        background, _, _, background_work = project(background_matrix, background_parts)
        raw_mass = math.fsum(scores.values())
        ratios = {}
        for c, value in scores.items():
            if value > 0:
                if background.get(c, 0) <= 0:
                    raise ArithmeticError("Positive evidence flow has no background support")
                ratios[c] = value / background[c]
        total = math.fsum(ratios.values())
        scores = {c: value / total * raw_mass for c,value in sorted(ratios.items())} if total else {}
        extras.update(raw_emitted_mass=raw_mass, background_emitted_mass=math.fsum(background.values()))
        for key,value in background_work.items():
            work[key] += value
    return scores, first, second, work, extras
