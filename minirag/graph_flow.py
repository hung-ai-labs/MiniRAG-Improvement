"""Opt-in, bounded two-step graph flow and provenance projection (P3).

No LLM, graph/index writes, gold labels, or path enumeration in the full mode.
Relationship similarities come from the existing query retrieval. See
reproduce/path/P3_PROTOCOL.md for the frozen mechanism and ablations.
"""
import asyncio
from collections import defaultdict
import math
import time

from .prompt import GRAPH_FIELD_SEP

MODES = frozenset(("full", "no_normalization", "node_provenance", "enumerated"))
EVIDENCE_MODES = frozenset(("evidence_flow", "evidence_uniform", "evidence_sum",
                           "evidence_dense", "evidence_enumerated", "evidence_substochastic",
                           "evidence_lift", "evidence_lift_enumerated"))
MODES = MODES | EVIDENCE_MODES
CANDIDATE_MODES = frozenset(("semantic", "topology", "topology_threshold"))


async def topology_candidates(seed_names, graph, relationship_vectors, query, *, mode="topology"):
    """Discover two-hop edges in O(E), then score existing relation vectors.

    Expand adjacency once per distinct seed/first-hop node. Global semantic
    top-k is not an admission gate; no paths or provenance labels are read.
    """
    if mode not in CANDIDATE_MODES - {"semantic"}:
        raise ValueError(f"Unknown topology candidate mode: {mode!r}")
    if not callable(getattr(relationship_vectors, "score_pairs", None)):
        raise ValueError("P3 topology candidates require relationship score_pairs support")
    started = time.perf_counter()
    seeds = set(seed_names)
    pairs, loaded = set(), set()
    frontier = seeds
    for _ in range(2):
        names = sorted(frontier - loaded)
        lists = await asyncio.gather(*(graph.get_node_edges(name) for name in names))
        loaded.update(names)
        next_frontier = set()
        for edges in lists:
            for u, v in edges or []:
                if u != v:
                    pairs.add(tuple(sorted((u, v))))
                    next_frontier.update((u, v))
        frontier = next_frontier - loaded
    discovery_ms = 1000 * (time.perf_counter() - started)
    rows = await relationship_vectors.score_pairs(query, sorted(pairs))
    returned = {tuple(sorted((row["src_id"], row["tgt_id"]))) for row in rows}
    if returned != pairs:
        raise ValueError("P3 topology scorer did not return exactly the discovered edges")
    threshold = (float(relationship_vectors.cosine_better_than_threshold)
                 if mode == "topology_threshold" else 0.0)
    if not math.isfinite(threshold):
        raise ValueError("P3 requires a finite candidate threshold")
    retained = [row for row in rows if _positive(row["distance"]) > 0
                and float(row["distance"]) >= threshold]
    return retained, {
        "mode": mode, "adjacency_loads": len(loaded), "discovered_edges": len(pairs),
        "retained_rows": len(retained), "cosine_threshold": threshold,
        "discovery_ms": discovery_ms, "total_ms": 1000 * (time.perf_counter() - started),
    }


def _positive(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("P3 requires finite cosine similarities")
    return max(0.0, number)


def _sources(data, valid_chunks):
    raw = (data or {}).get("source_id", "")
    if not isinstance(raw, str):
        raise ValueError("P3 source_id must be a string")
    ids = {part.strip() for part in raw.split(GRAPH_FIELD_SEP) if part.strip()}
    return tuple(sorted(ids if valid_chunks is None else ids & valid_chunks))


def seed_distribution(seed_groups, eligible):
    """One budget per query entity; duplicate seeds in a group get no extra mass."""
    mass = defaultdict(float)
    if not seed_groups:
        return {}
    budget = 1.0 / len(seed_groups)
    for group in seed_groups:
        unique = {}
        for name, similarity in group:
            if name in eligible:
                unique[name] = max(unique.get(name, 0.0), _positive(similarity))
        if not unique:
            continue
        denominator = math.fsum(unique.values())
        for name, score in unique.items():
            probability = score / denominator if denominator else 1.0 / len(unique)
            mass[name] += budget * probability
    return dict(mass)


def _two_steps(adjacency, seeds, mode):
    """Directed edge states implement non-backtracking h2 in linear edge work."""
    normalized = mode not in ("no_normalization", "enumerated_raw")
    totals = {u: math.fsum(neighbors.values()) for u, neighbors in adjacency.items()}
    first = {}
    first_updates = 0
    for u, mass in sorted(seeds.items()):
        denominator = totals.get(u, 0.0) if normalized else 1.0
        if denominator <= 0:
            continue
        for v, weight in adjacency.get(u, {}).items():
            first[(u, v)] = mass * weight / denominator
            first_updates += 1

    second = defaultdict(float)
    second_updates = 0
    enumerated_paths = 0
    if mode in ("enumerated", "enumerated_raw"):
        # Reference ablation only; full never enters this nested expansion.
        for (u, v), mass in sorted(first.items()):
            denominator = totals[v] - adjacency[v][u] if normalized else 1.0
            if denominator <= 0:
                continue
            for x, weight in adjacency[v].items():
                if x != u:
                    second[(v, x)] += mass * weight / denominator
                    second_updates += 1
                    enumerated_paths += 1
    else:
        incoming = defaultdict(list)
        reverse_terms = {}
        for (u, v), mass in sorted(first.items()):
            denominator = totals[v] - adjacency[v][u] if normalized else 1.0
            if denominator > 0:
                term = mass / denominator
                incoming[v].append(term)
                reverse_terms[(u, v)] = term
        for v, terms in sorted(incoming.items()):
            total = math.fsum(terms)
            for x, weight in adjacency[v].items():
                value = weight * max(0.0, total - reverse_terms.get((x, v), 0.0))
                if value > 0:
                    second[(v, x)] = value
                second_updates += 1
    return first, dict(second), {
        "first_step_updates": first_updates,
        "second_step_updates": second_updates,
        "paths_enumerated": enumerated_paths,
    }


async def score_graph_flow(seed_groups, eligible, relationships, graph, *,
                           valid_chunks=None, mode="full", chunk_vectors=None, query=None):
    """Return global chunk scores and a JSON-safe diagnostic trace.

    eligible is the retained seed-name set; groups contain (name, cosine) pairs
    for distinct query entity strings. relationships is the selected edge pool.
    All caches are query-local, so later index mutations cannot leave stale data.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown P3 mode: {mode!r}")
    native = getattr(graph, "_graph", None)
    if native is not None and native.is_directed():
        raise ValueError("P3 currently requires an undirected graph")
    started = time.perf_counter()
    seeds = seed_distribution(seed_groups, eligible)
    weights = {}
    self_loops = 0
    for row in relationships:
        u, v = row["src_id"], row["tgt_id"]
        weight = _positive(row["distance"])
        if u == v:
            self_loops += 1
            continue
        if weight > 0:
            key = tuple(sorted((u, v)))
            weights[key] = max(weights.get(key, 0.0), weight)
    keys = sorted(weights)
    data = await asyncio.gather(*(graph.get_edge(u, v) for u, v in keys))
    adjacency = defaultdict(dict)
    provenance = {}
    nodes = {}
    if mode == "node_provenance":
        names = sorted({name for key in keys for name in key})
        node_data = await asyncio.gather(*(graph.get_node(name) for name in names))
        nodes = dict(zip(names, node_data))
    for (u, v), edge in zip(keys, data):
        if edge is None:
            raise ValueError(f"P3 relationship has no graph edge: {(u, v)!r}")
        adjacency[u][v] = weights[(u, v)]
        adjacency[v][u] = weights[(u, v)]
        if mode == "node_provenance":
            ids = tuple(sorted(set(_sources(nodes[u], valid_chunks)) |
                               set(_sources(nodes[v], valid_chunks))))
        else:
            ids = _sources(edge, valid_chunks)
        provenance[(u, v)] = ids
    adjacency = {u: dict(sorted(neighbors.items())) for u, neighbors in sorted(adjacency.items())}
    extras = {}
    if mode in EVIDENCE_MODES:
        from .evidence_flow import score_evidence_flow
        scores, first, second, operations, extras = await score_evidence_flow(
            seeds, weights, provenance, chunk_vectors, query, mode)
        unprojected_mass = 0.0
    else:
        first, second, operations = _two_steps(adjacency, seeds, mode)

        edge_credits = defaultdict(list)
        for flow in (first, second):
            for (u, v), mass in sorted(flow.items()):
                edge_credits[tuple(sorted((u, v)))].append(mass / 2.0)
        contributions = defaultdict(list)
        unprojected_mass = 0.0
        for edge, values in sorted(edge_credits.items()):
            credit = math.fsum(values)
            ids = provenance[edge]
            if ids:
                for cid in ids:
                    contributions[cid].append(credit / len(ids))
            else:
                unprojected_mass += credit
        scores = {cid: math.fsum(values) for cid, values in sorted(contributions.items())}
    emitted = math.fsum(scores.values())
    if mode != "no_normalization" and emitted > 1.0 + 1e-9:
        raise ArithmeticError("P3 violated its query mass budget")
    trace = {
        "mode": mode, "query_entities": len(seed_groups), "seeds": len(seeds),
        "seed_mass": math.fsum(seeds.values()), "weighted_edges": len(keys),
        "edge_loads": len(keys), "self_loops_ignored": self_loops,
        "step1_mass": math.fsum(first.values()), "step2_mass": math.fsum(second.values()),
        "emitted_mass": emitted, "unprojected_mass": unprojected_mass,
        "node_loads": len(nodes), "chunk_scores": scores, **operations, **extras,
        "candidate_chunk_ids": sorted({cid for ids in provenance.values() for cid in ids}),
        "kernel_ms": 1000 * (time.perf_counter() - started),
    }
    return scores, trace


def rank_chunks(scores, count):
    """Numerical tie convention is shared by dynamic and enumerated variants."""
    return sorted((cid for cid, score in scores.items() if score > 0),
                  key=lambda cid: (-round(scores[cid], 12), cid))[:count]
