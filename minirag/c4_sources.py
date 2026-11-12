"""Source-grounded C4 prototype: quoted relation occurrences, not inferred events.

An exact quote grounds the source location, not the truth of a model's relation.
Unknown witnesses fall back to C3 chunk affinity; no temporal hard filtering.
"""
import re
import math
import copy


def normalized(text):
    return ''.join(c for c in text.casefold() if c.isalnum())


def validate_fact(record, pairs, content):
    i = record.get('pair_id')
    quote = record.get('quote')
    relation = record.get('relation')
    if type(i) is not int or not 0 <= i < len(pairs):
        return None, 'invalid_pair'
    if not isinstance(quote, str) or len(quote.strip()) < 12 or quote not in content:
        return None, 'not_exact_quote'
    span = record.get('source_span')
    if span is not None:
        if (not isinstance(span, list) or len(span) != 2 or
                any(type(x) is not int for x in span) or
                not 0 <= span[0] < span[1] <= len(content) or content[span[0]:span[1]] != quote):
            return None, 'invalid_source_span'
        start, end = span
    else:
        if content.count(quote) != 1:
            return None, 'ambiguous_quote_location'
        start = content.index(quote)
        end = start + len(quote)
    if not isinstance(relation, str) or not relation.strip():
        return None, 'missing_relation'
    u, v = pairs[i]
    if not all(normalized(n) and normalized(n) in normalized(quote) for n in (u, v)):
        return None, 'endpoints_not_in_quote'
    markers = list(re.finditer(r'(?m)^Time:\s*(\d{8}_\d{2}:\d{2})', content))
    preceding = [m for m in markers if m.start() <= start]
    if any(start < m.start() < end for m in markers):
        return None, 'cross_event_quote'
    event = preceding[-1] if preceding else None
    return {'pair': sorted((u, v)), 'quote': quote, 'relation': relation.strip(),
            'span': [start, end], 'event_time': event.group(1) if event else None,
            'event_offset': event.start() if event else None,
            'status': 'exact_quote_model_relation_unverified'}, None


def fact_text(fact):
    return f"Time: {fact.get('event_time') or 'unknown'}\n{fact['pair'][0]} {fact['relation']} {fact['pair'][1]}\n{fact['quote']}"


def source_affinities(chunk_affinities, facts, similarities, *, flat=False):
    """Only a voted edge/source occurrence may receive its grounded fact score."""
    by_pair = {}
    by_chunk = {}
    for fact, score in zip(facts, similarities, strict=True):
        if not math.isfinite(float(score)) or not -1.000001 <= float(score) <= 1.000001:
            raise ValueError('Invalid witness cosine')
        score = max(0., min(1., float(score)))
        pair = tuple(sorted(fact['pair']))
        cid = fact['source_chunk']
        by_pair[(pair, cid)] = max(by_pair.get((pair, cid), 0.), score)
        by_chunk[cid] = max(by_chunk.get(cid, 0.), score)

    def affinity(kind, key, cid):
        fallback = chunk_affinities[cid]
        if flat:
            return by_chunk.get(cid, fallback)
        if kind == 'edge':
            return by_pair.get((tuple(sorted(key)), cid), fallback)
        return fallback
    return affinity


async def score_sources(entries, graph, pairs, query, max_chunks, *, valid,
                        relation_vectors, bundle, facts, similarities, flat=False,
                        fixed_candidates=False):
    """C4 projection; preserve original path scores and both source channels."""
    from .p0_path_scoring import p0_path_weights, rank_scoring, edge_key
    from .path_projection import project_paths
    relevant = [(f, s) for f, s in zip(facts, similarities, strict=True)
                if f['source_chunk'] in bundle['affinities']]
    if not relevant:
        return rank_scoring(bundle, max_chunks)
    keys = {edge_key(u,v) for e in entries.values() for p in e['Path'] for u,v in zip(p,p[1:])}
    rows = await relation_vectors.score_pairs(query,keys)
    scores = {edge_key(r['src_id'],r['tgt_id']):r['distance'] for r in rows}
    weights, _, _ = p0_path_weights(entries,scores)
    affinity = source_affinities(bundle['affinities'], [f for f,s in relevant],
                                 [s for f,s in relevant],flat=flat)
    kwargs = dict(valid=valid,source_affinity=affinity)
    prior = await project_paths(copy.deepcopy(entries),graph,pairs,query,len(valid),'local',**kwargs)
    conditioned = await project_paths(copy.deepcopy(entries),graph,pairs,query,len(valid),'local',
                                      path_weights=weights,**kwargs)
    if fixed_candidates:
        selected = rank_scoring(bundle,max_chunks)
    else:
        selected = {}
        for seed, e in prior.items():
            ranked = [c for c in e['Path'] if e['PathChunkScores'][c] > 0][:max_chunks]
            selected[seed] = {'Score':e['Score'],'Path':ranked}
    result = {}
    for seed,e in selected.items():
        denominator = bundle['background'][seed]['PathChunkScores']
        numerator = conditioned[seed]['PathChunkScores']
        values = {c:numerator[c]/denominator[c] for c in e['Path']}
        if any(not math.isfinite(v) or v < 0 for v in values.values()):
            raise ValueError('Invalid C4 source score')
        result[seed] = {'Score':e['Score'],'Path':list(e['Path']),'PathChunkScores':values}
    return result
