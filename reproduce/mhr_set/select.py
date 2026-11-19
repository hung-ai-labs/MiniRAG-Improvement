"""Registered LiHua greedy selector, parameterised by MultiHop corpus maps."""
import math

BUDGET = 4000


def facets(r, pool, by_time):
    """Label-free keyword seed-path and article publication-date facets."""
    inpool = set(pool)
    out = {}
    for kw, seeds in r['groups'].items():
        support = {c for s in seeds for c in s['path'] if c in inpool}
        if support:
            out['kw:' + kw] = support
    for a in r['anchors']:
        support = {c for c in by_time.get(a, ()) if c in inpool}
        if support:
            out['date:' + a] = support
    n = max(len(pool), 1)
    return {f: (s, math.log(1 + n / len(s))) for f, s in out.items()}


def select(pool, p, facet_map, chunks, *, beta=1.0, tau=0.0, budget=BUDGET, group_docs=True):
    pool = [c for c in dict.fromkeys(pool) if c in chunks]
    by_chunk = {c: [] for c in pool}
    for f, (support, w) in facet_map.items():
        for c in support:
            by_chunk[c].append(f)
    miss_doc = {}
    miss_facet = {f: 1.0 for f in facet_map}
    chosen, used, trace = [], 0, []
    left = set(pool)
    while True:
        best, best_ratio, best_gain = None, -1.0, 0.0
        for c in pool:
            if c not in left or used + chunks[c]['tokens'] > budget:
                continue
            q = 1.0 - min(max(p.get(c, 0.0), 0.0), 1.0)
            key = chunks[c]['doc'] if group_docs else c
            gain = miss_doc.get(key, 1.0) * (1.0 - q)
            if beta:
                gain += beta * sum(facet_map[f][1] * miss_facet[f] * (1.0 - q) for f in by_chunk[c])
            ratio = 1000.0 * gain / chunks[c]['tokens']
            if ratio > best_ratio:
                best, best_ratio, best_gain = c, ratio, gain
        if best is None or best_ratio < tau or (tau > 0 and best_gain <= 0):
            break
        q = 1.0 - min(max(p.get(best, 0.0), 0.0), 1.0)
        key = chunks[best]['doc'] if group_docs else best
        miss_doc[key] = miss_doc.get(key, 1.0) * q
        for f in by_chunk[best]:
            miss_facet[f] *= q
        chosen.append(best)
        used += chunks[best]['tokens']
        left.discard(best)
        trace.append(round(best_ratio, 4))
    return chosen, {'tokens': used, 'gain_per_1k': trace}
