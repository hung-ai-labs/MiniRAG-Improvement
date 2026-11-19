"""Synthetic parity with an inline reference from evidence_set/select.py; no corpus import."""
import math
import random
import unittest

from reproduce.mhr_set.select import facets, select


def reference(pool, p, facet_map, chunks, *, beta=1.0, tau=0.0, budget=4000, group_docs=True):
    # Inline reference transcribed from evidence_set.select.select; only corpus()
    # is replaced with the supplied synthetic chunks map.
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


class SelectionTests(unittest.TestCase):
    def test_parity(self):
        chunks = {'a': {'doc': 'one', 'tokens': 1100}, 'b': {'doc': 'one', 'tokens': 900},
                  'c': {'doc': 'two', 'tokens': 700}, 'long': {'doc': 'three', 'tokens': 5000},
                  'zero': {'doc': 'four', 'tokens': 100}}
        pool = ['long', 'a', 'b', 'c', 'zero', 'a', 'unknown']
        fm = {'kw:x': ({'a', 'b'}, math.log(3)), 'date:x': ({'c'}, math.log(6))}
        rng = random.Random(13)
        for _ in range(12):
            p = {c: rng.uniform(-.2, 1.2) for c in chunks}
            for beta in (0, .25, .5, 1, 2):
                for tau in (0, .05, .1, .2, .5, 1):
                    for grouped in (True, False):
                        kw = dict(beta=beta, tau=tau, budget=2500, group_docs=grouped)
                        self.assertEqual(select(pool, p, fm, chunks, **kw),
                                         reference(pool, p, fm, chunks, **kw))

    def test_skip_and_ties_and_zero(self):
        chunks = {c: {'doc': c, 'tokens': t} for c, t in [('long', 5000), ('a', 2000), ('b', 2000), ('c', 10)]}
        self.assertEqual(select(['long', 'b', 'a'], {'long': 1, 'a': .5, 'b': .5}, {}, chunks)[0], ['b', 'a'])
        self.assertEqual(select(['long', 'c'], {'long': 1, 'c': .1}, {}, chunks)[0], ['c'])
        self.assertEqual(select(['c'], {}, {}, chunks, tau=0)[0], ['c'])
        self.assertEqual(select(['c'], {}, {}, chunks, tau=.05)[0], [])

    def test_facets(self):
        r = {'groups': {'x': [{'path': ['a', 'outside']}], 'empty': []}, 'anchors': ['day', 'absent']}
        fm = facets(r, ['a', 'b'], {'day': ['b', 'outside']})
        self.assertEqual(fm, {'kw:x': ({'a'}, math.log(3)), 'date:day': ({'b'}, math.log(3))})


if __name__ == '__main__':
    from reproduce.mhr_set.offline import guard
    guard()
    unittest.main()
