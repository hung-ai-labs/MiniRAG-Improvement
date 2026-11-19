"""Registered nested dev400 SET evaluation. Offline, zero API calls."""
from __future__ import annotations

import ast
import json
import math
import statistics

from reproduce.mhr_ce import offline as ce
from reproduce.mhr_f import study as base
from reproduce.mhr_f.pool import OUT as F_OUT
from reproduce.mhr_set import offline
from reproduce.mhr_set.select import facets, select
from reproduce.mhr_st.common import ROOT, read, records, sha

SCORES = ce.OUT / 'dev_offline/scores_CEF.jsonl'
MODEL = ce.OUT / 'model_CE.json'
DEV_POOLS = F_OUT / 'dev/s101/pools.jsonl'


def grid_constants():
    # Read literal constants without importing evaluate: its imports load LiHua.
    tree = ast.parse((ROOT / 'reproduce/evidence_set/evaluate.py').read_text(encoding='utf-8'))
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ('BETAS', 'TAUS')}
    assert values == {'BETAS': (0., .25, .5, 1., 2.), 'TAUS': (0., .05, .1, .2, .5, 1.)}
    return values['BETAS'], values['TAUS']


BETAS, TAUS = grid_constants()


def data():
    rows = ce.csv_rows('dev')
    assert len(rows) == 400, 'Registered dev400 is required'
    summary = read(F_OUT / 'dev/s101/pools_summary.json')
    assert sha(DEV_POOLS) == summary['pools_sha256'], 'Dev pools changed'
    raw = records(DEV_POOLS)
    saved = {r['query_id']: r for r in raw}
    assert len(raw) == len(saved) and set(saved) == {r['query_id'] for r in rows}
    pool_name = read(F_OUT / 'dev_offline/coverage.json')['chosen_pool']
    assert pool_name == 'native@all', 'Registered pool differs'
    pools = {r['query_id']: base.variants(saved[r['query_id']])[pool_name]
             if saved[r['query_id']]['status'] == 'ok' else [] for r in rows}
    fold = base.folds(rows)
    p = {}
    for s in records(SCORES):
        q, c, v = s['query_id'], s['chunk_id'], float(s['p'])
        assert q in saved and s['fold'] == fold[q], 'Scored fold differs'
        assert s['fold'] not in s['trained_on_folds'], 'Cross-fit leakage'
        assert set(s['trained_on_folds']) == set(range(base.FOLDS)) - {fold[q]}
        assert math.isfinite(v) and 0 <= v <= 1, 'p must be a probability'
        assert c in pools[q] and c not in p.get(q, {}), 'Unknown or duplicate score'
        p.setdefault(q, {})[c] = v
    assert all(set(p.get(q, {})) == set(pool) for q, pool in pools.items()), 'Incomplete scores'
    chunks, _, by_time = base.corpus()
    fmap = {q: facets(saved[q], pool, by_time) for q, pool in pools.items()}
    return rows, pools, p, fold, fmap, chunks


def grid_results(rows, pools, p, fmap, chunks):
    gs = base.gold_sets()
    lab = [r['query_id'] for r in rows if r['query_id'] in gs]
    grid = {}
    for beta in BETAS:
        for tau in TAUS:
            scores = {}
            for q in lab:
                ids, info = select(pools[q], p.get(q, {}), fmap[q], chunks, beta=beta, tau=tau)
                scores[q] = (base.full(ids, gs[q]), info['tokens'])
            grid[beta, tau] = scores
    return lab, grid


def choose(ids, grid, betas=BETAS):
    """evaluate.tune / nondev.dev_beta_tau: 1% tolerance then mean tokens, -full, (b,t)."""
    assert ids, 'No labelled training questions'
    candidates = [bt for bt in grid if bt[0] in betas]
    count = lambda bt: sum(grid[bt][q][0] for q in ids)
    best = max(count(bt) for bt in candidates)
    ok = [bt for bt in candidates if count(bt) >= best - math.ceil(.01 * len(ids))]
    return min(ok, key=lambda bt: (statistics.mean(grid[bt][q][1] for q in ids), -count(bt), bt))


def grid_table(lab, grid):
    return [{'beta': b, 'tau': t, 'labelled': len(lab),
             'full_evidence': sum(v[q][0] for q in lab),
             'tokens_mean': statistics.mean(v[q][1] for q in lab)} for (b, t), v in grid.items()]


def hashes():
    return {'scores_CEF_sha256': sha(SCORES), 'model_CE_sha256': sha(MODEL),
            'pools_sha256': sha(DEV_POOLS)}


def main():
    offline.install()
    dest = offline.OUT / 'dev_offline'
    if any((dest / n).exists() for n in ('offline_eval.json', 'selections.jsonl')):
        raise FileExistsError('ADD-ONLY: dev outputs already exist')
    rows, pools, p, fold, fmap, chunks = data()
    gs = base.gold_sets()
    lab, grid = grid_results(rows, pools, p, fmap, chunks)
    choice = {f: choose([q for q in lab if fold[q] != f], grid) for f in range(base.FOLDS)}
    beta0 = {f: choose([q for q in lab if fold[q] != f], grid, betas=(0.,)) for f in range(base.FOLDS)}
    arms = {a: {} for a in ('CE+F', 'SET', 'SET_beta0', 'SET_tau0', 'ORACLE')}
    oldraw = records(ce.OUT / 'dev_offline/selections.jsonl')
    old = {r['query_id']: r['CE+F'] for r in oldraw}
    assert len(oldraw) == len(old) == len(rows) and set(old) == set(pools)
    for r in rows:
        q = r['query_id']
        pool = pools[q]
        b, t = choice[fold[q]]
        order = sorted(range(len(pool)), key=lambda i: (-p.get(q, {}).get(pool[i], 0.), i))
        arms['CE+F'][q] = base.a1([pool[i] for i in order])
        assert arms['CE+F'][q] == old[q], f'CE+F replay mismatch: {q}'
        for arm, bt in (('SET', (b, t)), ('SET_beta0', beta0[fold[q]]), ('SET_tau0', (b, 0.))):
            arms[arm][q] = select(pool, p.get(q, {}), fmap[q], chunks, beta=bt[0], tau=bt[1])[0]
        gold = gs.get(q, [])
        arms['ORACLE'][q] = base.a1([c for c in pool if any(c in g for g in gold)] +
                                    [c for c in pool if not any(c in g for g in gold)])
    tok = lambda ids: sum(chunks[c]['tokens'] for c in ids)
    for arm in arms.values():
        for q, ids in arm.items():
            assert tok(ids) <= base.BUDGET and len(ids) == len(set(ids)) and set(ids) <= set(pools[q])

    def stats(arm, subset):
        ids = [r['query_id'] for r in subset]
        hit = sum(base.facts_hit(arms[arm][q], gs[q]) for q in ids)
        total = sum(len(gs[q]) for q in ids)
        return {'n': len(ids), 'full_evidence': sum(base.full(arms[arm][q], gs[q]) for q in ids),
                'facts_hit': hit, 'facts': total, 'micro_recall': round(hit / total, 4),
                'macro_recall': round(statistics.mean(base.facts_hit(arms[arm][q], gs[q]) / len(gs[q]) for q in ids), 4),
                'tokens_mean': statistics.mean(tok(arms[arm][q]) for q in ids),
                'tokens_median': statistics.median(tok(arms[arm][q]) for q in ids),
                'chunks_mean': statistics.mean(len(arms[arm][q]) for q in ids),
                'chunks_median': statistics.median(len(arms[arm][q]) for q in ids)}

    labelled = [r for r in rows if r['query_id'] in lab]
    types = sorted({r['Type'] for r in labelled})
    up = sum(base.full(arms['SET'][q], gs[q]) and not base.full(arms['CE+F'][q], gs[q]) for q in lab)
    down = sum(base.full(arms['CE+F'][q], gs[q]) and not base.full(arms['SET'][q], gs[q]) for q in lab)
    fu = sum(base.facts_hit(arms['SET'][q], gs[q]) > base.facts_hit(arms['CE+F'][q], gs[q]) for q in lab)
    fd = sum(base.facts_hit(arms['SET'][q], gs[q]) < base.facts_hit(arms['CE+F'][q], gs[q]) for q in lab)
    result = {'seed': 101, 'questions': len(rows), 'labelled': len(lab), 'pool': 'native@all', 'budget': base.BUDGET,
              'fold_sizes': {f: sum(v == f for v in fold.values()) for f in range(base.FOLDS)},
              'tuned_beta_tau_per_fold': {'SET': choice, 'SET_beta0': beta0,
                                         'SET_tau0': {f: [bt[0], 0.] for f, bt in choice.items()}},
              'SET_tau0_rule': 'Nested SET beta, tau fixed to zero (evidence_set.evaluate SET_FILL)',
              'grid': grid_table(lab, grid), 'CE+F_selections_equal_reference': True,
              'CE+F_reference_questions_checked': len(old),
              'CE+F_reference_sha256': sha(ce.OUT / 'dev_offline/selections.jsonl'),
              'arms': {a: {'all': stats(a, labelled), **{t: stats(a, [r for r in labelled if r['Type'] == t]) for t in types}}
                       for a in arms},
              'selection_stats_all_questions': {
                  a: {'tokens_mean': statistics.mean(tok(v) for v in sel.values()),
                      'tokens_median': statistics.median(tok(v) for v in sel.values()),
                      'chunks_mean': statistics.mean(len(v) for v in sel.values()),
                      'chunks_median': statistics.median(len(v) for v in sel.values())} for a, sel in arms.items()},
              'SET_vs_CE+F': {'full_up': up, 'full_down': down, 'full_p_exact': base.exact_p(up, down),
                              'facts_up_questions': fu, 'facts_down_questions': fd,
                              'facts_sign_p_exact': base.exact_p(fu, fd)}, **hashes(), 'api_calls': 0}
    offline.jsonl(dest / 'selections.jsonl', ({'query_id': r['query_id'], **{a: v[r['query_id']] for a, v in arms.items()}} for r in rows))
    offline.write(dest / 'offline_eval.json', result)
    print(json.dumps({'arms': {a: result['arms'][a]['all'] for a in arms},
                      'tuned_beta_tau_per_fold': result['tuned_beta_tau_per_fold'],
                      'SET_vs_CE+F': result['SET_vs_CE+F']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
