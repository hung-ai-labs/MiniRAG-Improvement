"""Step A of +F on MultiHop-RAG: pool rule, features, cross-fitted scorer F, offline gate (0 API calls).

Same method as the LiHua scorer (reproduce/evidence_set: features.py, quick_scorer.py,
coverage.py), ported with the smallest changes the corpus forces:

  labels     chunk is in a gold fact's chunk set (dataset/MultiHop-RAG/processed/gold_chunks.jsonl)
  full       every gold fact has at least one of its chunks in the set
  time       anchors / publication-date chunks exactly as the +ST run (mhr_st adapter)
  structure  news articles have no conversation turns: is_continuation = chunk is not the
             article's first chunk; chunk_order, doc_chunks, tokens are unchanged in meaning
  folds      5, grouped by gold article (questions sharing an article share a fold),
             balanced per question type; null questions are their own group

Rules fixed before looking at any number (copied from coverage.py / quick_scorer.py):
  pool n*    smallest native cut in (30, 45, 60, 90, 120) whose full-evidence count is within
             ceil(2% of labelled questions) of the uncut ranking; else the uncut ranking
  scorer     StandardScaler + LogisticRegression(C=1), no grid; fold f is scored by a model
             fit on the other four folds only
  F arm      pool sorted by p (ties: native order), then native A1@4000 (stop at first overflow)

    python -m reproduce.mhr_f.study [--seed 101]
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
import statistics

import numpy as np

from reproduce.mhr_st.common import DATA, INDEX, OUT as ST_OUT, read, records, write, sha, setup
from reproduce.mhr_f.pool import OUT, c0_reference

BUDGET = 4000
FOLDS = 5
CUTS = (30, 45, 60, 90, 120)
C = 1.0
SEED = 13
FEATURES = ('native_rank', 'native_rank_log', 'in_native30', 'graph_rank', 'graph_score', 'graph_score_rel',
            'in_graph', 'in_dense30', 'dense_rank', 'date_match', 'time_boosted', 'n_seed_paths',
            'n_keyword_groups', 'is_path_head', 'is_continuation', 'chunk_order', 'doc_chunks',
            'tokens')  # identical names and order to reproduce/evidence_set/features.py


@lru_cache(maxsize=1)
def corpus():
    setup()
    from minirag.utils import encode_string_by_tiktoken
    from reproduce.mhr_st.adapter import publication_overlay
    raw = read(INDEX / 'kv_store_text_chunks.json')
    chunks = {c: {'doc': r['full_doc_id'], 'order': int(r['chunk_order_index']), 'content': r['content'],
                  'tokens': len(encode_string_by_tiktoken(r['content']))} for c, r in raw.items()}
    by_doc = {}
    for c, v in sorted(chunks.items(), key=lambda kv: (kv[1]['doc'], kv[1]['order'])):
        by_doc.setdefault(v['doc'], []).append(c)
    by_time, _ = publication_overlay(records(DATA / 'processed/docs.jsonl'), raw)
    return chunks, by_doc, by_time


@lru_cache(maxsize=1)
def gold_sets():
    """query_id -> chunk-id set per gold fact; questions without evidence (null type) are absent."""
    return {r['query_id']: [set(e['chunk_ids']) for e in r['evidence']]
            for r in records(DATA / 'processed/gold_chunks.jsonl') if r['evidence']}


def full(ids, sets):
    s = set(ids)
    return all(s & g for g in sets)


def facts_hit(ids, sets):
    s = set(ids)
    return sum(bool(s & g) for g in sets)


def a1(ids, budget=BUDGET):
    chunks, _, _ = corpus()
    t, out = 0, []
    for c in ids:
        t += chunks[c]['tokens']
        if t > budget:
            break
        out.append(c)
    return out


def folds(rows):
    """Grouped folds (same construction as evidence_set.data.folds)."""
    chunks, _, _ = corpus()
    gs = gold_sets()
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        node = 'q:' + r['query_id']
        find(node)
        for d in sorted({chunks[c]['doc'] for g in gs.get(r['query_id'], []) for c in g}):
            parent[find(node)] = find('d:' + d)
    groups = {}
    for r in rows:
        groups.setdefault(find('q:' + r['query_id']), []).append(r)
    key = lambda g: hashlib.sha256(b'mhr-f-folds-v1:' + min(r['query_id'] for r in g).encode()).hexdigest()
    load, out = {}, {}
    for g in sorted(groups.values(), key=lambda g: (-len(g), key(g))):
        t = g[0]['Type']
        f = min(range(FOLDS), key=lambda f: (load.get((t, f), 0), load.get(f, 0), f))
        for r in g:
            out[r['query_id']] = f
        load[(t, f)] = load.get((t, f), 0) + len(g)
        load[f] = load.get(f, 0) + len(g)
    return out


def variants(p):
    chunks, _, _ = corpus()
    ranked = list(dict.fromkeys(c for c in p['final_full'] if c in chunks))
    out = {f'native@{n}': ranked[:n] for n in CUTS}
    out['native@all'] = ranked
    out['all+dense30'] = list(dict.fromkeys(ranked + [c for c in p['dense_ids'] if c in chunks]))
    return out


def featurize(p, pool):
    chunks, by_doc, by_time = corpus()
    native = {c: i for i, c in enumerate(p['final_full'])}
    graph = {c: i for i, c in enumerate(p['graph_full'])}
    dense = {c: i for i, c in enumerate(p['dense_ids'])}
    top = max(p['graph_scores'].values(), default=1.0) or 1.0
    dates = {c for a in p['anchors'] for c in by_time.get(a, ())}
    boosted = {c for v in p['boosted'].values() for c in v}
    paths, groups, heads = {}, {}, set()
    for kw, seeds in p['groups'].items():
        for s in seeds:
            for c in s['path']:
                paths[c] = paths.get(c, 0) + 1
                groups.setdefault(c, set()).add(kw)
            if s['path']:
                heads.add(s['path'][0])
    far = len(pool)
    out = []
    for c in pool:
        g = p['graph_scores'].get(c, 0.0)
        f = {'native_rank': native.get(c, far), 'native_rank_log': math.log1p(native.get(c, far)),
             'in_native30': int(native.get(c, far) < 30), 'graph_rank': graph.get(c, far),
             'graph_score': g, 'graph_score_rel': g / top, 'in_graph': int(c in graph),
             'in_dense30': int(c in dense), 'dense_rank': dense.get(c, 30), 'date_match': int(c in dates),
             'time_boosted': int(c in boosted), 'n_seed_paths': paths.get(c, 0),
             'n_keyword_groups': len(groups.get(c, ())), 'is_path_head': int(c in heads),
             'is_continuation': int(chunks[c]['order'] > 0), 'chunk_order': chunks[c]['order'],
             'doc_chunks': len(by_doc[chunks[c]['doc']]), 'tokens': chunks[c]['tokens']}
        assert tuple(f) == FEATURES
        out.append((c, [f[k] for k in FEATURES]))
    return out


def exact_p(b, c):
    n = b + c
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.


def coverage(rows, pools):
    gs = gold_sets()
    lab = [r for r in rows if r['query_id'] in gs]
    names = list(variants(pools[lab[0]['query_id']]))
    table = {}
    for k in names:
        v = {r['query_id']: variants(pools[r['query_id']])[k] for r in rows}
        table[k] = {'full_evidence': sum(full(v[r['query_id']], gs[r['query_id']]) for r in lab),
                    'size_median': statistics.median(len(x) for x in v.values()),
                    'size_max': max(len(x) for x in v.values())}
    tol = math.ceil(0.02 * len(lab))
    ok = [n for n in CUTS if table[f'native@{n}']['full_evidence'] >= table['native@all']['full_evidence'] - tol]
    return {'labelled': len(lab), 'tolerance_questions': tol, 'table': table,
            'chosen_pool': f'native@{ok[0]}' if ok else 'native@all'}


def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=SEED)).fit(X, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--partial', action='store_true',
                    help='code check on the pools captured so far; writes dev_offline_partial/, never the result')
    a = ap.parse_args()
    chunks, _, _ = corpus()
    gs = gold_sets()
    src = OUT / 'dev' / f's{a.seed}'
    with (ST_OUT / 'inputs/dev.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if a.partial:
        pools = {p['query_id']: p for k in range(3) for p in records(src / f'pools_part{k}.jsonl')}
        rows = [r for r in rows if r['query_id'] in pools]
        summary = {'pools_sha256': None}
        dest = OUT / 'dev_offline_partial'
    else:
        summary = read(src / 'pools_summary.json')
        assert summary['pools_sha256'] == sha(src / 'pools.jsonl'), 'pools changed'
        pools = {p['query_id']: p for p in records(src / 'pools.jsonl')}
        assert set(pools) == {r['query_id'] for r in rows}
        dest = OUT / 'dev_offline'
    cov = coverage(rows, pools)
    write(dest / 'coverage.json', cov)
    pool_name = cov['chosen_pool']
    fold = folds(rows)

    # ---- scorer F, cross-fitted
    X, y, meta = [], [], []
    for r in rows:
        qid = r['query_id']
        if pools[qid]['status'] != 'ok':
            continue
        for c, f in featurize(pools[qid], variants(pools[qid])[pool_name]):
            X.append(f)
            y.append(int(any(c in g for g in gs.get(qid, []))))
            meta.append((qid, c, fold[qid]))
    X, y = np.array(X, float), np.array(y)
    p = {}
    from sklearn.metrics import roc_auc_score
    auc = {}
    for k in range(FOLDS):
        tr = np.array([m[2] != k for m in meta])
        te = ~tr
        model = fit(X[tr], y[tr])
        pr = model.predict_proba(X[te])[:, 1]
        for (qid, c, _), v in zip([m for m, t in zip(meta, te) if t], pr):
            p.setdefault(qid, {})[c] = float(v)
        if y[te].any() and not y[te].all():
            auc[k] = round(roc_auc_score(y[te], pr), 4)
    all_model = fit(X, y)
    coef = dict(zip(FEATURES, (round(float(v), 4) for v in all_model[-1].coef_[0])))
    with (dest / 'scores_F.jsonl').open('w', encoding='utf-8', newline='\n') as fh:
        for qid in sorted(p):
            for c, v in p[qid].items():
                fh.write(json.dumps({'query_id': qid, 'chunk_id': c, 'p': v, 'fold': fold[qid],
                                     'trained_on_folds': [g for g in range(FOLDS) if g != fold[qid]]}) + '\n')

    # ---- offline gate
    c0 = c0_reference('dev', a.seed)
    qtext = {r['query_id']: r['Question'] for r in rows}
    arms = {'C0': {}, 'ST': {}, 'F': {}, 'ORACLE': {}}
    for r in rows:
        qid = r['query_id']
        pool = variants(pools[qid])[pool_name] if pools[qid]['status'] == 'ok' else []
        arms['C0'][qid] = c0.get(qtext[qid], [])
        arms['ST'][qid] = pools[qid]['kept']
        order = sorted(range(len(pool)), key=lambda i: (-p.get(qid, {}).get(pool[i], 0.0), i))
        arms['F'][qid] = a1([pool[i] for i in order])
        g = gs.get(qid, [])
        arms['ORACLE'][qid] = a1([c for c in pool if any(c in s for s in g)] +
                                 [c for c in pool if not any(c in s for s in g)])
    lab = [r for r in rows if r['query_id'] in gs]

    def stats(arm, subset):
        ids = [r['query_id'] for r in subset]
        hit = sum(facts_hit(arms[arm][q], gs[q]) for q in ids)
        total = sum(len(gs[q]) for q in ids)
        return {'n': len(ids), 'full_evidence': sum(full(arms[arm][q], gs[q]) for q in ids),
                'facts_hit': hit, 'facts': total, 'micro_recall': round(hit / total, 4),
                'macro_recall': round(statistics.mean(facts_hit(arms[arm][q], gs[q]) / len(gs[q]) for q in ids), 4),
                'chunks_median': statistics.median(len(arms[arm][q]) for q in ids)}

    types = sorted({r['Type'] for r in lab})
    result = {'seed': a.seed, 'pool': pool_name, 'features': list(FEATURES), 'C': C,
              'auc_per_fold': auc, 'coef_all_folds': coef,
              'fold_sizes': {k: sum(v == k for v in fold.values()) for k in range(FOLDS)},
              'arms': {arm: {'all': stats(arm, lab), **{t: stats(arm, [r for r in lab if r['Type'] == t])
                                                       for t in types}} for arm in arms}}
    for x, base in (('F', 'ST'), ('ST', 'C0')):
        b = sum(full(arms[x][r['query_id']], gs[r['query_id']]) and not full(arms[base][r['query_id']], gs[r['query_id']]) for r in lab)
        c = sum(full(arms[base][r['query_id']], gs[r['query_id']]) and not full(arms[x][r['query_id']], gs[r['query_id']]) for r in lab)
        up = sum(facts_hit(arms[x][r['query_id']], gs[r['query_id']]) > facts_hit(arms[base][r['query_id']], gs[r['query_id']]) for r in lab)
        down = sum(facts_hit(arms[x][r['query_id']], gs[r['query_id']]) < facts_hit(arms[base][r['query_id']], gs[r['query_id']]) for r in lab)
        result[f'{x}_vs_{base}'] = {'full_up': b, 'full_down': c, 'full_p_exact': exact_p(b, c),
                                    'facts_up_questions': up, 'facts_down_questions': down,
                                    'facts_sign_p_exact': exact_p(up, down)}
    result['pools_sha256'] = summary['pools_sha256']
    result['api_calls'] = 0
    write(dest / 'offline_eval.json', result)
    with (dest / 'selections.jsonl').open('w', encoding='utf-8', newline='\n') as fh:
        for r in rows:
            fh.write(json.dumps({'query_id': r['query_id'], **{k: arms[k][r['query_id']] for k in arms}}) + '\n')
    print(json.dumps({k: result[k] for k in ('pool', 'auc_per_fold', 'F_vs_ST', 'ST_vs_C0')}, indent=1))
    print(json.dumps({arm: result['arms'][arm]['all'] for arm in arms}, indent=1))


if __name__ == '__main__':
    main()
