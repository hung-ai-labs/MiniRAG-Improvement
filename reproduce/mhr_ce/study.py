"""Registered dev400 cross-fit: C0, ST, F, CE+F, CE and ORACLE. No API calls."""
from __future__ import annotations

import argparse
import json
import statistics

from reproduce.mhr_ce import offline
from reproduce.mhr_f import study as base
from reproduce.mhr_f.pool import OUT as F_OUT
from reproduce.mhr_st.common import read, records, sha

FEATURES = ('ce_logit',) + base.FEATURES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, choices=(101,), default=101)
    a = ap.parse_args()
    from reproduce.mhr_ce.local_runtime import install
    install()
    offline.guard()
    import numpy as np
    from sklearn.metrics import roc_auc_score

    dest = offline.OUT / 'dev_offline'
    outputs = [dest / n for n in ('offline_eval.json', 'scores_CEF.jsonl', 'selections.jsonl')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('ADD-ONLY: dev outputs already exist')
    rows, pools, pool_name, ce, fold, X, y, meta, summary = offline.dev_data()
    gs = base.gold_sets()
    predictions, auc = {}, {}
    for arm, features in (('F', X[:, 1:]), ('CE+F', X)):
        predictions[arm], auc[arm] = {}, {}
        for k in range(base.FOLDS):
            train = np.array([m[2] != k for m in meta])
            test = ~train
            model = base.fit(features[train], y[train])
            pr = model.predict_proba(features[test])[:, 1]
            for (qid, c, _), value in zip([m for m, t in zip(meta, test) if t], pr):
                predictions[arm].setdefault(qid, {})[c] = float(value)
            if y[test].any() and not y[test].all():
                auc[arm][k] = round(float(roc_auc_score(y[test], pr)), 4)
    all_model = base.fit(X, y)
    coef = dict(zip(FEATURES, (round(float(v), 4) for v in all_model[-1].coef_[0])))

    # Read the already captured C0 reference directly: c0_reference() can copy a
    # missing cache into the protected +F directory, which ADD-ONLY forbids.
    c0path = F_OUT / 'inputs/dev_c0_s101_context.jsonl'
    c0 = {r['query']: r.get('chunk_ids', []) for r in records(c0path)}
    if not c0path.exists():
        raise FileNotFoundError(c0path)
    arms = {arm: {} for arm in ('C0', 'ST', 'F', 'CE+F', 'CE', 'ORACLE')}
    for r in rows:
        qid = r['query_id']
        pool = base.variants(pools[qid])[pool_name] if pools[qid]['status'] == 'ok' else []
        arms['C0'][qid] = c0.get(r['Question'], [])
        arms['ST'][qid] = pools[qid]['kept']
        for arm in ('F', 'CE+F'):
            order = sorted(range(len(pool)), key=lambda i: (-predictions[arm].get(qid, {}).get(pool[i], 0.), i))
            arms[arm][qid] = base.a1([pool[i] for i in order])
        arms['CE'][qid] = base.a1([pool[i] for i in sorted(range(len(pool)), key=lambda i: (-ce[qid][pool[i]], i))])
        gold = gs.get(qid, [])
        arms['ORACLE'][qid] = base.a1([c for c in pool if any(c in g for g in gold)] +
                                     [c for c in pool if not any(c in g for g in gold)])
    old = {r['query_id']: r['F'] for r in records(F_OUT / 'dev_offline/selections.jsonl')}
    if set(old) != set(arms['F']):
        raise AssertionError('Historical F query IDs differ')
    for qid, chosen in arms['F'].items():
        if chosen != old[qid]:
            raise AssertionError(f'Cross-fitted F selection differs: {qid}')

    lab = [r for r in rows if r['query_id'] in gs]

    def stats(arm, subset):
        ids = [r['query_id'] for r in subset]
        hit = sum(base.facts_hit(arms[arm][q], gs[q]) for q in ids)
        total = sum(len(gs[q]) for q in ids)
        return {'n': len(ids), 'full_evidence': sum(base.full(arms[arm][q], gs[q]) for q in ids),
                'facts_hit': hit, 'facts': total, 'micro_recall': round(hit / total, 4),
                'macro_recall': round(statistics.mean(base.facts_hit(arms[arm][q], gs[q]) / len(gs[q]) for q in ids), 4),
                'chunks_median': statistics.median(len(arms[arm][q]) for q in ids)}

    types = sorted({r['Type'] for r in lab})
    result = {'seed': a.seed, 'pool': pool_name, 'features': list(FEATURES), 'C': base.C,
              'auc_per_fold': auc, 'coef_all_folds': coef,
              'fold_sizes': {k: sum(v == k for v in fold.values()) for k in range(base.FOLDS)},
              'arms': {arm: {'all': stats(arm, lab), **{t: stats(arm, [r for r in lab if r['Type'] == t]) for t in types}}
                       for arm in arms}, 'F_selections_equal_reference': True,
              'F_reference_questions_checked': len(old)}
    for arm, reference in (('CE+F', 'F'), ('F', 'ST'), ('CE', 'F')):
        b = c = up = down = 0
        for r in lab:
            q = r['query_id']
            x, z = arms[arm][q], arms[reference][q]
            b += base.full(x, gs[q]) and not base.full(z, gs[q])
            c += base.full(z, gs[q]) and not base.full(x, gs[q])
            up += base.facts_hit(x, gs[q]) > base.facts_hit(z, gs[q])
            down += base.facts_hit(x, gs[q]) < base.facts_hit(z, gs[q])
        key = 'CE+F_vs_F' if arm == 'CE+F' else f'{arm}_vs_{reference}'
        result[key] = {'full_up': b, 'full_down': c, 'full_p_exact': base.exact_p(b, c),
                       'facts_up_questions': up, 'facts_down_questions': down,
                       'facts_sign_p_exact': base.exact_p(up, down)}
    result['CEF_vs_F'] = result['CE+F_vs_F']
    result.update(pools_sha256=summary['pools_sha256'], ce_logits_dev_sha256=sha(offline.OUT / 'ce_logits_dev.jsonl'), api_calls=0)
    offline.jsonl(dest / 'scores_CEF.jsonl', (
        {'query_id': qid, 'chunk_id': c, 'p': value, 'fold': fold[qid],
         'trained_on_folds': [k for k in range(base.FOLDS) if k != fold[qid]]}
        for qid in sorted(predictions['CE+F']) for c, value in predictions['CE+F'][qid].items()))
    offline.jsonl(dest / 'selections.jsonl', ({'query_id': r['query_id'], **{k: arms[k][r['query_id']] for k in arms}} for r in rows))
    offline.write(dest / 'offline_eval.json', result)
    print(json.dumps({'auc_per_fold': auc, 'CE+F_vs_F': result['CE+F_vs_F'],
                      'arms': {arm: result['arms'][arm]['all'] for arm in arms},
                      'F_reference_questions_checked': len(old)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
