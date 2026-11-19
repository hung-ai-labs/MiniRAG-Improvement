"""Row 5 (+set): offline freeze/contexts and operator-only online steps.

Codex executes only freeze and contexts. answers/execute/finish/report are
implemented for the experiment operator and are never invoked by this task.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics

from reproduce.mhr_ce import run as ce
from reproduce.mhr_f import study as base
from reproduce.mhr_f.pool import OUT as F_OUT
from reproduce.mhr_set import offline, study
from reproduce.mhr_set.select import facets, select
from reproduce.mhr_st.common import INDEX, ROOT, SEEDS, read, records, sha

OUT = offline.OUT
PARAMS = OUT / 'set_params.json'
now = ce.now
test_rows = ce.test_rows
read_csv = ce.read_csv


def folder(seed):
    return OUT / f'test/SET_s{seed}'


def token_stats(values):
    values = list(values)
    return {'n': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values)}


def provenance():
    return {**study.hashes(), 'offline_eval_sha256': sha(OUT / 'dev_offline/offline_eval.json')}


def frozen():
    value = read(PARAMS)
    for key, expected in provenance().items():
        if value[key] != expected:
            raise ValueError(f'Frozen SET provenance changed: {key}')
    if value['beta'] not in study.BETAS or value['tau'] not in study.TAUS:
        raise ValueError('Parameters outside registered grid')
    return value


def freeze(a):
    if PARAMS.exists():
        raise FileExistsError('ADD-ONLY: set_params.json already exists; refusing to retune')
    evaluation = read(OUT / 'dev_offline/offline_eval.json')
    if not evaluation['CE+F_selections_equal_reference'] or evaluation['CE+F_reference_questions_checked'] != 400:
        raise ValueError('Dev CE+F replay check missing')
    for key, expected in study.hashes().items():
        if evaluation[key] != expected:
            raise ValueError(f'Dev input changed: {key}')
    rows, pools, p, fold, fmap, chunks = study.data()
    lab, grid = study.grid_results(rows, pools, p, fmap, chunks)
    table = study.grid_table(lab, grid)
    if table != evaluation['grid']:
        raise ValueError('Recomputed full dev grid differs from step A')
    beta, tau = study.choose(lab, grid)
    value = {'beta': beta, 'tau': tau, 'grid': table, **provenance(), 'frozen_at': now(),
             'pool': 'native@all', 'budget': base.BUDGET, 'labelled': len(lab),
             'trained_on': 'All labelled dev400 seed-101 questions, cross-fitted CE+F probabilities only',
             'rule': 'Within ceil(1% of labelled n) of best full count; mean tokens, -full, (beta,tau)'}
    offline.write(PARAMS, value)
    print(json.dumps({'beta': beta, 'tau': tau, 'labelled': len(lab), 'frozen_at': value['frozen_at']}), flush=True)
    return 0


def ce_probabilities(pool_record, model, logits):
    """Expose p from mhr_ce.run.ce_order's exact features and closed-form scorer."""
    if model['features'] != list(ce.FEATURES) or model['pool'] != 'native@all':
        raise ValueError('Frozen CE model specification differs from protocol')
    pool = base.variants(pool_record)[model['pool']]
    if not pool:
        return pool, {}, []
    qid = pool_record['query_id']
    features = [[logits[qid][c]] + f for c, f in base.featurize(pool_record, pool)]
    pr = ce.probability(model, features)
    p = {c: float(v) for c, v in zip(pool, pr)}
    if len(p) != len(pool) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
        raise ValueError('Invalid test probabilities')
    order = [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-pr[i], i))]
    if order != ce.ce_order(pool_record, model, logits):
        raise AssertionError('Exposed probabilities differ from CE order machinery')
    return pool, p, order


def contexts(a):
    dest = folder(a.seed)
    if not PARAMS.exists():
        raise RuntimeError('Freeze set_params.json before building test contexts')
    if any((dest / n).exists() for n in ('contexts.jsonl', 'context_summary.json', 'context_check_failed.json')):
        raise FileExistsError('ADD-ONLY: SET context outputs already exist')
    qid = None
    try:
        params = frozen()
        model = read(study.MODEL)
        if model['ce_logits_dev_sha256'] != sha(ce.OUT / 'ce_logits_dev.jsonl'):
            raise ValueError('CE dev logits changed')
        if model['offline_eval_sha256'] != sha(ce.OUT / 'dev_offline/offline_eval.json'):
            raise ValueError('CE dev evaluation changed')
        cpath = ce.folder(a.seed) / 'contexts.jsonl'
        ppath = F_OUT / f'test/s{a.seed}/pools.jsonl'
        summary = read(ce.folder(a.seed) / 'context_summary.json')
        expected = {'contexts_sha256': sha(cpath), 'pools_sha256': sha(ppath),
                    'model_CE_sha256': sha(study.MODEL), 'chunks_sha256': sha(INDEX / 'kv_store_text_chunks.json'),
                    'index_manifest_sha256': sha(INDEX / 'index_manifest.json'),
                    'ce_logits_test_sha256': sha(ce.OUT / 'ce_logits_test.jsonl'),
                    'ce_logits_test_summary_sha256': sha(ce.OUT / 'ce_logits_test_summary.json')}
        if not summary['checks']['passed'] or any(summary[k] != v for k, v in expected.items()):
            raise ValueError('Stored CE reference checksums/checks differ')
        rows = test_rows()
        wanted = {r['query_id'] for r in rows}
        cr, pr = records(cpath), records(ppath)
        refs, pools = {r['query_id']: r for r in cr}, {r['query_id']: r for r in pr}
        if len(rows) != 2156 or len(wanted) != len(rows) or len(cr) != len(rows) or len(pr) != len(rows):
            raise ValueError('Incomplete or duplicate test rows')
        if set(refs) != wanted or set(pools) != wanted:
            raise ValueError('CE contexts/pools and test query IDs differ')
        logits = ce.offline.logits('test')  # read-only checksum and merged-score validation
        if set(logits) != wanted:
            raise ValueError('Test CE query IDs differ')
        chunks, _, by_time = base.corpus()
        texts = read(ce.OUT / 'chunks_text.json')
        if set(texts) != set(chunks) or any(texts[c] != v['content'] for c, v in chunks.items()):
            raise ValueError('CE chunk contents differ from index')
        results, checked = [], 0
        for row in rows:
            qid = row['query_id']
            ref, record = refs[qid], pools[qid]
            if ref['status'] == 'ok':
                if record['status'] != 'ok':
                    raise ValueError(f'CE/pool status mismatch: {qid}')
                pool, p, order = ce_probabilities(record, model, logits)
                if base.a1(order) != ref['chunk_ids']:
                    raise AssertionError(f'CE source selection mismatch: {qid}')
                replay = ce.rebuild(ref['context'], ref['chunk_ids'], chunks)
                if replay.encode('utf-8') != ref['context'].encode('utf-8'):
                    raise AssertionError(f'CE context byte replay mismatch: {qid}')
                checked += 1
                ids, info = select(pool, p, facets(record, pool, by_time), chunks,
                                   beta=params['beta'], tau=params['tau'])
                tokens = sum(chunks[c]['tokens'] for c in ids)
                if tokens != info['tokens'] or tokens > base.BUDGET or len(ids) != len(set(ids)) or not set(ids) <= set(pool):
                    raise AssertionError(f'Invalid SET selection: {qid}')
                context = ce.rebuild(ref['context'], ids, chunks)
            else:
                ids, context, tokens = [], ref['context'], 0
            results.append({'query_id': qid, 'status': ref['status'], 'context': context, 'chunk_ids': ids,
                            'same_as_ce': context.encode('utf-8') == ref['context'].encode('utf-8'), 'tokens': tokens})
        result = {'questions': len(rows), 'seed': a.seed, 'beta': params['beta'], 'tau': params['tau'],
                  'status': {s: sum(r['status'] == s for r in results) for s in sorted({r['status'] for r in results})},
                  'same_as_ce': sum(r['same_as_ce'] for r in results),
                  'same_as_ce_ok': sum(r['same_as_ce'] and r['status'] == 'ok' for r in results),
                  'checks': {'ok_questions_checked': checked, 'ce_sources_recompute_mismatch': 0,
                             'ce_context_byte_rebuild_mismatch': 0, 'selection_budget_violations': 0,
                             'selection_duplicate_violations': 0, 'selection_outside_pool': 0, 'passed': True},
                  'tokens': token_stats(r['tokens'] for r in results),
                  'tokens_ok': token_stats(r['tokens'] for r in results if r['status'] == 'ok') if checked else None,
                  'chunks': token_stats(len(r['chunk_ids']) for r in results),
                  'ce_contexts_sha256': sha(cpath), 'ce_context_summary_sha256': sha(ce.folder(a.seed) / 'context_summary.json'),
                  'set_params_sha256': sha(PARAMS), **{k: v for k, v in expected.items() if k != 'contexts_sha256'},
                  'llm_calls': 0}
        offline.jsonl(dest / 'contexts.jsonl', results)
        result['contexts_sha256'] = sha(dest / 'contexts.jsonl')
        offline.write(dest / 'context_summary.json', result)
        print(json.dumps(result), flush=True)
        return 0
    except BaseException as exc:
        offline.write(dest / 'context_check_failed.json', {'seed': a.seed, 'query_id': qid, 'at': now(),
                      'error_type': type(exc).__name__, 'message': str(exc)})
        raise


def answers(a):
    from reproduce.mhr_set.deferred import answers as action
    return action(a)


def execute(a):
    from reproduce.mhr_set.deferred import execute as action
    return action(a)


def finish(a):
    from reproduce.mhr_set.deferred import finish as action
    return action(a)


def report():
    from reproduce.mhr_set.reporting import report as action
    return action()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('freeze', 'contexts', 'answers', 'execute', 'finish', 'report'))
    parser.add_argument('--seed', type=int, choices=SEEDS, default=101)
    parser.add_argument('--rounds', type=int, default=6)
    a = parser.parse_args()
    if a.step in ('freeze', 'contexts', 'report'):
        offline.install()
    else:
        import platform
        platform.uname()
        offline.setup()
        offline.exclusive_library_io()
        offline.guard(network=True)
    if a.step == 'report':
        report()
        return 0
    return {'freeze': freeze, 'contexts': contexts, 'answers': answers, 'execute': execute, 'finish': finish}[a.step](a)


if __name__ == '__main__':
    raise SystemExit(main())
