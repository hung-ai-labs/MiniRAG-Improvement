"""MultiHop-RAG ablation row 3 on test: C0 + ST + F, 2,156 questions x 3 generation seeds.

F is the step-A scorer (study.py) refit once on all of dev400 (seed 101) and frozen in
model_F.json before any test question is retrieved. On test it only re-orders the ST
candidate ranking without its top_k/2 cut (pool native@all, chosen on dev by the
registered pool rule); A1@4000 then cuts Sources exactly as before. Entities, keyword
caches, index, embedding, prompt, SLM and judge are those of the +ST run.

One retrieval pass per seed both records the candidate pool (kept for rows 4-5) and
builds the F context. Checks before any GPU call:
  * the first 2 questions of every worker are also run without F and must reproduce the
    stored ST context byte for byte;
  * every question: A1(pool[:30]) must equal the ST Sources, and the F Sources in the
    context log must equal A1(F order) recomputed offline from the pool and model_F.json.

Answers: an F context identical to the ST context reuses the ST answer. Judge
(gemini-3.5-flash-lite, 3 passes): an F answer identical to the C0 or ST answer of the
same query and seed shares that answer's verdicts (LiHua Table 1 rule); only the rest
is judged, in F_s<seed>/judge_delta/.

Registered decision (same as +ST): F improves on ST iff ACC rises in all three seeds and
the exact McNemar test (F vs ST) has p < 0.05 in each.

    python -m reproduce.mhr_f.run freeze                # once, before test retrieval
    python -m reproduce.mhr_f.run execute --workers 3   # contexts + answers, all seeds
    python -m reproduce.mhr_f.run finish                # judge each seed when ready, report
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from reproduce.mhr_st.common import DATA, INDEX, JUDGE, OUT as ST_OUT, ROOT, SEEDS, append, read, records, setup, sha, write
from reproduce.mhr_f import study
from reproduce.mhr_f.pool import OUT, check_embedding, hf_cache, index_manifest, kw_cache, sha_text

MODEL = OUT / 'model_F.json'
QUESTIONS = ST_OUT / 'inputs/test.csv'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def status(name='status.json', **kw):
    write(OUT / name, {'state': 'running', 'at': now(), 'judge': JUDGE} | kw)


def test_rows():
    with QUESTIONS.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def folder(seed):
    return OUT / 'test' / f'F_s{seed}'


# ------------------------------------------------------------------ frozen scorer
def freeze(a):
    if MODEL.exists():
        print(json.dumps({'model_F': 'exists', 'sha256': sha(MODEL)}))
        return 0
    src = OUT / 'dev/s101'
    summary = read(src / 'pools_summary.json')
    assert summary['pools_sha256'] == sha(src / 'pools.jsonl'), 'dev pools changed'
    pool_name = read(OUT / 'dev_offline/coverage.json')['chosen_pool']
    gs = study.gold_sets()
    pools = records(src / 'pools.jsonl')
    X, y = [], []
    for p in pools:
        if p['status'] != 'ok':
            continue
        for c, f in study.featurize(p, study.variants(p)[pool_name]):
            X.append(f)
            y.append(int(any(c in g for g in gs.get(p['query_id'], []))))
    X, y = np.array(X, float), np.array(y)
    model = study.fit(X, y)
    sc, lr = model[0], model[-1]
    frozen = {'features': list(study.FEATURES), 'pool': pool_name, 'C': study.C, 'random_state': study.SEED,
              'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(),
              'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]),
              'trained_on': 'dev400 seed 101, all questions (no test data)', 'rows': len(y), 'positives': int(y.sum()),
              'dev_pools_sha256': summary['pools_sha256'],
              'dev_offline_eval_sha256': sha(OUT / 'dev_offline/offline_eval.json'), 'frozen_at': now()}
    gap = float(np.abs(probability(frozen, X) - model.predict_proba(X)[:, 1]).max())
    if gap > 1e-9:
        raise ValueError(f'closed-form scorer differs from sklearn by {gap}')
    frozen['closed_form_max_abs_gap'] = gap
    write(MODEL, frozen)
    print(json.dumps({'model_F': str(MODEL), 'rows': len(y), 'positives': int(y.sum()), 'gap': gap}))
    return 0


def probability(m, X):
    z = ((np.asarray(X, float) - np.array(m['mean'])) / np.array(m['scale'])) @ np.array(m['coef']) + m['intercept']
    return 1.0 / (1.0 + np.exp(-z))


def f_order(p, m):
    """Pool sorted by F probability; ties keep the native order (study.py F arm)."""
    pool = study.variants(p)[m['pool']]
    if not pool:
        return []
    feats = study.featurize(p, pool)
    pr = probability(m, [f for _, f in feats])
    return [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-pr[i], i))]


# ------------------------------------------------------------------ retrieval
def st_reference(seed):
    return {r['query_id']: r for r in records(ST_OUT / f'test/ST_s{seed}/contexts.jsonl')}


def worker(a):
    setup()
    os.environ['HF_HUB_CACHE'] = str(hf_cache())
    import phased_qa as phased
    import minirag.operate as op
    from minirag import QueryParam
    from minirag.prompt import PROMPTS
    from reproduce.mhr_st.adapter import SourceTime, load_temporal

    m = read(MODEL)
    dest = folder(a.seed)
    dest.mkdir(parents=True, exist_ok=True)
    pools_out = OUT / 'test' / f's{a.seed}' / f'pools_part{a.shard}.jsonl'
    ctx_out = dest / f'contexts_part{a.shard}.jsonl'
    log = dest / f'context_log_part{a.shard}.jsonl'
    done = {r['query_id'] for r in records(ctx_out)}
    rows = test_rows()[a.shard::a.workers][:a.limit]
    st = st_reference(a.seed)
    phased.apply_env('C0', a.seed, kw_cache=kw_cache('test', a.seed), cache_only=True, context_log=log)
    for key in ('MINIRAG_PATH_PRUNE', 'MINIRAG_PATH_SCORE', 'MINIRAG_RERANK', 'MINIRAG_MAX_TOKEN_TEXT_UNIT'):
        os.environ.pop(key, None)
    temporal = load_temporal(ST_OUT / 'source/temporal_slide.py')
    docs = records(DATA / 'processed/docs.jsonl')
    chunks = read(INDEX / 'kv_store_text_chunks.json')
    cap, state = {}, {'f': True, 'in_rank': False}

    def pool_record():
        g = cap.get('graph_full', [])
        return {'final_full': cap.get('final_full', g), 'graph_full': g, 'graph_scores': cap.get('graph_scores', {}),
                'dense_ids': cap.get('dense_ids', []), 'groups': cap.get('groups', {}),
                'boosted': cap.get('boosted', {}), 'anchors': adapter.trace['anchors']}

    def reorder(native):
        if not state['f']:
            return native
        order = f_order(pool_record(), m)
        cap['f_order'] = order
        return order

    class Capture(SourceTime):
        def rank(self, groups, vector_ids, limit, graph, pairs):
            state['in_rank'] = True
            try:
                ranked = super().rank(groups, vector_ids, limit, graph, pairs)
            finally:
                state['in_rank'] = False
            scores = self.scores(groups, vector_ids, limit)
            h = self.temporal.entity_h(graph, pairs)
            full, lists = self.temporal.rank_temporal(scores, self.by_time, self.trace['anchors'], h, vector_ids,
                                                      expand=True, h_one=False, limit=None)
            assert full[:limit] == ranked and lists == self.trace['promoted'], 'rank_temporal prefix changed'
            cap.update(final_full=full, boosted=lists)
            return reorder(ranked)

    adapter = Capture(op, temporal, docs, chunks)
    native_kwd2chunk = op.kwd2chunk

    def kwd2chunk(ent_from_query_dict, chunks_ids, chunk_nums):
        native = native_kwd2chunk(ent_from_query_dict, chunks_ids, chunk_nums)
        scores = adapter.scores(ent_from_query_dict, chunks_ids, chunk_nums)
        assert native == [c for c, _ in scores.most_common(chunk_nums)], 'kwd2chunk prefix changed'
        cap.update(graph_full=[c for c, _ in scores.most_common()],
                   graph_scores={c: float(s) for c, s in scores.items()},
                   dense_ids=list(chunks_ids), graph_limit=chunk_nums,
                   groups={k: [{'score': float(d['Score']), 'path': list(d['Path'])} for d in v]
                           for k, v in ent_from_query_dict.items()})
        # Inside ST's rank the native list is only an input; F re-orders ST's output there.
        return native if state['in_rank'] else reorder(native)

    work = OUT / 'snapshots' / f'test_s{a.seed}_w{a.shard}'
    if not work.exists():
        phased.snapshot(work, INDEX)
    rag = phased.build(work, llm_max_async=1)
    rag.llm_model_func = phased._forbidden_llm
    replays = dest / f'st_replay_w{a.shard}.jsonl'
    validated = len(records(replays))

    async def query(q, use_f):
        cap.clear()
        state['f'] = use_f
        n_log = len(records(log))
        op.kwd2chunk = kwd2chunk
        adapter.install()
        try:
            ctx = await rag.aquery(q, param=QueryParam(mode='mini', only_need_context=True))
        finally:
            op._build_mini_query_context = adapter.original
            op.kwd2chunk = native_kwd2chunk
        st_ = 'no_context' if ctx is None else 'early_exit' if ctx == PROMPTS['fail_response'] else 'ok'
        new = records(log)[n_log:]
        return ctx, st_, (new[-1]['chunk_ids'] if st_ == 'ok' and new else [])

    async def run():
        nonlocal validated
        for row in rows:
            qid, q = row['query_id'], row['Question']
            if qid in done:
                continue
            ref = st[qid]
            if validated < 2:
                # Without F the runtime must rebuild the stored ST context exactly.
                ctx, s, _ = await query(q, False)
                want = ref['context'] if ref['status'] == 'ok' else None
                if s != ref['status'] or (ctx if s == 'ok' else None) != want:
                    write(dest / f'replay_failed_w{a.shard}.json', {'query_id': qid, 'status': s,
                          'expected_sha256': sha_text(str(want)), 'actual_sha256': sha_text(str(ctx))})
                    raise RuntimeError(f'ST replay differs for {qid}; refusing to continue')
                append(replays, {'query_id': qid, 'matches': True})
                validated += 1
            started = time.perf_counter()
            ctx, s, kept = await query(q, True)
            elapsed = round(time.perf_counter() - started, 3)
            pool = pool_record()
            append(pools_out, {'query_id': qid, 'status': s, 'retrieval_s': elapsed, 'kept_F': kept,
                               'f_order_head': cap.get('f_order', [])[:30], **pool})
            append(ctx_out, {'query_id': qid, 'status': s, 'context': ctx if s == 'ok' else None,
                             'context_sha256': sha_text(ctx) if s == 'ok' else None, 'anchors': pool['anchors'],
                             'chunk_ids': kept, 'retrieval_s': elapsed,
                             'same_as_st': s == ref['status'] and (ctx if s == 'ok' else None) == (
                                 ref['context'] if ref['status'] == 'ok' else None)})
            print(json.dumps({'seed': a.seed, 'shard': a.shard, 'query_id': qid, 'status': s,
                              'pool': len(pool['final_full']), 'retrieval_s': elapsed}), flush=True)

    asyncio.run(run())
    return 0


def contexts(a):
    dest = folder(a.seed)
    pdir = OUT / 'test' / f's{a.seed}'
    dest.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)
    if (dest / 'context_summary.json').exists():
        s = read(dest / 'context_summary.json')
        if s['contexts_sha256'] == sha(dest / 'contexts.jsonl'):
            print(json.dumps({'contexts_reused': s['questions'], 'seed': a.seed}), flush=True)
            return 0
    layout = {'workers': a.workers, 'seed': a.seed, 'model_F_sha256': sha(MODEL)}
    if (dest / 'worker_layout.json').exists() and read(dest / 'worker_layout.json') != layout:
        raise ValueError('worker layout or model changed; resume with the original --workers')
    write(dest / 'worker_layout.json', layout)
    revision = check_embedding()
    procs = []
    for k in range(a.workers):
        cmd = [sys.executable, '-u', '-m', 'reproduce.mhr_f.run', 'worker', '--seed', str(a.seed),
               '--workers', str(a.workers), '--shard', str(k)]
        fh = (dest / f'worker{k}.log').open('a', encoding='utf-8')
        procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                      env=os.environ | {'PYTHONHASHSEED': '13', 'PYTHONUTF8': '1'}))
        fh.close()
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f'F context workers failed: {codes}; see {dest}/worker*.log')
    wanted = test_rows()
    ctx = {r['query_id']: r for k in range(a.workers) for r in records(dest / f'contexts_part{k}.jsonl')}
    pools = {r['query_id']: r for k in range(a.workers) for r in records(pdir / f'pools_part{k}.jsonl')}
    missing = [r['query_id'] for r in wanted if r['query_id'] not in ctx or r['query_id'] not in pools]
    if missing:
        raise ValueError(f'{len(missing)} contexts/pools missing')
    # Offline checks over every question.
    m = read(MODEL)
    st = st_reference(a.seed)
    chunks, _, _ = study.corpus()
    bad_st, bad_f, bad_status = [], [], []
    for r in wanted:
        qid = r['query_id']
        p, c = pools[qid], ctx[qid]
        if c['status'] != st[qid]['status']:
            bad_status.append(qid)
        if c['status'] != 'ok':
            continue
        if study.a1([x for x in p['final_full'][:30] if x in chunks]) != st[qid]['chunk_ids']:
            bad_st.append(qid)
        if study.a1(f_order(p, m)) != c['chunk_ids']:
            bad_f.append(qid)
    for name, rows_, path in (('contexts', ctx, dest / 'contexts.jsonl'), ('pools', pools, pdir / 'pools.jsonl')):
        with path.open('w', encoding='utf-8', newline='\n') as f:
            for r in wanted:
                f.write(json.dumps(rows_[r['query_id']], ensure_ascii=False) + '\n')
    summary = {'questions': len(wanted), 'seed': a.seed,
               'status': {s: sum(c['status'] == s for c in ctx.values()) for s in ('ok', 'no_context', 'early_exit')},
               'status_differs_from_ST': len(bad_status), 'st_sources_mismatch': len(bad_st),
               'f_sources_recompute_mismatch': len(bad_f),
               'st_replays': sum(len(records(dest / f'st_replay_w{k}.jsonl')) for k in range(a.workers)),
               'same_context_as_ST': sum(c['same_as_st'] for c in ctx.values()),
               'contexts_sha256': sha(dest / 'contexts.jsonl'), 'pools_sha256': sha(pdir / 'pools.jsonl'),
               'model_F_sha256': sha(MODEL), 'embedding_revision': revision,
               'index_manifest_sha256': index_manifest(), 'llm_calls': 0,
               'examples': {'status': bad_status[:5], 'st': bad_st[:5], 'f': bad_f[:5]}}
    if summary['index_manifest_sha256'] != read(ST_OUT / 'protocol.json')['index_manifest_sha256']:
        raise ValueError('index files changed')
    if bad_status or bad_st or bad_f or summary['st_replays'] < 2 * a.workers:
        write(dest / 'context_check_failed.json', summary)
        raise RuntimeError(f'seed {a.seed}: context checks failed: {json.dumps(summary["examples"])}')
    write(dest / 'context_summary.json', summary)
    for k in range(a.workers):  # merged copies verified; the parts only cost disk
        (pdir / f'pools_part{k}.jsonl').unlink()
        shutil.rmtree(OUT / 'snapshots' / f'test_s{a.seed}_w{k}', ignore_errors=True)
    print(json.dumps({k: v for k, v in summary.items() if k != 'examples'}), flush=True)
    return 0


# ------------------------------------------------------------------ answers
def configure_slm(phased, seed):
    """Same serving check as reproduce.mhr_st.pipeline, logged in this run's folder."""
    import httpx
    phased.use_slm()
    endpoint = os.environ['MINIRAG_SLM_URL'].rstrip('/').removesuffix('/v1')
    headers = {'Authorization': 'Bearer ' + os.environ['MINIRAG_SLM_KEY']}
    expected = read(ST_OUT / 'source/reproduce/research/serving_qwen_modal.json')
    observed = {}
    for field, path in [('version', '/version'), ('models', '/v1/models')]:
        for attempt in range(6):
            try:
                response = httpx.get(endpoint + path, headers=headers, timeout=60)
                response.raise_for_status()
                observed[field] = response.json()
                break
            except httpx.TimeoutException:
                if attempt == 5:
                    raise RuntimeError(f'SLM {field} check timed out during cold start') from None
    names = {x['id'] for x in observed['models'].get('data', [])}
    if not {'minirag-slm', expected['model']} <= names:
        raise RuntimeError('The SLM endpoint is serving a different model')
    if observed['version'].get('version') != expected['vllm_version']:
        raise RuntimeError('The SLM endpoint changed vLLM version')
    append(OUT / 'serving_checks.jsonl', {'seed': seed, 'phase': 'answers', 'model': expected['model'],
           'vllm_version': observed['version']['version'], 'served_models': sorted(names), 'at': now()})


def answers(a):
    setup()
    os.environ['HF_HUB_CACHE'] = str(hf_cache())  # rag.build loads the local embedding model
    import phased_qa as phased
    from minirag.base import QueryParam
    dest = folder(a.seed)
    if not (dest / 'context_summary.json').exists():
        raise ValueError('contexts not verified; refusing GPU calls')
    rows = test_rows()
    ctx = {r['query_id']: r for r in records(dest / 'contexts.jsonl')}
    if set(ctx) != {r['query_id'] for r in rows}:
        raise ValueError('incomplete contexts; refusing GPU calls')
    old = {r['query_id']: r for r in records(ST_OUT / f'test/ST_s{a.seed}/answers.jsonl')}
    done = {r['query_id']: r for r in records(dest / 'answers.jsonl')}
    for row in rows:
        qid = row['query_id']
        if qid not in done and ctx[qid]['same_as_st']:
            value = old[qid] | {'source': 'frozen_ST_identical_context'}
            append(dest / 'answers.jsonl', value)
            done[qid] = value
    if len(done) < len(rows):
        phased.apply_env('C0', a.seed)
        configure_slm(phased, a.seed)

        async def run():
            rag = phased.build(dest / 'answer_work', llm_max_async=32)
            await phased.answer_phase(rag.llm_model_func, rows, ctx, 'mini', dest / 'answers.jsonl',
                                      QueryParam().response_type)
        phased.traced(dest, run)
    got = {r['query_id']: r for r in records(dest / 'answers.jsonl')}
    if len(got) != len(rows):
        raise ValueError('incomplete answers')
    with (dest / 'answers.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['Question', 'Gold Answer', 'minirag'])
        for row in rows:
            w.writerow([row['Question'], row['Gold Answer'], got[row['query_id']]['answer']])
    write(dest / 'answer_summary.json', {'questions': len(rows), 'answers_sha256': sha(dest / 'answers.jsonl'),
          'reused_identical_ST_context': sum(x.get('source') == 'frozen_ST_identical_context' for x in got.values()),
          'context_status': {s: sum(x.get('context_status') == s for x in got.values())
                             for s in ('ok', 'no_context', 'early_exit', 'context_overflow')}})
    shutil.rmtree(dest / 'answer_work', ignore_errors=True)
    return 0


def execute(a):
    if not MODEL.exists():
        raise RuntimeError('run `freeze` first: model_F.json must exist before test retrieval')
    def answer(seed):
        status(step='answers', seed=seed)
        # New process: OpenAI/httpx clients and key-pool locks cannot cross event loops.
        return subprocess.run([sys.executable, '-u', '-m', 'reproduce.mhr_f.run', 'answers', '--seed', str(seed)],
                              cwd=ROOT).returncode == 0

    try:
        failed = []
        for seed in SEEDS:
            status(step='contexts', seed=seed)
            a.seed = seed
            contexts(a)
            if not answer(seed):  # e.g. SLM endpoint down: keep the CPU busy, retry at the end
                failed.append(seed)
        for _ in range(3):
            failed = [s for s in failed if not answer(s)]
            if not failed:
                break
            time.sleep(1800)
        if failed:
            raise RuntimeError(f'answer step failed for seeds {failed}; checkpoints preserved, rerun to resume')
        status(step='contexts_and_answers_done')
        return 0
    except BaseException as exc:
        write(OUT / 'status.json', {'state': 'failed', 'at': now(), 'error_type': type(exc).__name__,
                                    'message': str(exc)})
        raise


# ------------------------------------------------------------------ judge
def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def loaded(seed):
    import judge_run
    from minirag.prompt import PROMPTS
    out = {}
    for arm, path in (('C0', ST_OUT / f'test/C0_s{seed}/answers.csv'), ('ST', ST_OUT / f'test/ST_s{seed}/answers.csv'),
                      ('F', folder(seed) / 'answers.csv')):
        out[arm] = {r['query_id']: r for r in judge_run.load_rows(QUESTIONS, path, 'minirag', PROMPTS['fail_response'])}
    return out


def judge_seed(seed, rounds):
    from reproduce.mhr_st.finish import run_judge
    import judge_run
    arms = loaded(seed)
    delta = folder(seed) / 'judge_delta'
    qfields, questions = read_csv(QUESTIONS)
    afields, fans = read_csv(folder(seed) / 'answers.csv')
    pick = [i for i, q in enumerate(questions)
            if arms['F'][q['query_id']]['pred'] not in (arms['C0'][q['query_id']]['pred'], arms['ST'][q['query_id']]['pred'])]
    qpath, apath = delta / 'questions.csv', delta / 'answers.csv'
    if not qpath.exists():
        write_csv(qpath, qfields, [questions[i] for i in pick])
        write_csv(apath, afields, [fans[i] for i in pick])
    run_judge(qpath, apath, delta / 'run', rounds)
    # Compose F verdicts: C0 journal, ST delta journal or F delta journal (one judgment per prediction).
    journals = {'C0': judge_run.read_records(ST_OUT / f'test/C0_s{seed}/judge_gemini35/judge.jsonl'),
                'ST': judge_run.read_records(ST_OUT / f'test/ST_s{seed}/judge_delta/run/judge.jsonl'),
                'F': judge_run.read_records(delta / 'run/judge.jsonl')}
    rows = [arms['F'][q['query_id']] for q in questions]
    latest, origin = {}, {}
    for r in rows:
        qid, pred = r['query_id'], r['pred']
        src = ('C0' if pred == arms['C0'][qid]['pred'] else
               'ST' if pred == arms['ST'][qid]['pred'] else 'F')
        origin[qid] = {'C0': 'C0_identical_answer', 'ST': 'ST_identical_answer', 'F': 'judged_F'}[src]
        for p in (1, 2, 3):
            rec = journals[src].get((qid, p))
            if rec is None or rec['status'] != 'ok':
                raise RuntimeError(f'seed {seed}: missing verdict {qid} pass {p} ({src})')
            latest[(qid, p)] = rec
    target = folder(seed) / 'judge_gemini35'
    target.mkdir(parents=True, exist_ok=True)
    with (target / 'per_question.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['query_id', 'Type', 'pass1', 'pass2', 'pass3', 'majority', 'verdict_origin'])
        for r in rows:
            labels = [latest[(r['query_id'], p)]['label'] for p in (1, 2, 3)]
            w.writerow([r['query_id'], r['type'], *labels, judge_run.majority(labels), origin[r['query_id']]])
    counts = {k: sum(v == k for v in origin.values()) for k in ('C0_identical_answer', 'ST_identical_answer', 'judged_F')}
    write(target / 'summary.json', {'status': 'completed', 'n_questions': len(rows), 'passes': 3,
                                    'groups': judge_run.summarize(rows, latest, 3), 'verdict_origin': counts})
    write(target / 'provenance.json', {'judge_model': JUDGE, 'composed_at': now(), 'verdict_origin': counts,
          'f_answers_sha256': sha(folder(seed) / 'answers.csv'),
          'delta_journal_sha256': sha(delta / 'run/judge.jsonl')})
    return counts


def finish(a):
    setup()
    try:
        for seed in SEEDS:
            while not (folder(seed) / 'answer_summary.json').exists():
                status('status_judge.json', step='waiting_for_answers', seed=seed)
                time.sleep(600)
            if not (folder(seed) / 'judge_gemini35/summary.json').exists():
                status('status_judge.json', step='judge_F_delta', seed=seed)
                counts = judge_seed(seed, a.rounds)
                print(json.dumps({'seed': seed, 'verdict_origin': counts}), flush=True)
        status('status_judge.json', step='report')
        report()
        write(OUT / 'status_judge.json', {'state': 'completed', 'at': now(), 'judge': JUDGE})
        return 0
    except BaseException as exc:
        write(OUT / 'status_judge.json', {'state': 'failed', 'at': now(), 'error_type': type(exc).__name__,
                                    'message': str(exc)})
        raise


# ------------------------------------------------------------------ report
def report():
    import statistics
    from reproduce.mhr_st.report import paired, quality
    gold = {r['query_id']: r for r in records(DATA / 'processed/gold_chunks.jsonl')}
    queries = {r['query_id']: r['Question'] for r in test_rows()}
    result = {'judge': JUDGE, 'n': len(queries), 'seeds': {}, 'model_F_sha256': sha(MODEL),
              'sd_definition': 'sample SD between generation seeds'}
    by_seed = {'C0': [], 'ST': [], 'F': []}
    for seed in SEEDS:
        arms = {}
        for arm, path in (('C0', ST_OUT / f'test/C0_s{seed}/judge_gemini35'), ('ST', ST_OUT / f'test/ST_s{seed}/judge_gemini35'),
                          ('F', folder(seed) / 'judge_gemini35')):
            s = read(path / 'summary.json')
            if s['status'] != 'completed' or s['n_questions'] != len(queries):
                raise RuntimeError(f'{arm} seed {seed}: judge incomplete')
            with (path / 'per_question.csv').open(encoding='utf-8', newline='') as f:
                arms[arm] = {r['query_id']: r for r in csv.DictReader(f)}
            by_seed[arm].append(quality(list(arms[arm].values())))
        sr = {'quality': {k: quality(list(v.values())) for k, v in arms.items()},
              'verdict_origin': read(folder(seed) / 'judge_gemini35/summary.json')['verdict_origin'],
              'F_vs_ST': paired(arms['ST'], arms['F']), 'F_vs_C0': paired(arms['C0'], arms['F']), 'by_type': {}}
        for typ in ('comparison', 'inference', 'temporal', 'null'):
            sub = {k: {q: r for q, r in v.items() if r['Type'] == typ} for k, v in arms.items()}
            sr['by_type'][typ] = {'quality': {k: quality(list(v.values())) for k, v in sub.items()},
                                  'F_vs_ST': paired(sub['ST'], sub['F'])}
        old = {r['query']: r for r in records(ST_OUT / f'archive_c0/qa/s{seed}/C0_s{seed}/context.jsonl')}
        sel = {'C0': {k: old.get(q, {}).get('chunk_ids', []) for k, q in queries.items()},
               'ST': {r['query_id']: r['chunk_ids'] for r in records(ST_OUT / f'test/ST_s{seed}/contexts.jsonl')},
               'F': {r['query_id']: r['chunk_ids'] for r in records(folder(seed) / 'contexts.jsonl')}}
        recall = {}
        for arm, chosen in sel.items():
            hits = total = complete = n = 0
            macro = []
            for qid, ids in chosen.items():
                facts = gold[qid]['evidence']
                if not facts:
                    continue
                found = sum(bool(set(ids) & set(e['chunk_ids'])) for e in facts)
                hits, total, complete, n = hits + found, total + len(facts), complete + (found == len(facts)), n + 1
                macro.append(found / len(facts))
            recall[arm] = {'n': n, 'facts_hit': hits, 'gold_facts': total, 'micro': hits / total,
                           'macro': statistics.mean(macro), 'full': complete / n}
        sr['evidence_recall'] = recall
        result['seeds'][str(seed)] = sr
    result['generation_mean_sd'] = {arm: {k: {'mean': statistics.mean(r[k] for r in v), 'sd': statistics.stdev(r[k] for r in v)}
                                          for k in v[0]} for arm, v in by_seed.items()}
    result['F_vs_ST_all_three_seeds_positive_and_significant'] = all(
        s['F_vs_ST']['up'] > s['F_vs_ST']['down'] and s['F_vs_ST']['p_exact'] < .05 for s in result['seeds'].values())
    manifest = read(INDEX / 'index_manifest.json')
    result['index_unchanged'] = all(sha(INDEX / p) == h for p, h in manifest['file_sha256'].items())
    if not result['index_unchanged']:
        raise RuntimeError('index changed during experiment')
    write(OUT / 'RESULTS.json', result)
    t = ['MultiHopRAG: C0 vs +ST vs +F', f'Qwen2.5-3B | Judge: {JUDGE}',
         '2,156 questions; 3 generation seeds; 3 judge passes per answer.',
         'ACC/ERR/Neither use majority labels; +/- is SD between generation seeds.', '',
         '| Configuration | ACC mean +/- SD | ERR | Neither | No majority | Evidence recall (micro) | Full evidence |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('C0', 'ST', 'F'):
        g = result['generation_mean_sd'][arm]
        rec = statistics.mean(s['evidence_recall'][arm]['micro'] for s in result['seeds'].values())
        fu = statistics.mean(s['evidence_recall'][arm]['full'] for s in result['seeds'].values())
        name = {'C0': 'C0', 'ST': 'C0 + ST', 'F': 'C0 + ST + F'}[arm]
        t.append(f"| {name} | {g['accurate']['mean']:.2f} +/- {g['accurate']['sd']:.2f} | {g['error']['mean']:.2f} | "
                 f"{g['neither']['mean']:.2f} | {g['no_majority']['mean']:.2f} | {rec:.3f} | {100 * fu:.1f}% |")
    for key, title in (('F_vs_ST', 'F vs ST'), ('F_vs_C0', 'F vs C0')):
        t += ['', f'{title}:', '| Seed | Delta ACC (pp) | Wrong -> correct | Correct -> wrong | Exact McNemar p |',
              '|---|---:|---:|---:|---:|']
        for seed, v in result['seeds'].items():
            p = v[key]
            t.append(f"| {seed} | {p['delta_acc_pp']:+.2f} | {p['up']} | {p['down']} | {p['p_exact']:.6g} |")
    ok = result['F_vs_ST_all_three_seeds_positive_and_significant']
    t += ['', 'F over ST: positive and significant in all three seeds.' if ok else
          'F over ST: improvement is not confirmed under the registered three-seed rule.', '',
          'Subgroups, verdict origins and judge variation: RESULTS.json.', 'Frozen index checksums verified unchanged.']
    (OUT / 'BANG_SO_SANH.txt').write_text('\n'.join(t) + '\n', encoding='utf-8')
    print('\n'.join(t))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('freeze', 'worker', 'contexts', 'answers', 'execute', 'finish', 'report'))
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--limit', type=int, default=None, help='worker only: smoke on the first N of the shard')
    a = ap.parse_args()
    if a.step == 'report':
        setup()
        report()
        return 0
    return {'freeze': freeze, 'worker': worker, 'contexts': contexts, 'answers': answers,
            'execute': execute, 'finish': finish}[a.step](a)


if __name__ == '__main__':
    raise SystemExit(main())
