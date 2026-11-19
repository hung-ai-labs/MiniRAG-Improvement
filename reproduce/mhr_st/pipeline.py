from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .common import ARCHIVE, DATA, INDEX, JUDGE, OUT, ROOT, SEEDS, append, read, records, setup, sha, write


def rows_for(split, limit=None):
    with (OUT / f'inputs/{split}.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def reference(split, seed):
    if split == 'test':
        return ARCHIVE / f'qa/s{seed}/C0_s{seed}'
    if seed != 101:
        raise ValueError('The frozen dev C0 has seed 101 only')
    return ROOT / 'runs/multihop_rag/dev400/C0_s101'


def folder(split, seed):
    return OUT / split / f'ST_s{seed}'


def worker(args):
    setup()
    import phased_qa as phased
    import minirag.operate as op
    from minirag import QueryParam
    from .adapter import SourceTime, load_temporal
    dest = folder(args.split, args.seed)
    dest.mkdir(parents=True, exist_ok=True)
    chunk = args.shard
    out = dest / f'contexts_part{chunk}.jsonl'
    done = {r['query_id'] for r in records(out)}
    rows = rows_for(args.split, args.limit)[chunk::args.workers]
    old = {r['query_id']: r for r in records(reference(args.split, args.seed) / 'contexts.jsonl')}
    old_log = {r['query']: r for r in records(reference(args.split, args.seed) / 'context.jsonl')}
    kw = OUT / 'inputs/kw_dev_s101.jsonl' if args.split == 'dev' else OUT / f's{args.seed}/kw_s{args.seed}.jsonl'
    phased.apply_env('C0', args.seed, kw_cache=kw, cache_only=True,
                     context_log=dest / f'context_part{chunk}.jsonl')
    for key in ('MINIRAG_PATH_PRUNE', 'MINIRAG_PATH_SCORE', 'MINIRAG_RERANK', 'MINIRAG_MAX_TOKEN_TEXT_UNIT'):
        os.environ.pop(key, None)
    temporal = load_temporal(OUT / 'source/temporal_slide.py')
    docs = records(DATA / 'processed/docs.jsonl')
    chunks = read(INDEX / 'kv_store_text_chunks.json')
    from .adapter import publication_overlay
    _, years = publication_overlay(docs, chunks)
    rag = adapter = None
    validated = 0

    async def run():
        nonlocal rag, adapter, validated
        for row in rows:
            qid, question = row['query_id'], row['Question']
            if qid in done:
                continue
            anchors = temporal.parse_query(question, years)
            previous = old[qid]
            if not anchors:
                append(out, previous | {'source': 'frozen_C0_no_anchor', 'anchors': [], 'same_as_c0': True,
                                       'chunk_ids': old_log.get(question, {}).get('chunk_ids', [])})
                continue
            if rag is None:
                work = OUT / 'snapshots' / f'{args.split}_s{args.seed}_w{chunk}'
                if not work.exists():
                    phased.snapshot(work, INDEX)
                rag = phased.build(work, llm_max_async=1)
                rag.llm_model_func = phased._forbidden_llm
                adapter = SourceTime(op, temporal, docs, chunks)
            # Replay real anchored C0 queries before trusting a new runtime.
            # Hash mismatch stops the worker before any paid generation.
            if validated < 2:
                actual = await rag.aquery(question, param=QueryParam(mode='mini', only_need_context=True))
                expected = previous['context'] if previous['status'] == 'ok' else None
                if actual != expected:
                    write(dest / f'validation_failed_w{chunk}.json', {
                        'query_id': qid, 'seed': args.seed,
                        'expected_sha256': hashlib.sha256(str(expected).encode()).hexdigest(),
                        'actual_sha256': hashlib.sha256(str(actual).encode()).hexdigest(),
                        'expected_context': expected, 'actual_context': actual})
                    raise RuntimeError(f'C0 replay differs for {qid}; refusing generation')
                append(dest / f'c0_replay_w{chunk}.jsonl', {'query_id': qid, 'matches': True})
                validated += 1
            adapter.install()
            started = time.perf_counter()
            actual = await rag.aquery(question, param=QueryParam(mode='mini', only_need_context=True))
            op._build_mini_query_context = adapter.original
            from minirag.prompt import PROMPTS
            status = 'no_context' if actual is None else 'early_exit' if actual == PROMPTS['fail_response'] else 'ok'
            record = {'query_id': qid, 'status': status, 'context': actual if status == 'ok' else None,
                      'retrieval_s': round(time.perf_counter() - started, 4), 'source': 'ST_retrieval',
                      'same_as_c0': status == previous['status'] and (actual if status == 'ok' else None) == previous['context'],
                      **adapter.trace}
            ranked = [c for c in adapter.trace.get('ranked_ids', []) if c in chunks]
            selected = op.truncate_list_by_token_size(ranked, key=lambda c: chunks[c]['content'], max_token_size=4000)
            record['chunk_ids'] = selected if status == 'ok' else []
            append(out, record)
            print(f'seed {args.seed} worker {chunk}: {qid}, changed={not record["same_as_c0"]}', flush=True)
    asyncio.run(run())
    return 0


def contexts(args):
    dest = folder(args.split, args.seed)
    dest.mkdir(parents=True, exist_ok=True)
    completed = dest / 'context_summary.json'
    if completed.exists() and (dest / 'contexts.jsonl').exists():
        summary = read(completed)
        wanted = {r['query_id'] for r in rows_for(args.split, args.limit)}
        existing = {r['query_id'] for r in records(dest / 'contexts.jsonl')}
        if (summary['questions'] == len(wanted) and existing == wanted
                and summary['contexts_sha256'] == sha(dest / 'contexts.jsonl')):
            print(json.dumps({'contexts_reused': len(existing), 'seed': args.seed}), flush=True)
            return 0
    layout = {'workers': args.workers, 'split': args.split, 'seed': args.seed}
    if (dest / 'worker_layout.json').exists() and read(dest / 'worker_layout.json') != layout:
        raise ValueError('Worker layout changed; keep the original shard count when resuming')
    write(dest / 'worker_layout.json', layout)
    processes = []
    for shard in range(args.workers):
        cmd = [sys.executable, '-m', 'reproduce.mhr_st.run', 'worker', '--split', args.split,
               '--seed', str(args.seed), '--workers', str(args.workers), '--shard', str(shard)]
        if args.limit:
            cmd += ['--limit', str(args.limit)]
        log = (dest / f'worker{shard}.log').open('a', encoding='utf-8')
        processes.append(subprocess.Popen(cmd, cwd=ROOT, env=os.environ | {'PYTHONHASHSEED': '13', 'PYTHONUTF8': '1'},
                                          stdout=log, stderr=subprocess.STDOUT))
        log.close()
    codes = [p.wait() for p in processes]
    if any(codes):
        raise RuntimeError(f'Context workers failed: {codes}; inspect {dest}')
    all_rows = {r['query_id']: r for k in range(args.workers) for r in records(dest / f'contexts_part{k}.jsonl')}
    wanted = rows_for(args.split, args.limit)
    missing = [r['query_id'] for r in wanted if r['query_id'] not in all_rows]
    if missing:
        raise ValueError(f'Missing {len(missing)} contexts')
    with (dest / 'contexts.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for row in wanted:
            f.write(json.dumps(all_rows[row['query_id']], ensure_ascii=False) + '\n')
    summary = {'questions': len(wanted), 'seed': args.seed,
               'has_anchor': sum(bool(all_rows[r['query_id']].get('anchors')) for r in wanted),
               'changed_contexts': sum(not all_rows[r['query_id']]['same_as_c0'] for r in wanted),
               'c0_replays': sum(len(records(dest / f'c0_replay_w{k}.jsonl')) for k in range(args.workers)),
               'contexts_sha256': sha(dest / 'contexts.jsonl'), 'llm_calls': 0}
    write(dest / 'context_summary.json', summary)
    print(json.dumps(summary), flush=True)
    return 0


def configure_slm(phased, seed):
    """Call the frozen runner's zero-argument API and independently verify serving."""
    import httpx
    phased.use_slm()
    endpoint = os.environ['MINIRAG_SLM_URL'].rstrip('/').removesuffix('/v1')
    headers = {'Authorization': 'Bearer ' + os.environ['MINIRAG_SLM_KEY']}
    expected = read(OUT / 'source/reproduce/research/serving_qwen_modal.json')
    observed = {}
    for field, path in [('version', '/version'), ('models', '/v1/models')]:
        for attempt in range(4):
            try:
                response = httpx.get(endpoint + path, headers=headers, timeout=45)
                response.raise_for_status()
                observed[field] = response.json()
                break
            except httpx.TimeoutException:
                if attempt == 3:
                    raise RuntimeError(f'SLM {field} check timed out during cold start') from None
    names = {m['id'] for m in observed['models'].get('data', [])}
    if not {'minirag-slm', expected['model']} <= names:
        raise RuntimeError('The SLM endpoint is serving a different model')
    if observed['version'].get('version') != expected['vllm_version']:
        raise RuntimeError('The SLM endpoint changed vLLM version')
    append(OUT / 'serving_checks.jsonl', {'seed': seed, 'phase': 'answers',
           'model': expected['model'], 'vllm_version': observed['version']['version'],
           'served_models': sorted(names), 'at': datetime.datetime.now(datetime.timezone.utc).isoformat()})


def generate(split, seed):
    setup()
    import phased_qa as phased
    import llm_trace
    from minirag.base import QueryParam
    from minirag.prompt import PROMPTS
    dest = folder(split, seed)
    rows = rows_for(split)
    ctx = {r['query_id']: r for r in records(dest / 'contexts.jsonl')}
    if set(ctx) != {r['query_id'] for r in rows}:
        raise ValueError('Incomplete contexts; refusing GPU calls')
    old = {r['query_id']: r for r in records(reference(split, seed) / 'answers.jsonl')}
    done = {r['query_id']: r for r in records(dest / 'answers.jsonl')}
    for row in rows:
        qid = row['query_id']
        if qid not in done and ctx[qid]['same_as_c0']:
            value = old[qid] | {'source': 'frozen_C0_identical_context'}
            append(dest / 'answers.jsonl', value)
            done[qid] = value
    if len(done) < len(rows):
        phased.apply_env('C0', seed)
        configure_slm(phased, seed)
        async def run():
            rag = phased.build(dest / 'answer_work', llm_max_async=32)
            await phased.answer_phase(rag.llm_model_func, rows, ctx, 'mini', dest / 'answers.jsonl',
                                      QueryParam().response_type)
        phased.traced(dest, run)
    answers = {r['query_id']: r for r in records(dest / 'answers.jsonl')}
    if len(answers) != len(rows):
        raise ValueError('Incomplete answers')
    for arm, values in [('ST', answers), ('C0', old)]:
        target = dest / 'answers.csv' if arm == 'ST' else OUT / split / f'C0_s{seed}' / 'answers.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['Question', 'Gold Answer', 'minirag'])
            for row in rows:
                writer.writerow([row['Question'], row['Gold Answer'], values[row['query_id']]['answer']])
    write(dest / 'answer_summary.json', {'questions': len(rows),
          'reused_identical_context': sum(x.get('source') == 'frozen_C0_identical_context' for x in answers.values()),
          'answers_sha256': sha(dest / 'answers.jsonl')})


def judge(split, seed):
    # New process for each cohort: the Gemini key pool must not inherit the SLM
    # client or its effectively unbounded GPU rate-limiter configuration.
    for arm in ('C0', 'ST'):
        dest = OUT / split / f'{arm}_s{seed}'
        log = dest / 'judge_runner.log'
        env = os.environ.copy()
        for key in ('GEMINI_API_KEY_ONLY', 'MINIRAG_SLM_SEED', 'MINIRAG_MAX_TOKENS', 'GEMINI_RPM'):
            env.pop(key, None)
        env['GEMINI_API_BASE'] = 'https://generativelanguage.googleapis.com/v1beta/openai/'
        env['GEMINI_TPM'] = '3000'
        cmd = [sys.executable, '-m', 'reproduce.mhr_st.judge', '--questions', str(OUT / f'inputs/{split}.csv'),
               '--answers', str(dest / 'answers.csv'), '--run-dir', str(dest / 'judge_gemini35')]
        if (dest / 'judge_gemini35/manifest.json').exists():
            cmd += ['--resume']
        with log.open('a', encoding='utf-8') as f:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{arm} seed {seed}: judge incomplete; resume preserves all valid votes')


def dispatch(args):
    if args.step == 'worker':
        return worker(args)
    if args.step == 'contexts':
        return contexts(args)
    if args.step == 'answers':
        generate(args.split, args.seed)
        return 0
    if args.step == 'status':
        result = {'run': str(OUT), 'state': read(OUT / 'status.json') if (OUT / 'status.json').exists() else 'preparing',
                  'cohorts': {}}
        for split in ('dev', 'test'):
            for seed in SEEDS:
                dest = folder(split, seed)
                if not dest.exists():
                    continue
                value = {'context_records': sum(p.read_bytes().count(b'\n') for p in dest.glob('contexts_part*.jsonl'))}
                if (dest / 'context_summary.json').exists():
                    value['context_summary'] = read(dest / 'context_summary.json')
                for arm in ('C0', 'ST'):
                    journal = OUT / split / f'{arm}_s{seed}/judge_gemini35/judge.jsonl'
                    if journal.exists():
                        value[arm + '_judge_attempt_records'] = journal.read_bytes().count(b'\n')
                result['cohorts'][f'{split}_s{seed}'] = value
        print(json.dumps(result, indent=2))
        return 0
    if args.step == 'execute':
        dev = OUT / 'dev/ST_s101/context_summary.json'
        if not dev.exists() or read(dev)['questions'] < 20 or read(dev)['c0_replays'] < 4:
            raise RuntimeError('The dev20 C0 replay and ST smoke must pass before execution')
        import contextlib
        setup()
        import qa_run
        with qa_run.run_lock(OUT):
            return execute(args)
    raise ValueError(args.step)


def execute(args):
    try:
        from .run import prepare
        prepare()
        implementation = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
        identity = {'implementation_sha256': implementation, 'protocol_sha256': sha(OUT / 'protocol.json'),
                    'embedding_cache': read(OUT / 'embedding_cache.json'), 'workers': args.workers,
                    'dev_smoke_sha256': sha(OUT / 'dev/ST_s101/context_summary.json')}
        if (OUT / 'execution_identity.json').exists() and read(OUT / 'execution_identity.json') != identity:
            raise RuntimeError('Implementation or inputs changed since execution; refusing unsafe resume')
        write(OUT / 'execution_identity.json', identity)
        write(OUT / 'execution_receipt.json', {'pid': os.getpid(),
              'at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        write(OUT / 'status.json', {'state': 'running', 'step': 'contexts', 'seeds': list(SEEDS)})
        for seed in SEEDS:
            args.seed, args.split, args.limit = seed, 'test', None
            contexts(args)
        # All expensive CPU retrieval is complete before warming the GPU.
        for seed in SEEDS:
            write(OUT / 'status.json', {'state': 'running', 'step': 'answers', 'seed': seed})
            # Match the historical runner's process boundary: OpenAI/httpx clients
            # and the key-pool asyncio locks cannot cross asyncio.run event loops.
            result = subprocess.run([sys.executable, '-u', '-m', 'reproduce.mhr_st.run', 'answers',
                                     '--split', 'test', '--seed', str(seed)], cwd=ROOT)
            if result.returncode:
                raise RuntimeError(f'Answer worker seed {seed} failed; checkpoint preserved')
        for seed in SEEDS:
            write(OUT / 'status.json', {'state': 'running', 'step': 'judge', 'seed': seed})
            judge('test', seed)
        from .report import report
        report()
        write(OUT / 'status.json', {'state': 'completed', 'judge': JUDGE})
        return 0
    except BaseException as exc:
        write(OUT / 'status.json', {'state': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)})
        raise
