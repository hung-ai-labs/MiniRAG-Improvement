"""Three-phase Phi C0, using the frozen MiniRAG QA runner."""

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from . import kwguard, profile as p, runtime
from .launch import C0_MODULE, install_subprocess_rewrite

WORKERS = 3
HASH_SEED = '13'
OFFLINE = p.OUT / '_offline_test'
COHORTS = (('dev', 101), *(('test', seed) for seed in runtime.SEEDS))


@dataclass(frozen=True)
class Cohort:
    split: str
    seed: int
    pilot_dir: Path | None = None

    @property
    def dest(self):
        if self.pilot_dir:
            return self.pilot_dir / 'C0_s101'
        if self.split == 'dev':
            return p.DEV_C0
        return p.ARCHIVE / f'qa/s{self.seed}/C0_s{self.seed}'

    @property
    def index(self):
        return p.PILOT if self.pilot_dir else p.INDEX

    @property
    def keywords(self):
        if self.pilot_dir:
            return self.pilot_dir / 'synthetic_kw.jsonl'
        if self.split == 'dev':
            return p.ST / 'inputs/kw_dev_s101.jsonl'
        return p.ST / f's{self.seed}/kw_s{self.seed}.jsonl'

    def rows(self):
        rows = runtime.questions(self.split)
        return rows[:5] if self.pilot_dir else rows


def write_json(path, value, *, fresh=False):
    if fresh:
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
    else:
        runtime.phased_runner().qa_run.atomic_json(path, value)


def write_records(path, rows, *, fresh=False):
    target = path if fresh else path.with_suffix('.jsonl.tmp')
    with target.open('x' if fresh else 'w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    if not fresh:
        os.replace(target, path)


def artifact_hashes(root, names):
    return {name: runtime.sha(root / name) for name in names}


def require_hashes(root, hashes):
    if artifact_hashes(root, hashes) != hashes:
        raise RuntimeError(f'C0 checkpoint files changed in {root}')


def identity(cohort, phased):
    return {'split': cohort.split, 'seed': cohort.seed, 'index': str(cohort.index),
            'index_manifest_sha256': runtime.sha(cohort.index / 'index_manifest.json'),
            'questions_sha256': runtime.QUESTION_HASHES[cohort.split],
            'frozen_source_sha256': runtime.sha(p.OUT / 'frozen_source_sha256.json'),
            'serving_spec_sha256': runtime.sha(p.SERVING), 'model': p.MODEL,
            'mode': phased.MODE['C0'], 'environment': phased.config_env('C0'),
            'python_hash_seed': int(HASH_SEED), 'workers': WORKERS, 'llm_max_async': 32,
            'questions': len(cohort.rows())}


def prepare(cohort, phased):
    cohort.dest.mkdir(parents=True, exist_ok=True)
    runtime.source_checks()
    runtime.verify_index(cohort.index, pilot=bool(cohort.pilot_dir))
    expected = identity(cohort, phased)
    file = cohort.dest / 'run_note.json'
    if file.exists():
        if runtime.read(file) != expected:
            raise RuntimeError('C0 run identity changed')
    else:
        write_json(file, expected, fresh=bool(cohort.pilot_dir))


def require_keyword_cache(cohort, phased):
    if not cohort.pilot_dir:
        receipt = runtime.read(cohort.dest / 'keywords_summary.json')
        if receipt['sha256'] != runtime.sha(cohort.keywords):
            raise RuntimeError('Frozen Phi keyword cache changed')
    cached = phased.read_jsonl(cohort.keywords, key='query_id')
    wanted = {r['query_id'] for r in cohort.rows()}
    if cached.keys() != wanted or not all(
        {'prompt_sha256', 'result', 'query_id'} == row.keys() for row in cached.values()
    ):
        raise RuntimeError('Frozen keyword cache has incomplete or unexpected records')


def keywords(cohort, phased):
    receipt = cohort.dest / 'keywords_summary.json'
    if receipt.exists():
        require_keyword_cache(cohort, phased)
        print(f'{cohort.split} s{cohort.seed}: keyword cache already complete', flush=True)
        return
    rows = cohort.rows()
    cohort.keywords.parent.mkdir(parents=True, exist_ok=True)
    phased.apply_env('C0', cohort.seed)
    trace_dir = cohort.dest / 'keywords'
    trace_dir.mkdir(exist_ok=True)
    serving = runtime.check_serving()
    phased.apply_env('C0', cohort.seed, kw_cache=cohort.keywords)
    write_json(trace_dir / 'serving_check.json', serving)
    snapshot = trace_dir / 'snapshot'
    if not snapshot.exists():
        phased.snapshot(snapshot, cohort.index)
    # Index input files are immutable. The runner's private log/cache can append.
    fixed = runtime.read(cohort.index / 'index_manifest.json')['file_sha256']
    require_hashes(snapshot, {name: value for name, value in fixed.items()
                              if name != 'kv_store_llm_response_cache.json'})

    async def run():
        rag = phased.build(snapshot, llm_max_async=32)
        await phased.keyword_phase(rag, rows, cohort.keywords)

    phased.traced(trace_dir, run)
    if phased.read_jsonl(cohort.keywords, key='query_id').keys() != {row['query_id'] for row in rows}:
        raise RuntimeError('Keyword phase incomplete; rerun to resume')
    write_json(receipt, {'questions': len(rows), 'sha256': runtime.sha(cohort.keywords)})
    print(f'{cohort.split} s{cohort.seed}: {len(rows)} keyword records', flush=True)


async def _phase3_c0(phased, rag, rows, output):
    """Replay the C0 half of the phase-3 shared prefix/suffix worker."""
    import time
    from minirag import operate
    from minirag.base import QueryParam
    from minirag.prompt import PROMPTS
    from research.mhr_phase3.shared_context import SharedContext, context_record

    adapter = SharedContext(operate)
    done = phased.read_jsonl(output)
    with adapter.installed(), output.open('a', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            if row['query_id'] in done:
                continue
            adapter.clear()
            started = time.perf_counter()
            value = await rag.aquery(row['Question'], param=QueryParam(mode='mini', only_need_context=True))
            record = context_record(row['query_id'], value, PROMPTS['fail_response'],
                                    time.perf_counter() - started)
            phased.append_jsonl(handle, record | kwguard.fields(value, include_context=True))


def context_worker(cohort, phased, shard):
    if os.environ.get('PYTHONHASHSEED') != HASH_SEED:
        raise RuntimeError('C0 context worker requires PYTHONHASHSEED=13 set before startup')
    for name in runtime.RETRIEVAL_OVERRIDES:
        os.environ.pop(name, None)
    dest = cohort.dest / f'worker{shard}'
    snapshot = dest / 'snapshot'
    if not snapshot.exists():
        phased.snapshot(snapshot, cohort.index)
    require_hashes(snapshot, runtime.read(cohort.index / 'index_manifest.json')['file_sha256'])
    phased.apply_env('C0', cohort.seed, kw_cache=cohort.keywords, cache_only=True,
                     context_log=dest / 'context.jsonl')
    rag = phased.build(snapshot, llm_max_async=1)
    rows = cohort.rows()[shard::WORKERS]
    output = dest / 'contexts.jsonl'
    if cohort.split == 'test':
        asyncio.run(_phase3_c0(phased, rag, rows, output))
    else:
        asyncio.run(phased.context_phase(rag, rows, phased.MODE['C0'], output))


def child_command(step, cohort):
    args = [sys.executable, '-B', '-u', '-m', C0_MODULE, step, '--split', cohort.split,
            '--seed', str(cohort.seed)]
    if cohort.pilot_dir:
        args += ['--pilot-dir', str(cohort.pilot_dir)]
    return args


def contexts(cohort, phased):
    require_keyword_cache(cohort, phased)
    receipt = cohort.dest / 'contexts_summary.json'
    kw_hash = runtime.sha(cohort.keywords)
    if receipt.exists():
        summary = runtime.read(receipt)
        if summary['keyword_sha256'] != kw_hash:
            raise RuntimeError('Frozen keyword cache changed since phase B')
        require_hashes(cohort.dest, summary['files_sha256'])
        if 'keyword_guard_count' not in summary:
            records = phased.read_jsonl(cohort.dest / 'contexts.jsonl')
            summary['keyword_guard_count'] = kwguard.count(records.values())
            write_json(receipt, summary)
        print(f'{cohort.split} s{cohort.seed}: contexts already complete', flush=True)
        return
    frozen_file = cohort.dest / 'context_identity.json'
    if frozen_file.exists() and runtime.read(frozen_file)['keyword_sha256'] != kw_hash:
        raise RuntimeError('Frozen keyword cache changed during phase B')
    if not frozen_file.exists():
        write_json(frozen_file, {'keyword_sha256': kw_hash}, fresh=bool(cohort.pilot_dir))

    processes = []
    try:
        for shard in range(WORKERS):
            path = cohort.dest / f'worker{shard}'
            path.mkdir(exist_ok=True)
            with (path / 'worker.log').open('a', encoding='utf-8') as handle:
                processes.append(subprocess.Popen(
                    child_command('context-worker', cohort) + ['--shard', str(shard)], cwd=p.ROOT,
                    env=os.environ | {'PYTHONHASHSEED': HASH_SEED, 'PYTHONUTF8': '1'},
                    stdout=handle, stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    if any(codes):
        raise RuntimeError(f'C0 context workers failed: {codes}. See {cohort.dest}/worker*/worker.log')

    rows = cohort.rows()
    parts, logs = {}, []
    for shard in range(WORKERS):
        dest = cohort.dest / f'worker{shard}'
        part = phased.read_jsonl(dest / 'contexts.jsonl')
        expected = {row['query_id'] for row in rows[shard::WORKERS]}
        if part.keys() != expected:
            raise RuntimeError(f'C0 context shard {shard} has incomplete or unexpected rows')
        parts.update(part)
        shard_logs = phased.read_jsonl(dest / 'context.jsonl', key='query')
        logs.extend(shard_logs.values())
    merged = [parts[row['query_id']] for row in rows]
    write_records(cohort.dest / 'contexts.jsonl', merged, fresh=bool(cohort.pilot_dir))
    write_records(cohort.dest / 'context.jsonl', logs, fresh=bool(cohort.pilot_dir))
    summary = {'questions': len(merged), 'statuses': dict(Counter(row['status'] for row in merged)),
               'keyword_guard_count': kwguard.count(merged),
               'retrieval_log_records': len(logs), 'llm_calls': 0, 'worker_exit_codes': codes,
               'kw_cache_only': True, 'keyword_sha256': kw_hash,
               'files_sha256': artifact_hashes(cohort.dest, ('contexts.jsonl', 'context.jsonl'))}
    write_json(receipt, summary, fresh=bool(cohort.pilot_dir))
    print(json.dumps(summary, indent=2), flush=True)


def answers(cohort, phased):
    from minirag.base import QueryParam

    receipt = cohort.dest / 'answers_summary.json'
    if receipt.exists():
        require_completed(cohort.split, cohort.seed)
        print(f'{cohort.split} s{cohort.seed}: answers already complete', flush=True)
        return
    summary = runtime.read(cohort.dest / 'contexts_summary.json')
    require_keyword_cache(cohort, phased)
    if summary['keyword_sha256'] != runtime.sha(cohort.keywords):
        raise RuntimeError('Frozen keyword cache no longer matches the contexts')
    require_hashes(cohort.dest, summary['files_sha256'])
    rows = cohort.rows()
    retrieved = phased.read_jsonl(cohort.dest / 'contexts.jsonl')
    wanted = {row['query_id'] for row in rows}
    if retrieved.keys() != wanted:
        raise RuntimeError('Incomplete contexts; refusing generation')
    phased.apply_env('C0', cohort.seed)
    write_json(cohort.dest / 'serving_check.json', runtime.check_serving())

    async def run():
        rag = phased.build(cohort.dest / 'answer_work', llm_max_async=32)
        await phased.answer_phase(rag.llm_model_func, rows, retrieved, phased.MODE['C0'],
                                  cohort.dest / 'answers.jsonl', QueryParam().response_type)

    phased.traced(cohort.dest, run)
    generated = phased.read_jsonl(cohort.dest / 'answers.jsonl')
    if generated.keys() != wanted:
        raise RuntimeError('Answer phase incomplete; rerun to resume')
    write_json(receipt, {'questions': len(generated),
                         'files_sha256': artifact_hashes(cohort.dest, ('answers.jsonl', 'answers.csv', 'requests.jsonl')),
                         'contexts_sha256': summary['files_sha256']['contexts.jsonl']})
    print(f'{cohort.split} s{cohort.seed}: {len(generated)} answers', flush=True)


def require_completed(split, seed):
    cohort = Cohort(split, seed)
    phased = runtime.phased_runner()
    if runtime.read(cohort.dest / 'run_note.json') != identity(cohort, phased):
        raise RuntimeError(f'Completed {split} s{seed} C0 belongs to another index or configuration')
    require_keyword_cache(cohort, phased)
    ctx = runtime.read(cohort.dest / 'contexts_summary.json')
    ans = runtime.read(cohort.dest / 'answers_summary.json')
    require_hashes(cohort.dest, ctx['files_sha256'] | ans['files_sha256'])
    if (ctx['keyword_sha256'] != runtime.sha(cohort.keywords)
            or ans['contexts_sha256'] != ctx['files_sha256']['contexts.jsonl']):
        raise RuntimeError(f'Completed {split} s{seed} C0 phase hashes disagree')


def execute(phased):
    p.C0.mkdir(parents=True, exist_ok=True)
    status = p.C0 / 'status.json'
    completed = []
    with phased.qa_run.run_lock(p.C0):
        try:
            for split, seed in COHORTS:
                cohort = Cohort(split, seed)
                for step in ('keywords', 'contexts', 'answers'):
                    write_json(status, {'state': 'running', 'at': runtime.now(), 'phase': step,
                                        'split': split, 'seed': seed, 'completed_cohorts': completed})
                    subprocess.run(child_command(step, cohort), cwd=p.ROOT, check=True)
                completed.append({'split': split, 'seed': seed})
            write_json(status, {'state': 'completed', 'at': runtime.now(), 'completed_cohorts': completed})
        except BaseException as exc:
            write_json(status, {'state': 'failed', 'at': runtime.now(), 'phase': step,
                                'split': split, 'seed': seed, 'completed_cohorts': completed,
                                'error_type': type(exc).__name__})
            raise


def pilot_fixture():
    """Re-key five copied Qwen keyword responses for the pilot's own TYPE_POOL."""
    from minirag.prompt import PROMPTS

    OFFLINE.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='c0_pilot_', dir=OFFLINE))
    wanted = {row['query_id']: row for row in runtime.questions('dev')[:5]}
    source = p.SHARED_INPUTS / 'kw_dev_s101.jsonl'
    originals = {}
    with source.open('rb') as handle:
        for line in handle:
            record = json.loads(line)
            if record['query_id'] in wanted:
                originals[record['query_id']] = (line, record)
    if originals.keys() != wanted.keys():
        raise RuntimeError('Missing Qwen keyword results for the pilot questions')
    with (root / 'qwen_keyword_records.jsonl').open('xb') as handle:
        for qid in wanted:
            handle.write(originals[qid][0])

    graph = ET.parse(p.PILOT / 'graph_chunk_entity_relation.graphml')
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    entity_type = next(key.get('id') for key in graph.findall('g:key', ns)
                       if key.get('attr.name') == 'entity_type' and key.get('for') == 'node')
    types = sorted({value.text.lower() for value in graph.findall('g:graph/g:node/g:data', ns)
                    if value.get('key') == entity_type and value.text})
    synthetic = []
    for qid, row in wanted.items():
        prompt = PROMPTS['minirag_query2kwd'].format(query=row['Question'], TYPE_POOL=types)
        prompt_sha = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        synthetic.append(originals[qid][1] | {'prompt_sha256': prompt_sha})
    write_records(root / 'synthetic_kw.jsonl', synthetic, fresh=True)
    write_json(root / 'fixture.json', {'question_ids': list(wanted), 'copied_from': str(source),
               'pilot_type_count': len(types), 'rekeyed': sum(
                   row['prompt_sha256'] != originals[row['query_id']][1]['prompt_sha256'] for row in synthetic),
               'synthetic_keyword_sha256': runtime.sha(root / 'synthetic_kw.jsonl')}, fresh=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('keywords', 'contexts', 'answers', 'execute', 'context-worker'))
    parser.add_argument('--split', choices=('dev', 'test'), default='test')
    parser.add_argument('--seed', type=int, choices=runtime.SEEDS, default=101)
    parser.add_argument('--pilot-test', action='store_true', help='offline five-question dev pilot')
    parser.add_argument('--pilot-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--shard', type=int, choices=range(WORKERS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.split == 'dev' and args.seed != 101:
        parser.error('dev C0 uses seed 101 only')
    if args.pilot_test and (args.step, args.split, args.seed) != ('contexts', 'dev', 101):
        parser.error('--pilot-test requires contexts --split dev --seed 101')
    if args.step == 'context-worker' and args.shard is None:
        parser.error('context-worker requires --shard')
    if args.pilot_dir is not None:
        args.pilot_dir = args.pilot_dir.resolve()
        if (args.step != 'context-worker' or args.split != 'dev' or args.pilot_dir.parent != OFFLINE
                or not args.pilot_dir.name.startswith('c0_pilot_')):
            parser.error('--pilot-dir is reserved for offline dev pilot workers')
    offline = args.step in ('contexts', 'context-worker')
    if not offline and not p._installed:
        parser.error('invoke online C0 phases through reproduce.mhr_phi.launch')
    p.apply(offline=offline)
    install_subprocess_rewrite()
    phased = runtime.phased_runner()
    if args.step == 'execute':
        return execute(phased)
    cohort = Cohort(args.split, args.seed, pilot_fixture() if args.pilot_test else args.pilot_dir)
    if args.step == 'context-worker':
        return context_worker(cohort, phased, args.shard)
    operations = {'keywords': keywords, 'contexts': contexts, 'answers': answers}
    if cohort.pilot_dir:
        prepare(cohort, phased)
        contexts(cohort, phased)
        if (cohort.dest / 'requests.jsonl').exists():
            raise RuntimeError('Offline phase B unexpectedly wrote a request trace')
        print(f'Offline pilot artifacts: {cohort.pilot_dir}', flush=True)
        return 0
    cohort.dest.mkdir(parents=True, exist_ok=True)
    with phased.qa_run.run_lock(cohort.dest):
        prepare(cohort, phased)
        operations[args.step](cohort, phased)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
