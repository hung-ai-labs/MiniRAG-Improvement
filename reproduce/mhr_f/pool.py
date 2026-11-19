"""Capture the full candidate ranking behind C0 + source-time on MultiHop-RAG (0 LLM calls).

Runs the frozen C0 code (review/mhr_st_20261112/source) with the ST adapter of
reproduce/mhr_st exactly as the +ST run did, and records through thin wrappers
what lies behind the native top_k/2 = 30 cut:

  graph_full    every chunk kwd2chunk scored, in native Counter order
  graph_scores  its kwd2chunk score
  dense_ids     the native dense top 30 (C0 uses them only as a x10 path-head boost)
  groups        query keyword -> seed entities -> (score, chunk path)
  final_full    the ST ranking without the cut (= graph_full when no date anchor)
  boosted       chunks ST promoted per anchor (<= 3 each)
  kept          chunk ids that reached Sources (A1@4000), from the context log

Wrappers call the original functions once more without the limit and assert the
native result is the prefix of the unlimited one, so retrieval is untouched.
Validation: for questions without an anchor ST == C0, so `kept` must equal the
frozen C0 Sources of the same seed (dev: MiniRAG dev400 C0_s101; test: archived C0).
Parser output comes only from the frozen keyword caches (MINIRAG_KW_CACHE_ONLY=1).

    python -m reproduce.mhr_f.pool run --split dev --workers 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from reproduce.mhr_st.common import DATA, INDEX, OUT as ST_OUT, ROOT, read, records, append, write, sha, setup

OUT = ROOT / 'review/mhr_f_20261114'
DEV_C0_SRC = ROOT.parent / 'MiniRAG/runs/multihop_rag/dev400/C0_s101/context.jsonl'


def questions(split, limit=None):
    import csv
    with (ST_OUT / f'inputs/{split}.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def kw_cache(split, seed):
    return ST_OUT / 'inputs/kw_dev_s101.jsonl' if split == 'dev' else ST_OUT / f's{seed}/kw_s{seed}.jsonl'


def c0_reference(split, seed):
    """query text -> frozen C0 Sources chunk ids of the same seed."""
    if split == 'dev':
        if seed != 101:
            raise ValueError('the frozen dev C0 has seed 101 only')
        local = OUT / 'inputs/dev_c0_s101_context.jsonl'
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DEV_C0_SRC, local)
        path = local
    else:
        path = ST_OUT / f'archive_c0/qa/s{seed}/C0_s{seed}/context.jsonl'
    return {r['query']: r.get('chunk_ids', []) for r in records(path)}


def hf_cache():
    """The embedding model cache lives in the MiniRAG checkout (never copied: E: is nearly full)."""
    local = ROOT / 'data/leef/cache/huggingface/hub'
    return local if local.exists() else ROOT.parent / 'MiniRAG/data/leef/cache/huggingface/hub'


def check_embedding():
    """Same all-MiniLM-L6-v2 files as the +ST run (embedding_cache.json)."""
    want = read(ST_OUT / 'embedding_cache.json')
    snap = hf_cache() / 'models--sentence-transformers--all-MiniLM-L6-v2/snapshots' / want['revision']
    got = {n: sha(snap / n) for n in want['files_sha256']}
    if got != want['files_sha256']:
        raise ValueError(f'embedding files differ from the +ST run: {snap}')
    return want['revision']


def worker(a):
    setup()
    os.environ['HF_HUB_CACHE'] = str(hf_cache())
    import phased_qa as phased
    import minirag.operate as op
    from minirag import QueryParam
    from minirag.prompt import PROMPTS
    from reproduce.mhr_st.adapter import SourceTime, load_temporal

    dest = OUT / a.split / f's{a.seed}'
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f'pools_part{a.shard}.jsonl'
    log = dest / f'context_log_part{a.shard}.jsonl'
    done = {r['query_id'] for r in records(out)}
    rows = questions(a.split, a.limit)[a.shard::a.workers]
    ref = c0_reference(a.split, a.seed)
    phased.apply_env('C0', a.seed, kw_cache=kw_cache(a.split, a.seed), cache_only=True, context_log=log)
    for key in ('MINIRAG_PATH_PRUNE', 'MINIRAG_PATH_SCORE', 'MINIRAG_RERANK', 'MINIRAG_MAX_TOKEN_TEXT_UNIT'):
        os.environ.pop(key, None)
    temporal = load_temporal(ST_OUT / 'source/temporal_slide.py')
    docs = records(DATA / 'processed/docs.jsonl')
    chunks = read(INDEX / 'kv_store_text_chunks.json')
    cap = {}

    class Capture(SourceTime):
        def rank(self, groups, vector_ids, limit, graph, pairs):
            ranked = super().rank(groups, vector_ids, limit, graph, pairs)
            scores = self.scores(groups, vector_ids, limit)
            h = self.temporal.entity_h(graph, pairs)
            full, lists = self.temporal.rank_temporal(scores, self.by_time, self.trace['anchors'], h, vector_ids,
                                                      expand=True, h_one=False, limit=None)
            assert full[:limit] == ranked and lists == self.trace['promoted'], 'rank_temporal prefix changed'
            cap.update(final_full=full, boosted=lists)
            return ranked

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
        return native

    work = OUT / 'snapshots' / f'{a.split}_s{a.seed}_w{a.shard}'
    if not work.exists():
        phased.snapshot(work, INDEX)
    rag = phased.build(work, llm_max_async=1)
    rag.llm_model_func = phased._forbidden_llm

    async def run():
        for row in rows:
            qid, q = row['query_id'], row['Question']
            if qid in done:
                continue
            cap.clear()
            n_log = len(records(log))
            op.kwd2chunk = kwd2chunk
            adapter.install()
            started = time.perf_counter()
            try:
                ctx = await rag.aquery(q, param=QueryParam(mode='mini', only_need_context=True))
            finally:
                op._build_mini_query_context = adapter.original
                op.kwd2chunk = native_kwd2chunk
            anchors = adapter.trace['anchors']
            status = 'no_context' if ctx is None else 'early_exit' if ctx == PROMPTS['fail_response'] else 'ok'
            new = records(log)[n_log:]
            kept = new[-1]['chunk_ids'] if status == 'ok' and new else []
            graph_full = cap.get('graph_full', [])
            rec = {'query_id': qid, 'status': status, 'anchors': anchors,
                   'retrieval_s': round(time.perf_counter() - started, 3),
                   'context_sha256': sha_text(ctx) if status == 'ok' else None,
                   'kept': kept, 'final_full': cap.get('final_full', graph_full), 'graph_full': graph_full,
                   'graph_scores': cap.get('graph_scores', {}), 'dense_ids': cap.get('dense_ids', []),
                   'groups': cap.get('groups', {}), 'boosted': cap.get('boosted', {})}
            if status == 'ok':
                assert rec['final_full'][:30] == adapter.trace.get('ranked_ids', graph_full[:30]), qid
            if not anchors:
                # ST falls back to C0 exactly: Sources must equal the frozen C0 Sources.
                rec['c0_sources_match'] = kept == ref.get(q, [])
            append(out, rec)
            print(json.dumps({'seed': a.seed, 'shard': a.shard, 'query_id': qid, 'status': status,
                              'anchors': len(anchors), 'pool': len(rec['final_full']),
                              'c0_match': rec.get('c0_sources_match')}), flush=True)

    asyncio.run(run())
    return 0


def index_manifest():
    """sha of the frozen index manifest, after checking every listed file against it."""
    manifest = read(INDEX / 'index_manifest.json')
    bad = [p for p, h in manifest['file_sha256'].items() if sha(INDEX / p) != h]
    if bad:
        raise ValueError(f'frozen index checksum mismatch: {bad}')
    return sha(INDEX / 'index_manifest.json')


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def run(a):
    dest = OUT / a.split / f's{a.seed}'
    dest.mkdir(parents=True, exist_ok=True)
    layout = {'workers': a.workers, 'split': a.split, 'seed': a.seed, 'limit': a.limit}
    if (dest / 'worker_layout.json').exists() and read(dest / 'worker_layout.json') != layout:
        raise ValueError('worker layout changed; resume with the original --workers/--limit')
    write(dest / 'worker_layout.json', layout)
    revision = check_embedding()
    c0_reference(a.split, a.seed)  # copy the dev reference before workers start
    procs = []
    for k in range(a.workers):
        cmd = [sys.executable, '-u', '-m', 'reproduce.mhr_f.pool', 'worker', '--split', a.split,
               '--seed', str(a.seed), '--workers', str(a.workers), '--shard', str(k)]
        if a.limit:
            cmd += ['--limit', str(a.limit)]
        fh = (dest / f'worker{k}.log').open('a', encoding='utf-8')
        procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                      env=os.environ | {'PYTHONHASHSEED': '13', 'PYTHONUTF8': '1'}))
        fh.close()
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f'pool workers failed: {codes}; see {dest}/worker*.log')
    got = {r['query_id']: r for k in range(a.workers) for r in records(dest / f'pools_part{k}.jsonl')}
    wanted = questions(a.split, a.limit)
    missing = [r['query_id'] for r in wanted if r['query_id'] not in got]
    if missing:
        raise ValueError(f'{len(missing)} pools missing')
    with (dest / 'pools.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for r in wanted:
            f.write(json.dumps(got[r['query_id']], ensure_ascii=False) + '\n')
    rows = [got[r['query_id']] for r in wanted]
    checked = [r for r in rows if 'c0_sources_match' in r]
    summary = {'split': a.split, 'seed': a.seed, 'questions': len(rows),
               'status': {s: sum(r['status'] == s for r in rows) for s in ('ok', 'no_context', 'early_exit')},
               'with_anchor': sum(bool(r['anchors']) for r in rows),
               'c0_sources_checked': len(checked), 'c0_sources_match': sum(r['c0_sources_match'] for r in checked),
               'pools_sha256': sha(dest / 'pools.jsonl'), 'embedding_revision': revision,
               'index_manifest_sha256': index_manifest(), 'llm_calls': 0}
    if summary['index_manifest_sha256'] != read(ST_OUT / 'protocol.json')['index_manifest_sha256']:
        raise ValueError('index files changed')
    write(dest / 'pools_summary.json', summary)
    for k in range(a.workers):
        shutil.rmtree(OUT / 'snapshots' / f'{a.split}_s{a.seed}_w{k}', ignore_errors=True)
    print(json.dumps(summary), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('run', 'worker'))
    ap.add_argument('--split', choices=('dev', 'test'), default='dev')
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    return worker(a) if a.step == 'worker' else run(a)


if __name__ == '__main__':
    raise SystemExit(main())
