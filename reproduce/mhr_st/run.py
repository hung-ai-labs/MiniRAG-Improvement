"""Run only C0 + ST on the frozen MultiHop-RAG/Qwen benchmark.

  python -m reproduce.mhr_st.run prepare
  python -m reproduce.mhr_st.run preflight
  python -m reproduce.mhr_st.run contexts --split dev --limit 20 --seed 101
  python -m reproduce.mhr_st.run execute
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from .common import ARCHIVE, DATA, INDEX, JUDGE, OUT, ROOT, SEEDS, SOURCE_COMMIT, append, read, records, setup, sha, write


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / 'source'
    manifest = read(ARCHIVE / 'RUN_MANIFEST.json')
    if not source.exists():
        archive = subprocess.check_output(['git', 'archive', SOURCE_COMMIT, '--', 'minirag',
            'reproduce/gemini_common.py', 'reproduce/Step_2_evaluate.py', 'reproduce/research'])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                target = source / member.name
                if not target.resolve().is_relative_to(source.resolve()):
                    raise ValueError('Unsafe source archive member')
                if member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    content = tar.extractfile(member).read()
                    expected = manifest['source_hashes'].get(member.name)
                    if expected and hashlib.sha256(content).hexdigest() != expected:
                        crlf = content.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
                        if hashlib.sha256(crlf).hexdigest() == expected:
                            content = crlf
                    target.write_bytes(content)
        # Recover the exact historical backend when its transport-only patch was
        # not committed at the same point as the scientific source snapshot.
        key = 'minirag/llm/gemini.py'
        expected = manifest['source_hashes'][key]
        if sha(source / key) != expected:
            commits = subprocess.check_output(['git', 'log', '--all', '--format=%H', '--', key], text=True).splitlines()
            for commit in commits:
                content = subprocess.check_output(['git', 'show', f'{commit}:{key}'])
                for value in (content, content.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')):
                    if hashlib.sha256(value).hexdigest() == expected:
                        (source / key).write_bytes(value)
                        break
                if sha(source / key) == expected:
                    break
        # The original slide implementation is frozen independently, and installed
        # as an in-memory adapter; unrelated temporal-branch code is never imported.
        temporal = subprocess.check_output(['git', 'show', '1297b2e:minirag/temporal.py'])
        (source / 'temporal_slide.py').write_bytes(temporal)
    if not (OUT / 'protocol.json').exists():
        for p, expected in manifest['source_hashes'].items():
            content = (source / p).read_bytes()
            lf = content.replace(b'\r\n', b'\n')
            for candidate in (lf, lf.replace(b'\n', b'\r\n')):
                if hashlib.sha256(candidate).hexdigest() == expected:
                    (source / p).write_bytes(candidate)
                    break
    source_checks = {p: sha(source / p) == h for p, h in manifest['source_hashes'].items()}
    if not all(source_checks.values()):
        raise RuntimeError(f'Historical source mismatch: {[p for p, ok in source_checks.items() if not ok]}')
    index_manifest = read(INDEX / 'index_manifest.json')
    index_checks = {p: sha(INDEX / p) == h for p, h in index_manifest['file_sha256'].items()}
    if not all(index_checks.values()):
        raise RuntimeError('Frozen index checksum mismatch')
    (OUT / 'inputs').mkdir(exist_ok=True)
    for name, src in [('dev.csv', DATA / 'splits/dev400.csv'), ('test.csv', ARCHIVE / 'test_internal.csv')]:
        dest = OUT / 'inputs' / name
        if dest.exists() and sha(dest) != sha(src):
            raise ValueError('Frozen question copy changed')
        if not dest.exists():
            shutil.copyfile(src, dest)
    for seed in SEEDS:
        dest = OUT / f's{seed}'
        dest.mkdir(exist_ok=True)
        old = ARCHIVE / f'qa/s{seed}'
        cache = dest / f'kw_s{seed}.jsonl'
        if cache.exists() and sha(cache) != sha(old / cache.name):
            raise ValueError('Keyword cache changed')
        if not cache.exists():
            shutil.copyfile(old / cache.name, cache)
    dev_kw = ROOT / 'runs/multihop_rag/dev400/kw_s101.jsonl'
    dest_kw = OUT / 'inputs/kw_dev_s101.jsonl'
    if dest_kw.exists() and sha(dest_kw) != sha(dev_kw):
        raise ValueError('Dev keyword cache changed')
    if not dest_kw.exists():
        shutil.copyfile(dev_kw, dest_kw)
    protocol = {
        'experiment': 'MultiHop-RAG C0 + source-time only', 'date': '2026-11-12',
        'authorization': 'User requested running +ST and pinning gemini-3.5-flash-lite.',
        'source_commit': SOURCE_COMMIT, 'source_checks': source_checks,
        'slide_temporal_commit': '1297b2e', 'slide_temporal_sha256': sha(source / 'temporal_slide.py'),
        'index': str(INDEX), 'index_manifest_sha256': sha(INDEX / 'index_manifest.json'),
        'index_checks': index_checks, 'model': 'Qwen/Qwen2.5-3B-Instruct',
        'seeds': list(SEEDS), 'judge_model': JUDGE, 'judge_passes': 3,
        'budget_tokens': 4000, 'top_k': 60, 'max_answer_tokens': 1024,
        'st': {'metadata': 'published_at -> every chunk of its document',
               'anchors': 'unchanged slide day/month parser', 'per_anchor_limit': 3,
               'expand': True, 'entity_h': True, 'fallback': 'native C0 when no anchor'},
        'controls': 'C0 index, keyword caches, embedding, graph scoring, Entities, prompt and A1 are frozen.',
        'c0_answers': 'reuse archived C0; rejudge in separate folders with the pinned judge',
        'dev': 400, 'test': 2156,
        'statistics': 'paired exact McNemar per generation seed; report acc/err/neither/no_majority; separate generation/judge SD',
        'test_status': 'Existing internal test already evaluated by the earlier RRF experiment; no tuning on it.',
    }
    dest = OUT / 'protocol.json'
    if dest.exists() and read(dest) != protocol:
        raise RuntimeError('Protocol changed; use a new run folder')
    write(dest, protocol)
    print(json.dumps({'prepared': True, 'source_files_verified': len(source_checks),
                      'index_files_verified': len(index_checks), 'run': str(OUT)}), flush=True)


def preflight():
    from dotenv import dotenv_values
    import httpx
    cfg = dotenv_values(ROOT / '.env')
    result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'judge_model': JUDGE}
    endpoint = cfg['MINIRAG_SLM_URL'].rstrip('/').removesuffix('/v1')
    for field, path in [('version', '/version'), ('models', '/v1/models')]:
        try:
            r = httpx.get(endpoint + path, headers={'Authorization': 'Bearer ' + cfg['MINIRAG_SLM_KEY']}, timeout=30)
            result['slm_' + field] = {'status': r.status_code, 'data': r.json() if r.status_code == 200 else None}
        except Exception as exc:
            result['slm_' + field] = {'error': type(exc).__name__}
    keys = [v for k, v in cfg.items() if k.startswith('GEMINI_API_KEY_') and v]
    result['gemini'] = []
    for slot, key in enumerate(keys):
        try:
            r = httpx.get('https://generativelanguage.googleapis.com/v1beta/models/' + JUDGE,
                          headers={'x-goog-api-key': key}, timeout=15)
            record = {'key_slot': slot + 1, 'status': r.status_code}
            if r.status_code == 200:
                record['model'] = {k: r.json().get(k) for k in ('name', 'version', 'displayName')}
            else:
                record['error_status'] = r.json().get('error', {}).get('status')
            result['gemini'].append(record)
            if r.status_code == 200:
                break
        except Exception as exc:
            result['gemini'].append({'key_slot': slot + 1, 'error': type(exc).__name__})
            if slot >= 2:
                break
    write(OUT / 'preflight.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if (result.get('slm_models', {}).get('status') == 200
                 and any(x.get('status') == 200 for x in result['gemini'])) else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('step', choices=('prepare', 'preflight', 'contexts', 'worker', 'answers', 'execute', 'status'))
    ap.add_argument('--split', choices=('dev', 'test'), default='test')
    ap.add_argument('--seed', type=int, choices=SEEDS, default=101)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--shard', type=int, default=0)
    args = ap.parse_args()
    if args.step == 'prepare':
        prepare()
    elif args.step == 'preflight':
        return preflight()
    else:
        from .pipeline import dispatch
        return dispatch(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
