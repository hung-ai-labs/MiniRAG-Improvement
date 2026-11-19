"""Shared frozen-source setup, index checks, and Phi serving validation."""

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from . import profile as p

SEEDS = (101, 202, 303)
JUDGE = 'gemini-3.5-flash-lite'
QUESTION_HASHES = {
    'dev': '4633ef777b747ef0fb8cca86457454eed64680432d899a3d5410afce2b63ef77',
    'test': 'e37a4599e8485b5b76f6b60157648b60564f51ee7eceaeb6798f7dbd523d069d',
}
QUESTIONS_N = {'dev': 400, 'test': 2156}
SOURCE_COMMIT = 'b640765fb6ac46a817d3d221e2d652240f9e0c6c'
DOCS_SHA = '5816d2d43cb0bbd825f61414c1648c38e57b495c211b7d046936da0c32fd8077'
RETRIEVAL_OVERRIDES = ('MINIRAG_PATH_PRUNE', 'MINIRAG_PATH_SCORE', 'MINIRAG_RERANK',
                       'MINIRAG_MAX_TOKEN_TEXT_UNIT')
_phased = None


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def questions(split):
    path = p.ST / 'inputs' / f'{split}.csv'
    if sha(path) != QUESTION_HASHES[split]:
        raise RuntimeError(f'Frozen {split} question copy changed')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != QUESTIONS_N[split] or len({r['query_id'] for r in rows}) != len(rows):
        raise RuntimeError(f'Frozen {split} question count or IDs changed')
    return rows


def source_checks():
    expected = read(p.OUT / 'frozen_source_sha256.json')
    checks = {name: sha(p.SHARED_SOURCE / name) == digest for name, digest in expected.items()}
    if not all(checks.values()):
        raise RuntimeError(f'Frozen source changed: {[name for name, ok in checks.items() if not ok]}')
    return checks


def hf_cache():
    local = p.ROOT / 'data/leef/cache/huggingface/hub'
    return local if local.is_dir() else p.ROOT.parent / 'MiniRAG/data/leef/cache/huggingface/hub'


def setup(source=None):
    """Use the existing immutable snapshot and the local embedding cache."""
    global _phased
    source = Path(source or p.SHARED_SOURCE).resolve()
    if source != p.SHARED_SOURCE.resolve():
        raise ValueError('Phi ladder requires the frozen MiniRAG source')
    for path in (source, source / 'reproduce', source / 'reproduce/research'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', TOP_K='60',
                      HF_HUB_CACHE=str(hf_cache()), TIKTOKEN_CACHE_DIR=str(p.ROOT / 'data-gym-cache'))
    if _phased is None:
        phased = importlib.import_module('phased_qa')
        if Path(phased.__file__).resolve() != source / 'reproduce/research/phased_qa.py':
            raise RuntimeError('Loaded phased_qa from outside the frozen source')
        phased.INDEX = p.INDEX
        phased.snapshot.__defaults__ = (p.INDEX,)
        phased.use_slm = configure_slm
        phased.traced = traced
        original_build = phased.build

        def build(workdir, llm_max_async=32):
            if os.environ.get('MINIRAG_KW_CACHE_ONLY') != '1':
                return original_build(workdir, llm_max_async=llm_max_async)
            # The Phase 3 offline worker used the same construction bypass.
            # The LLM is replaced immediately, before any retrieval can run.
            with patch('gemini_common._resolve_api_keys', return_value=[]):
                rag = original_build(workdir, llm_max_async=llm_max_async)
            rag.llm_model_func = phased._forbidden_llm
            return rag

        phased.build = build
        _phased = phased
    return source


def phased_runner():
    setup()
    return _phased


def configure_slm():
    phased_runner().qa_run.configure_environment(argparse.Namespace(backend='modal'))
    os.environ['MINIRAG_MAX_TOKENS'] = '1024'


def serving_spec():
    spec = read(p.SERVING)
    if spec['model'] != p.MODEL or spec['vllm_version'] != '0.11.0':
        raise RuntimeError('Serving spec differs from the registered Phi model or vLLM version')
    return spec


def check_serving():
    import httpx

    spec = serving_spec()
    configure_slm()
    endpoint = os.environ['MINIRAG_SLM_URL'].rstrip('/').removesuffix('/v1')
    headers = {'Authorization': 'Bearer ' + os.environ['MINIRAG_SLM_KEY']}
    observed = {}
    for field, route in (('version', '/version'), ('models', '/v1/models')):
        for attempt in range(4):
            try:
                response = httpx.get(endpoint + route, headers=headers, timeout=60)
                response.raise_for_status()
                observed[field] = response.json()
                break
            except httpx.TimeoutException:
                if attempt == 3:
                    raise RuntimeError(f'Phi serving {field} check timed out during cold start') from None
    models = {item['id']: item for item in observed['models'].get('data', [])}
    if not {'minirag-slm', spec['model']} <= models.keys():
        raise RuntimeError('The SLM endpoint does not serve both registered Phi model names')
    if observed['version'].get('version') != spec['vllm_version']:
        raise RuntimeError('The SLM endpoint changed vLLM version')
    revision = models[spec['model']].get('model_revision')
    if revision is not None and revision != spec['model_revision']:
        raise RuntimeError('The SLM endpoint changed Phi model revision')
    return {'at': now(), 'model': spec['model'], 'served_models': sorted(models),
            'vllm_version': observed['version']['version'], 'registered_model_revision': spec['model_revision'],
            'observed_model_revision': revision}


def require_phi_response(record):
    if record.get('ok') and record.get('response_model') != p.MODEL:
        raise RuntimeError(f'SLM response did not identify the registered Phi model: {record.get("response_model")!r}')


def traced(directory, action):
    """Keep a request journal for the frozen runner and reject non-Phi replies."""
    import llm_trace

    llm_trace.install()
    with (Path(directory) / 'requests.jsonl').open('a', encoding='utf-8', newline='\n') as handle:
        def sink(record):
            record.pop('response_text', None)
            _phased.append_jsonl(handle, record | {'at': now()})
            require_phi_response(record)

        with llm_trace.trace(sink):
            return asyncio.run(action())


def verify_index(index, *, pilot=False):
    from reproduce.index_health import health

    manifest = read(index / 'index_manifest.json')
    kind, count = ('pilot', 30) if pilot else ('full', 609)
    if (manifest.get('status') != 'completed' or manifest.get('subset') != kind
            or manifest.get('counts', {}).get('docs') != count
            or manifest.get('docs_jsonl_sha256') != DOCS_SHA):
        raise RuntimeError(f'Expected a completed {count}-article Phi {kind} index')
    settings = {'chunk_token_size': 1200, 'chunk_overlap_token_size': 100,
                'tiktoken_model_name': 'gpt-4o-mini', 'entity_extract_max_gleaning': 1,
                'model': 'minirag-slm', 'embedmodel': 'local', 'max_tokens': '4096'}
    if pilot:
        settings.update(pilot_seed=13, pilot_size=30)
    if any(manifest['settings'].get(key) != value for key, value in settings.items()):
        raise RuntimeError('Index settings differ from the registered settings')
    verified = manifest.get('verification')
    files = manifest.get('file_sha256')
    if not verified or not all(value is True for value in verified.values()) or not files:
        raise RuntimeError('Index verification is incomplete')
    checks = {name: sha(index / name) == digest for name, digest in files.items()}
    if not all(checks.values()):
        raise RuntimeError(f'Index files changed: {[name for name, ok in checks.items() if not ok]}')
    coverage = health(index)
    if not coverage['chunks'] or coverage['zero_entity_chunks'] / coverage['chunks'] > .05:
        raise RuntimeError('Index gate I failed: more than 5% of chunks have zero entities')
    return checks, coverage


def prepare_st():
    from reproduce.mhr_f.pool import check_embedding
    from reproduce.mhr_st.common import write

    source = source_checks()
    checks, coverage = verify_index(p.INDEX)
    check_embedding()
    from .c0 import require_completed
    for split, seed in (('dev', 101), *(('test', seed) for seed in SEEDS)):
        require_completed(split, seed)
    receipt = {'experiment': 'MultiHop-RAG Phi C0 + source-time only', 'date': '2026-11-18',
               'source_commit': SOURCE_COMMIT, 'source_checks': source, 'slide_temporal_commit': '1297b2e',
               'slide_temporal_sha256': sha(p.SHARED_SOURCE / 'temporal_slide.py'),
               'index': str(p.INDEX), 'index_manifest_sha256': sha(p.INDEX / 'index_manifest.json'),
               'index_checks': checks, 'index_health': coverage, 'model': p.MODEL,
               'serving_config_sha256': sha(p.SERVING), 'question_sha256': QUESTION_HASHES,
               'seeds': list(SEEDS), 'judge_model': JUDGE, 'judge_passes': 3,
               'budget_tokens': 4000, 'top_k': 60, 'max_answer_tokens': 1024,
               'st': {'metadata': 'published_at -> every chunk of its document',
                      'anchors': 'unchanged slide day/month parser', 'per_anchor_limit': 3,
                      'expand': True, 'entity_h': True, 'fallback': 'native C0 when no anchor'},
               'controls': 'Phi C0 index, keyword caches, embedding, graph scoring, Entities, prompt and A1 are frozen.',
               'c0_answers': 'Use same-seed Phi C0 answers and share verdicts for identical ST answers.',
               'dev': 400, 'test': 2156,
               'statistics': 'Paired exact McNemar per generation seed. Report ACC, ERR, Neither and no majority.'}
    # JSON loads the health buckets as lists. Compare the persisted representation.
    receipt = json.loads(json.dumps(receipt, ensure_ascii=False))
    dest = p.ST / 'protocol.json'
    if dest.exists():
        if read(dest) != receipt:
            raise RuntimeError('Phi ST protocol or its inputs changed')
    else:
        write(dest, receipt)
    print(json.dumps({'prepared': True, 'source_files_verified': len(source),
                      'index_files_verified': len(checks), 'run': str(p.ST)}), flush=True)


def preflight_st():
    import httpx
    from reproduce.mhr_st.common import write

    result = {'at': now(), 'judge_model': JUDGE, 'slm': check_serving(), 'gemini': []}
    keys = [value for name, value in sorted(os.environ.items())
            if name.startswith('GEMINI_API_KEY_') and name != 'GEMINI_API_KEY_ONLY' and value]
    for slot, key in enumerate(keys, 1):
        try:
            response = httpx.get('https://generativelanguage.googleapis.com/v1beta/models/' + JUDGE,
                                 headers={'x-goog-api-key': key}, timeout=15)
            record = {'key_slot': slot, 'status': response.status_code}
            if response.status_code == 200:
                value = response.json()
                record['model'] = {field: value.get(field) for field in ('name', 'version', 'displayName')}
            result['gemini'].append(record)
            if response.status_code == 200:
                break
        except httpx.HTTPError as exc:
            result['gemini'].append({'key_slot': slot, 'error': type(exc).__name__})
    write(p.ST / 'preflight.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if any(row.get('status') == 200 for row in result['gemini']) else 2
