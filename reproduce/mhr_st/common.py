from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
# Moved 2026-11-13 from MiniRAG/runs/multihop_rag/st_20261112 (dev-worker index
# snapshots stayed behind). The archived C0 answers/context logs of the phase-3
# test run (x20-x20/work/phase3/test_run/qa/s*/C0_s*) were copied to archive_c0/.
OUT = ROOT / 'review/mhr_st_20261112'
ARCHIVE = OUT / 'archive_c0'
DATA = ROOT / 'dataset/MultiHop-RAG'
INDEX = ROOT / 'MultiHop-RAG-qwen-full609'
SOURCE_COMMIT = 'b640765fb6ac46a817d3d221e2d652240f9e0c6c'
JUDGE = 'gemini-3.5-flash-lite'
SEEDS = (101, 202, 303)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def records(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for n, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f'Corrupt JSONL at {p.name}:{n}; refusing to discard data') from None
    return out


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def append(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()


def setup(source=None):
    """Import the historical C0 package, never the current working branch."""
    from dotenv import load_dotenv
    # Gemini keys: MINIRAG_DOTENV may point at another checkout's .env (never copied here).
    load_dotenv(os.environ.get('MINIRAG_DOTENV') or ROOT / '.env')
    source = Path(source or OUT / 'source')
    for p in (source, source / 'reproduce', source / 'reproduce/research'):
        sys.path.insert(0, str(p))
    os.environ.update(PYTHONUTF8='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_CACHE=str(ROOT / 'data/leef/cache/huggingface/hub'),
                      TIKTOKEN_CACHE_DIR=str(ROOT / 'data-gym-cache'))
    return source
