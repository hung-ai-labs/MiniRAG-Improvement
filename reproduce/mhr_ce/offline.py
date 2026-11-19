"""Local-only setup and add-only outputs; never load credential files.

The historical setup() and minirag import both call load_dotenv. Disable that
function in memory before importing either; do not modify the frozen source.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time

from reproduce.mhr_st.common import ROOT, OUT as ST_OUT, read, records, sha

OUT = ROOT / 'review/mhr_ce_20261116'
FEATURE_PREFIX = ('ce_logit',)


def setup(source=None):
    sys.dont_write_bytecode = True
    os.environ.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                      PYTHON_DOTENV_DISABLED='1', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1',
                      TIKTOKEN_CACHE_DIR=str(ROOT / 'data-gym-cache'))
    import dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False
    source = Path(source or ST_OUT / 'source')
    for p in (source, source / 'reproduce', source / 'reproduce/research'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from reproduce.mhr_f.pool import hf_cache
    os.environ['HF_HUB_CACHE'] = str(hf_cache())
    from reproduce.mhr_f import study
    study.setup = setup
    return source


def guard():
    """Audit all executed offline work, including transitive library imports."""
    roots = (OUT.resolve(), (ROOT / 'reproduce/mhr_ce').resolve())

    def audit(event, args):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo',
                     'socket.sendto', 'socket.sendmsg'):
            raise RuntimeError('Network is forbidden during offline CE work')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.name.lower() == '.env' or path.name.lower().startswith('.env.'):
                raise RuntimeError('Credential-file access is forbidden')
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                if not any(path.is_relative_to(root) for root in roots):
                    raise RuntimeError(f'Write outside CE directories: {path}')
                if path.exists():
                    raise FileExistsError(f'ADD-ONLY: refusing to modify {path}')
        if event in ('os.remove', 'os.rename', 'os.rmdir', 'shutil.rmtree'):
            raise RuntimeError(f'ADD-ONLY: forbidden filesystem mutation {event}')
    sys.addaudithook(audit)
    setup()


def new_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value)


def write(path, value):
    new_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def csv_rows(split):
    with (ST_OUT / f'inputs/{split}.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate input query IDs')
    return rows


def logits(split):
    """Merge resumed records by query and chunk, with the latest chunk value winning."""
    path = OUT / f'ce_logits_{split}.jsonl'
    summary = read(OUT / f'ce_logits_{split}_summary.json')
    if summary['sha256'] != sha(path):
        raise ValueError(f'{split} CE logits checksum mismatch')
    for relative, expected in summary['pools_sha256'].items():
        if sha(ROOT / Path(relative.replace('\\', '/'))) != expected:
            raise ValueError(f'{split} CE source pools changed')
    if summary.get('chunks_text_sha256') != sha(OUT / 'chunks_text.json'):
        raise ValueError('CE chunk-text checksum mismatch')
    merged = {}
    for r in records(path):
        target = merged.setdefault(r['query_id'], {})
        for chunk, value in r['logits'].items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f'Nonfinite CE logit: {r["query_id"]}/{chunk}')
            target[chunk] = value
    if len(merged) != summary['questions'] or sum(map(len, merged.values())) != summary['pairs']:
        raise ValueError(f'{split} merged CE score coverage differs from summary')
    return merged


def wait_test_logits():
    """Do not inspect a partially written test journal; poll at registered 2 min intervals."""
    deadline = time.monotonic() + 90 * 60
    while not (OUT / 'ce_logits_test_summary.json').exists():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Test CE logits summary absent after 90 minutes')
        print('Waiting for ce_logits_test_summary.json (2 minute poll)', flush=True)
        time.sleep(min(120, remaining))


def dev_data():
    from reproduce.mhr_f import study
    from reproduce.mhr_f.pool import OUT as F_OUT
    src = F_OUT / 'dev/s101'
    summary = read(src / 'pools_summary.json')
    if sha(src / 'pools.jsonl') != summary['pools_sha256']:
        raise ValueError('Dev pools changed')
    rows = csv_rows('dev')
    pool_records = records(src / 'pools.jsonl')
    pools = {p['query_id']: p for p in pool_records}
    if len(pools) != len(pool_records) or set(pools) != {r['query_id'] for r in rows}:
        raise ValueError('Dev pools and questions differ')
    pool_name = read(F_OUT / 'dev_offline/coverage.json')['chosen_pool']
    if pool_name != 'native@all':
        raise ValueError('Registered pool must be native@all')
    ce = logits('dev')
    if set(ce) != set(pools):
        raise ValueError('Dev CE query IDs differ')
    gold, fold = study.gold_sets(), study.folds(rows)
    X, y, meta = [], [], []
    for row in rows:
        qid = row['query_id']
        if pools[qid]['status'] != 'ok':
            continue
        for c, f in study.featurize(pools[qid], study.variants(pools[qid])[pool_name]):
            X.append([ce[qid][c]] + f)  # missing scores fail hard
            y.append(int(any(c in g for g in gold.get(qid, []))))
            meta.append((qid, c, fold[qid]))
    import numpy as np
    return rows, pools, pool_name, ce, fold, np.array(X, float), np.array(y), meta, summary
