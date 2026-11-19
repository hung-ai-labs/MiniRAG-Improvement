"""JSONL readers that split records on "\\n" only (DEVIATIONS.md, 2026-11-19 reader fix).

The frozen readers use str.splitlines(), which also breaks on U+0085, U+2028, U+2029, \\x0b, \\x0c and \\x1c-\\x1e.
Phi writes U+0085 inside some answers; the writers (json.dumps(ensure_ascii=False) + "\\n") are correct, so only the
readers are replaced, in memory, under the Phi profile. For a file without those characters every function returns
exactly what the original returns (tested in test_jsonl.py).
"""
from __future__ import annotations

import json
from pathlib import Path


def lines(text):
    return [line[:-1] if line.endswith('\r') else line for line in text.split('\n')]


def read_jsonl(path, key='query_id'):
    """phased_qa.read_jsonl."""
    out = {}
    if Path(path).exists():
        for line in lines(Path(path).read_text(encoding='utf-8')):
            if line.strip():
                record = json.loads(line)
                out[record[key]] = record  # last record wins on resume
    return out


def read_records(path):
    """judge_run.read_records."""
    latest = {}
    if path.exists():
        rows = [line for line in lines(path.read_text(encoding='utf-8')) if line.strip()]
        for i, line in enumerate(rows):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                if i == len(rows) - 1:
                    break  # torn by a crash or power-off mid-write: that verdict is simply judged again
                raise ValueError(f'{path}: corrupt record on line {i + 1}') from None
            latest[(r['query_id'], r['pass'])] = r  # last record wins on resume
    return latest


def records(path):
    """reproduce.mhr_st.common.records."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for n, line in enumerate(lines(p.read_text(encoding='utf-8')), 1):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f'Corrupt JSONL at {p.name}:{n}; refusing to discard data') from None
    return out


PATCHES = {'phased_qa': {'read_jsonl': read_jsonl},
           'judge_run': {'read_records': read_records},
           'reproduce.mhr_st.common': {'records': records}}


def install(module):
    for name, func in PATCHES.get(module.__name__, {}).items():
        setattr(module, name, func)
