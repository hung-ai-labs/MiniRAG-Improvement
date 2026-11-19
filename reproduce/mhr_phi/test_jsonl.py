"""Offline test of the JSONL reader fix (reproduce/mhr_phi/jsonl.py).

    python -B -m reproduce.mhr_phi.test_jsonl
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from reproduce.mhr_phi import profile as p

p.apply(offline=True)
sys.path.insert(0, str(p.SHARED_SOURCE / 'reproduce/research'))
import judge_run  # noqa: E402
import phased_qa  # noqa: E402
from reproduce.mhr_st import common  # noqa: E402
from reproduce.mhr_phi import jsonl  # noqa: E402


def original(name, file):
    spec = importlib.util.spec_from_file_location(f'_orig_{name}', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    assert phased_qa.read_jsonl is jsonl.read_jsonl
    assert judge_run.read_records is jsonl.read_records
    assert common.records is jsonl.records
    o_pq = original('phased_qa', p.SHARED_SOURCE / 'reproduce/research/phased_qa.py')
    o_jr = original('judge_run', p.SHARED_SOURCE / 'reproduce/research/judge_run.py')
    o_cm = original('common', p.ROOT / 'reproduce/mhr_st/common.py')

    seps = '\x85  \x0b\x0c\x1c\x1d\x1e'
    files = [f for f in list(p.OUT.glob('c0/**/*.jsonl')) + list(p.OUT.glob('st/**/*.jsonl'))
             + list((p.ROOT / 'review/mhr_st_20261112/archive_c0').glob('**/*.jsonl'))
             + list((p.ROOT / 'review/mhr_st_20261112/test').glob('*/judge*/**/judge.jsonl'))
             if 'snapshot' not in f.parts and '_offline_test' not in f.parts]
    same = with_sep = 0
    for f in files:
        text = f.read_text(encoding='utf-8')
        n_sep = sum(text.count(c) for c in seps)
        if n_sep:
            with_sep += 1
            print(f'separators {n_sep:3d}  {f.relative_to(p.ROOT)}  records({len(jsonl.records(f))}) == "\\n" lines '
                  f'({sum(1 for line in text.split(chr(10)) if line.strip())})')
            continue
        assert jsonl.records(f) == o_cm.records(f), f
        if 'judge' in f.name:
            assert jsonl.read_records(f) == o_jr.read_records(f), f
        else:
            try:
                ref = o_pq.read_jsonl(f)
            except KeyError:
                continue  # file without query_id keys (requests logs); records() above already compared
            assert jsonl.read_jsonl(f) == ref, f
        same += 1
    print(f'{same} files read identically by old and new readers; {with_sep} contain separators')

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'x.jsonl'
        rows = [{'query_id': f'q{i}', 'pass': 1, 'answer': f'a{c}b'} for i, c in enumerate(seps)]
        f.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        assert jsonl.records(f) == rows
        assert list(jsonl.read_jsonl(f)) == [r['query_id'] for r in rows]
        assert len(jsonl.read_records(f)) == len(rows)
        try:
            o_cm.records(f)
            raise AssertionError('original reader should fail on separators')
        except ValueError:
            pass
    print('synthetic separators: new readers OK, original fails as diagnosed')


if __name__ == '__main__':
    main()
