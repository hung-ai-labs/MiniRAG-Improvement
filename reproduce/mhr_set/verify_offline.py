"""Additional offline QA: source parity, syntax, immutable provenance, CSV replay.

Does not invoke answers, execute, finish, report, or any online entry point.
"""
from __future__ import annotations

import ast
import csv
import io
import json
from pathlib import Path

from reproduce.mhr_ce import run as ce
from reproduce.mhr_f import study as base
from reproduce.mhr_f.pool import OUT as F_OUT
from reproduce.mhr_set import offline
from reproduce.mhr_set import run as r
from reproduce.mhr_st.common import ROOT, SEEDS, read, records, sha


def function(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def main():
    offline.install()
    directory = ROOT / 'reproduce/mhr_set'
    checked = []
    for path in sorted(directory.glob('*.py')):
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        checked.append(path.name)
    original = function(ROOT / 'reproduce/evidence_set/select.py', 'select')
    port = function(directory / 'select.py', 'select')
    inline = function(directory / 'test_select.py', 'reference')
    assert ast.dump(ast.Module(body=original.body[1:], type_ignores=[])) == ast.dump(ast.Module(body=port.body, type_ignores=[]))
    assert ast.dump(ast.Module(body=inline.body, type_ignores=[])) == ast.dump(ast.Module(body=port.body, type_ignores=[]))
    params = r.frozen()
    chunks, _, _ = base.corpus()
    seeds = {}
    marker = '-----Sources-----\n```csv\n'
    for seed in SEEDS:
        path = r.folder(seed) / 'contexts.jsonl'
        summary = read(r.folder(seed) / 'context_summary.json')
        contexts = records(path)
        old = {v['query_id']: v for v in records(ce.folder(seed) / 'contexts.jsonl')}
        pools = {v['query_id']: v for v in records(F_OUT / f'test/s{seed}/pools.jsonl')}
        assert summary['checks']['passed'] and summary['contexts_sha256'] == sha(path)
        assert summary['set_params_sha256'] == sha(r.PARAMS)
        assert len(contexts) == len({v['query_id'] for v in contexts}) == 2156
        assert {v['query_id'] for v in contexts} == set(old) == set(pools)
        same = checked_csv = 0
        for value in contexts:
            q = value['query_id']
            ids, context = value['chunk_ids'], value['context']
            assert len(ids) == len(set(ids))
            assert value['tokens'] == sum(chunks[c]['tokens'] for c in ids) <= 4000
            assert set(ids) <= set(base.variants(pools[q])['native@all'])
            assert value['same_as_ce'] == (context.encode('utf-8') == old[q]['context'].encode('utf-8'))
            same += value['same_as_ce']
            if value['status'] != 'ok':
                assert ids == [] and context == old[q]['context'] and value['status'] == old[q]['status']
                continue
            assert context.count(marker) == 1
            start = context.index(marker) + len(marker)
            end = context.index('\n```', start)
            parsed = list(csv.reader(io.StringIO(context[start:end])))
            assert parsed == [['id', 'content']] + [[str(i), chunks[c]['content']] for i, c in enumerate(ids)]
            assert ce.rebuild(old[q]['context'], ids, chunks).encode('utf-8') == context.encode('utf-8')
            checked_csv += 1
        assert same == summary['same_as_ce']
        seeds[str(seed)] = {'questions': len(contexts), 'sources_csv_checked': checked_csv,
                            'same_as_ce': same, 'sha256': sha(path), 'passed': True}
    manifest = read(base.INDEX / 'index_manifest.json')
    assert all(sha(base.INDEX / path) == expected for path, expected in manifest['file_sha256'].items())
    result = {'syntax_checked': checked, 'selector_body_AST_equal_reference': True,
              'inline_reference_body_AST_equal_reference': True, 'seeds': seeds,
              'index_manifest_verified': True, 'api_calls': 0, 'passed': True}
    offline.write(offline.OUT / 'validation.json', result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
