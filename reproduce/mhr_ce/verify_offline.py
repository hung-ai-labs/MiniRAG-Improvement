"""Offline regression checks only. Never execute answers/execute/finish/report."""
from __future__ import annotations

import ast
import csv
import io
import json
from pathlib import Path
from unittest.mock import patch

from reproduce.mhr_ce import offline
from reproduce.mhr_st.common import read, records, sha


def main():
    from reproduce.mhr_ce.local_runtime import install
    install()
    offline.guard()
    from reproduce.mhr_ce import run
    from reproduce.mhr_f import study
    from reproduce.mhr_f.pool import OUT as F_OUT
    import numpy as np

    # Compile every added source in memory: no .pyc or imports of online paths.
    sources = ['offline.py', 'local_runtime.py', 'study.py', 'run.py', 'judge_entry.py', 'verify_offline.py']
    for name in sources:
        p = Path(__file__).with_name(name)
        compile(ast.parse(p.read_text(encoding='utf-8'), filename=str(p)), str(p), 'exec')

    result = read(offline.OUT / 'dev_offline/offline_eval.json')
    reference = read(F_OUT / 'dev_offline/offline_eval.json')
    for arm in ('C0', 'ST', 'F', 'ORACLE'):
        assert result['arms'][arm] == reference['arms'][arm], arm
    assert result['auc_per_fold']['F'] == reference['auc_per_fold']
    assert result['F_vs_ST'] == reference['F_vs_ST']
    assert result['fold_sizes'] == reference['fold_sizes']
    rows = records(offline.OUT / 'dev_offline/selections.jsonl')
    saved = {r['query_id']: r['F'] for r in records(F_OUT / 'dev_offline/selections.jsonl')}
    assert len(rows) == 400 and all(r['F'] == saved[r['query_id']] for r in rows)
    scores = records(offline.OUT / 'dev_offline/scores_CEF.jsonl')
    assert len(scores) == len({(r['query_id'], r['chunk_id']) for r in scores}) == 83845
    assert all(r['fold'] not in r['trained_on_folds'] and len(r['trained_on_folds']) == 4 for r in scores)
    fold = study.folds(offline.csv_rows('dev'))
    assert all(r['fold'] == fold[r['query_id']] for r in scores)

    # Resumption can split a query across records. Preserve old chunk scores
    # while applying the latest value for chunks rescored in a later record.
    fake_summary = {'sha256': 's', 'pools_sha256': {}, 'chunks_text_sha256': 's', 'questions': 1, 'pairs': 2}
    fake_records = [{'query_id': 'q', 'logits': {'a': 1.0}},
                    {'query_id': 'q', 'logits': {'b': 2.0, 'a': 3.0}}]
    with patch.object(offline, 'read', return_value=fake_summary), patch.object(offline, 'sha', return_value='s'), \
            patch.object(offline, 'records', return_value=fake_records):
        assert offline.logits('dev') == {'q': {'a': 3.0, 'b': 2.0}}

    # Exercise CSV quoting, CRLF, Unicode and source numbering while proving
    # that the Entities and trailing context bytes survive a replacement.
    from minirag.utils import list_of_list_to_csv
    chunks = {'a': {'content': 'alpha, "quoted"\nsecond line'}, 'b': {'content': 'Tiếng Việt'}}
    prefix, suffix = 'Entities stay byte-identical\r\n-----Sources-----\n```csv\n', '\n```\nTAIL\r\n'
    ctx = prefix + list_of_list_to_csv([['id', 'content'], [0, chunks['a']['content']]]) + suffix
    assert run.rebuild(ctx, ['a'], chunks).encode() == ctx.encode()
    rebuilt = run.rebuild(ctx, ['b', 'a'], chunks)
    assert rebuilt.startswith(prefix) and rebuilt.endswith(suffix)
    csv_value = rebuilt[len(prefix):-len(suffix)]
    assert list(csv.reader(io.StringIO(csv_value))) == [['id', 'content'], ['0', chunks['b']['content']], ['1', chunks['a']['content']]]

    # CE ties must keep native order. Missing scores must fail rather than
    # silently fall back to native/F probabilities.
    fake_model = {'features': list(run.FEATURES), 'pool': 'native@all'}
    with patch.object(study, 'variants', return_value={'native@all': ['b', 'a']}), \
            patch.object(study, 'featurize', return_value=[('b', [0] * 18), ('a', [0] * 18)]), \
            patch.object(run, 'probability', return_value=np.array([.5, .5])):
        assert run.ce_order({'query_id': 'q'}, fake_model, {'q': {'a': 1, 'b': 1}}) == ['b', 'a']
        try:
            run.ce_order({'query_id': 'q'}, fake_model, {'q': {'a': 1}})
        except KeyError:
            pass
        else:
            raise AssertionError('Missing CE score did not fail')

    model = read(run.MODEL)
    assert model['features'] == list(run.FEATURES) and len(model['features']) == 19
    assert model['closed_form_max_abs_gap'] < 1e-9
    assert model['offline_eval_sha256'] == sha(offline.OUT / 'dev_offline/offline_eval.json')
    assert model['rows'] == len(scores)
    checks = {'source_files_compiled': len(sources), 'historical_arms_equal': ['C0', 'ST', 'F', 'ORACLE'],
              'F_reference_questions_checked': len(rows), 'cross_fitted_scores_checked': len(scores),
              'fold_exclusion_and_assignment': True, 'duplicate_logits_merge': True,
              'csv_bytes_and_quoting': True, 'native_tie_order': True, 'missing_logits_fail': True,
              'frozen_model_verified': True, 'api_calls': 0}
    offline.write(offline.OUT / 'offline_validation.json', checks)
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
