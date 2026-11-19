"""Row 4 (+CE): offline freeze/contexts, deferred answers/judge, four-arm report.

Offline commands never import the endpoint clients or load .env. Online commands
are supplied for the experiment operator and require credentials in the process
environment; they are not part of the authorized Codex execution.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from reproduce.mhr_ce import offline
from reproduce.mhr_f import run as previous
from reproduce.mhr_f import study
from reproduce.mhr_f.pool import OUT as F_OUT
from reproduce.mhr_st.common import (
    DATA, INDEX, JUDGE, OUT as ST_OUT, ROOT, SEEDS, append, read, records, sha,
    write as checkpoint,
)

OUT = offline.OUT
MODEL = OUT / 'model_CE.json'
FEATURES = ('ce_logit',) + study.FEATURES
QUESTIONS = ST_OUT / 'inputs/test.csv'
now = previous.now
probability = previous.probability
test_rows = previous.test_rows
read_csv = previous.read_csv


def folder(seed):
    return OUT / f'test/CE_s{seed}'


def status(name='status.json', **kw):
    checkpoint(OUT / name, {'state': 'running', 'at': now(), 'judge': JUDGE, **kw})


def freeze(a):
    if MODEL.exists():
        raise FileExistsError('ADD-ONLY: model_CE.json already exists; refusing to refit')
    evaluation = read(OUT / 'dev_offline/offline_eval.json')
    if not evaluation['F_selections_equal_reference'] or evaluation['F_reference_questions_checked'] != 400:
        raise ValueError('Dev F reproduction check missing')
    rows, pools, pool_name, ce, fold, X, y, meta, summary = offline.dev_data()
    if evaluation['pools_sha256'] != summary['pools_sha256'] or evaluation['ce_logits_dev_sha256'] != sha(OUT / 'ce_logits_dev.jsonl'):
        raise ValueError('Dev inputs differ from step A')
    model = study.fit(X, y)
    scaler, logistic = model[0], model[-1]
    frozen = {'features': list(FEATURES), 'pool': pool_name, 'C': study.C, 'random_state': study.SEED,
              'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
              'coef': logistic.coef_[0].tolist(), 'intercept': float(logistic.intercept_[0]),
              'trained_on': 'dev400 seed 101, all questions (no test data)',
              'rows': len(y), 'positives': int(y.sum()), 'dev_pools_sha256': summary['pools_sha256'],
              'ce_logits_dev_sha256': sha(OUT / 'ce_logits_dev.jsonl'),
              'offline_eval_sha256': sha(OUT / 'dev_offline/offline_eval.json'),
              'dev_offline_eval_sha256': sha(OUT / 'dev_offline/offline_eval.json'), 'frozen_at': now()}
    gap = float(np.max(np.abs(probability(frozen, X) - model.predict_proba(X)[:, 1])))
    if not gap < 1e-9:
        raise ValueError(f'Closed-form probability gap is {gap}')
    frozen['closed_form_max_abs_gap'] = gap
    offline.write(MODEL, frozen)
    print(json.dumps({'model_CE': str(MODEL), 'rows': len(y), 'positives': int(y.sum()), 'gap': gap}), flush=True)
    return 0


def ce_order(pool_record, model, ce):
    if model['features'] != list(FEATURES) or model['pool'] != 'native@all':
        raise ValueError('Frozen CE model specification differs from protocol')
    pool = study.variants(pool_record)[model['pool']]
    if not pool:
        return []
    qid = pool_record['query_id']
    features = [[ce[qid][c]] + f for c, f in study.featurize(pool_record, pool)]
    pr = probability(model, features)
    return [pool[i] for i in sorted(range(len(pool)), key=lambda i: (-pr[i], i))]


def rebuild(context, ids, chunks):
    """Change exactly the Sources CSV bytes, preserving both surrounding slices."""
    from minirag.utils import list_of_list_to_csv
    marker = '-----Sources-----\n```csv\n'
    if context.count(marker) != 1:
        raise ValueError('Expected exactly one Sources block')
    start = context.index(marker) + len(marker)
    end = context.index('\n```', start)
    source_csv = list_of_list_to_csv([['id', 'content']] + [[i, chunks[c]['content']] for i, c in enumerate(ids)])
    return context[:start] + source_csv + context[end:]


def contexts(a):
    dest = folder(a.seed)
    if not MODEL.exists():
        raise RuntimeError('Freeze model_CE.json before building test contexts')
    if any((dest / name).exists() for name in ('contexts.jsonl', 'context_summary.json', 'context_check_failed.json')):
        raise FileExistsError('ADD-ONLY: CE context outputs already exist')
    offline.wait_test_logits()
    qid = None
    try:
        model, model_f = read(MODEL), read(previous.MODEL)
        if model['ce_logits_dev_sha256'] != sha(OUT / 'ce_logits_dev.jsonl') or model['offline_eval_sha256'] != sha(OUT / 'dev_offline/offline_eval.json'):
            raise ValueError('Frozen dev provenance changed')
        cpath = previous.folder(a.seed) / 'contexts.jsonl'
        ppath = F_OUT / f'test/s{a.seed}/pools.jsonl'
        reference_summary = read(previous.folder(a.seed) / 'context_summary.json')
        if reference_summary['contexts_sha256'] != sha(cpath) or reference_summary['pools_sha256'] != sha(ppath):
            raise ValueError('Stored +F contexts/pools checksum mismatch')
        if reference_summary['model_F_sha256'] != sha(previous.MODEL):
            raise ValueError('Frozen F model differs from stored contexts')
        frows, prows = records(cpath), records(ppath)
        refs = {r['query_id']: r for r in frows}
        pools = {r['query_id']: r for r in prows}
        rows = test_rows()
        wanted = {r['query_id'] for r in rows}
        if set(refs) != wanted or set(pools) != wanted or len(frows) != len(rows) or len(prows) != len(rows):
            raise ValueError('Test questions, F contexts and pools differ')
        ce = offline.logits('test')
        if set(ce) != wanted:
            raise ValueError('Test CE query IDs differ')
        chunks = read(INDEX / 'kv_store_text_chunks.json')
        text = read(OUT / 'chunks_text.json')
        if set(text) != set(chunks) or any(text[c] != r['content'] for c, r in chunks.items()):
            raise ValueError('CE chunk contents differ from index')
        results, checked = [], 0
        for row in rows:
            qid = row['query_id']
            ref, pool = refs[qid], pools[qid]
            if ref['status'] == 'ok':
                if study.a1(previous.f_order(pool, model_f)) != ref['chunk_ids']:
                    raise AssertionError(f'F source selection mismatch for {qid}')
                if rebuild(ref['context'], ref['chunk_ids'], chunks).encode('utf-8') != ref['context'].encode('utf-8'):
                    raise AssertionError(f'F context byte replay mismatch for {qid}')
                checked += 1
                ids = study.a1(ce_order(pool, model, ce))
                ctx = rebuild(ref['context'], ids, chunks)
            else:
                ids, ctx = [], ref['context']
            results.append({'query_id': qid, 'status': ref['status'], 'context': ctx, 'chunk_ids': ids,
                            'same_as_f': ctx == ref['context']})
        summary = {'questions': len(rows), 'seed': a.seed,
                   'status': {s: sum(r['status'] == s for r in results) for s in sorted({r['status'] for r in results})},
                   'same_as_f': sum(r['same_as_f'] for r in results),
                   'same_as_f_ok': sum(r['same_as_f'] and r['status'] == 'ok' for r in results),
                   'checks': {'ok_questions_checked': checked, 'f_sources_recompute_mismatch': 0,
                              'f_context_byte_rebuild_mismatch': 0, 'passed': True},
                   'f_contexts_sha256': sha(cpath), 'pools_sha256': sha(ppath),
                   'model_F_sha256': sha(previous.MODEL), 'model_CE_sha256': sha(MODEL),
                   'ce_logits_test_sha256': sha(OUT / 'ce_logits_test.jsonl'),
                   'ce_logits_test_summary_sha256': sha(OUT / 'ce_logits_test_summary.json'),
                   'chunks_sha256': sha(INDEX / 'kv_store_text_chunks.json'),
                   'index_manifest_sha256': sha(INDEX / 'index_manifest.json'), 'llm_calls': 0}
        offline.jsonl(dest / 'contexts.jsonl', results)
        summary['contexts_sha256'] = sha(dest / 'contexts.jsonl')
        offline.write(dest / 'context_summary.json', summary)
        print(json.dumps(summary), flush=True)
        return 0
    except BaseException as exc:
        offline.write(dest / 'context_check_failed.json', {
            'seed': a.seed, 'query_id': qid, 'at': now(), 'error_type': type(exc).__name__, 'message': str(exc)})
        raise


def configure_slm(phased, seed):
    # Reuse the exact +F serving check; redirect only its in-memory output root.
    original = previous.OUT
    previous.OUT = OUT
    try:
        previous.configure_slm(phased, seed)
    finally:
        previous.OUT = original


def answers(a):
    offline.setup()
    import phased_qa as phased
    from minirag.base import QueryParam
    dest = folder(a.seed)
    summary = read(dest / 'context_summary.json')
    if not summary['checks']['passed'] or summary['contexts_sha256'] != sha(dest / 'contexts.jsonl'):
        raise ValueError('CE contexts not verified; refusing GPU calls')
    if (dest / 'answer_summary.json').exists():
        if read(dest / 'answer_summary.json')['answers_sha256'] != sha(dest / 'answers.jsonl'):
            raise ValueError('Completed CE answers changed')
        return 0
    rows = test_rows()
    ctx = {r['query_id']: r for r in records(dest / 'contexts.jsonl')}
    if set(ctx) != {r['query_id'] for r in rows}:
        raise ValueError('Incomplete CE contexts')
    old = {r['query_id']: r for r in records(previous.folder(a.seed) / 'answers.jsonl')}
    done = {r['query_id']: r for r in records(dest / 'answers.jsonl')}
    for row in rows:
        qid = row['query_id']
        if qid not in done and ctx[qid]['same_as_f']:
            value = old[qid] | {'source': 'frozen_F_identical_context'}
            append(dest / 'answers.jsonl', value)
            done[qid] = value
    if len(done) < len(rows):
        phased.apply_env('C0', a.seed)
        configure_slm(phased, a.seed)

        async def run():
            rag = phased.build(dest / 'answer_work', llm_max_async=32)
            await phased.answer_phase(rag.llm_model_func, rows, ctx, 'mini', dest / 'answers.jsonl',
                                      QueryParam().response_type)
        phased.traced(dest, run)
    got = {r['query_id']: r for r in records(dest / 'answers.jsonl')}
    if set(got) != {r['query_id'] for r in rows}:
        raise ValueError('Incomplete CE answers')
    # answer_phase already writes this CSV on an API run; reuse its contents if
    # present. On a fully reused run create it without overwriting anything.
    stream = io.StringIO(newline='')
    w = csv.writer(stream, lineterminator='\n')
    w.writerow(['Question', 'Gold Answer', 'minirag'])
    for row in rows:
        w.writerow([row['Question'], row['Gold Answer'], got[row['query_id']]['answer']])
    csv_path = dest / 'answers.csv'
    if csv_path.exists():
        with csv_path.open(encoding='utf-8', newline='') as f:
            if f.read() != stream.getvalue():
                raise ValueError('Existing CE answer CSV differs from journal')
    else:
        offline.new_text(csv_path, stream.getvalue())
    offline.write(dest / 'answer_summary.json', {
        'questions': len(rows), 'answers_sha256': sha(dest / 'answers.jsonl'),
        'reused_identical_F_context': sum(r.get('source') == 'frozen_F_identical_context' for r in got.values()),
        'context_status': {s: sum(r.get('context_status') == s for r in got.values())
                           for s in ('ok', 'no_context', 'early_exit', 'context_overflow')}})
    return 0


def execute(a):
    if not MODEL.exists():
        raise RuntimeError('Run freeze first')

    def answer(seed):
        status(step='answers', seed=seed)
        return subprocess.run([sys.executable, '-B', '-u', '-m', 'reproduce.mhr_ce.run', 'answers', '--seed', str(seed)],
                              cwd=ROOT, env=os.environ | {'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'}).returncode == 0
    try:
        failed = []
        for seed in SEEDS:
            status(step='contexts', seed=seed)
            dest = folder(seed)
            if (dest / 'context_summary.json').exists():
                s = read(dest / 'context_summary.json')
                if not s['checks']['passed'] or s['contexts_sha256'] != sha(dest / 'contexts.jsonl') or s['model_CE_sha256'] != sha(MODEL):
                    raise ValueError('Completed CE contexts changed')
            else:
                # Separate process keeps offline audit hooks away from API work.
                subprocess.run([sys.executable, '-B', '-u', '-m', 'reproduce.mhr_ce.run', 'contexts', '--seed', str(seed)],
                               cwd=ROOT, check=True, env=os.environ | {'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
            if not answer(seed):
                failed.append(seed)
        for _ in range(3):
            failed = [s for s in failed if not answer(s)]
            if not failed:
                break
            time.sleep(1800)
        if failed:
            raise RuntimeError(f'Answer step failed for seeds {failed}; rerun to resume')
        status(step='contexts_and_answers_done')
        return 0
    except BaseException as exc:
        checkpoint(OUT / 'status.json', {'state': 'failed', 'at': now(), 'error_type': type(exc).__name__, 'message': str(exc)})
        raise


def loaded(seed):
    import judge_run
    from minirag.prompt import PROMPTS
    paths = {'C0': ST_OUT / f'test/C0_s{seed}/answers.csv', 'ST': ST_OUT / f'test/ST_s{seed}/answers.csv',
             'F': previous.folder(seed) / 'answers.csv', 'CE': folder(seed) / 'answers.csv'}
    return {arm: {r['query_id']: r for r in judge_run.load_rows(QUESTIONS, path, 'minirag', PROMPTS['fail_response'])}
            for arm, path in paths.items()}


def run_judge(questions, answers_path, run_dir, rounds):
    """Same +F retry policy, with a no-dotenv entry point and CE-only journals."""
    env = os.environ.copy()
    for key in ('GEMINI_API_KEY_ONLY', 'MINIRAG_SLM_SEED', 'MINIRAG_MAX_TOKENS', 'GEMINI_RPM'):
        env.pop(key, None)
    env.update(GEMINI_API_BASE='https://generativelanguage.googleapis.com/v1beta/openai/', GEMINI_TPM='3000',
               PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1')
    log = Path(run_dir).parent / 'judge_runner.log'
    for attempt in range(1, rounds + 1):
        if (Path(run_dir) / 'summary.json').exists():
            return
        cmd = [sys.executable, '-B', '-u', '-m', 'reproduce.mhr_ce.judge_entry', '--questions', str(questions),
               '--answers', str(answers_path), '--run-dir', str(run_dir)]
        if (Path(run_dir) / 'manifest.json').exists():
            cmd.append('--resume')
        with log.open('a', encoding='utf-8') as f:
            f.write(f'=== CE judge round {attempt} {now()} ===\n')
            f.flush()
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if not (Path(run_dir) / 'summary.json').exists():
        raise RuntimeError(f'{run_dir}: judge incomplete after {rounds} rounds; rerun to resume')


def judge_seed(seed, rounds):
    import judge_run
    arms = loaded(seed)
    delta = folder(seed) / 'judge_delta'
    qfields, questions = read_csv(QUESTIONS)
    afields, ceanswers = read_csv(folder(seed) / 'answers.csv')
    if len(questions) != len(ceanswers) or any(q['Question'] != a['Question'] or q['Gold Answer'] != a['Gold Answer']
                                              for q, a in zip(questions, ceanswers)):
        raise ValueError('CE answer CSV is not aligned with questions')
    pick = [i for i, q in enumerate(questions) if arms['CE'][q['query_id']]['pred'] not in
            tuple(arms[arm][q['query_id']]['pred'] for arm in ('C0', 'ST', 'F'))]
    qpath, apath = delta / 'questions.csv', delta / 'answers.csv'
    if not qpath.exists():
        previous.write_csv(qpath, qfields, [questions[i] for i in pick])
        previous.write_csv(apath, afields, [ceanswers[i] for i in pick])
    if pick:
        run_judge(qpath, apath, delta / 'run', rounds)
    journals = {'C0': judge_run.read_records(ST_OUT / f'test/C0_s{seed}/judge_gemini35/judge.jsonl'),
                'ST': judge_run.read_records(ST_OUT / f'test/ST_s{seed}/judge_delta/run/judge.jsonl'),
                'F': judge_run.read_records(previous.folder(seed) / 'judge_delta/run/judge.jsonl'),
                'CE': judge_run.read_records(delta / 'run/judge.jsonl')}
    rows = [arms['CE'][q['query_id']] for q in questions]
    latest, origin = {}, {}
    for row in rows:
        qid, pred = row['query_id'], row['pred']
        src = next((arm for arm in ('C0', 'ST', 'F') if pred == arms[arm][qid]['pred']), 'CE')
        origin[qid] = 'judged_CE' if src == 'CE' else f'{src}_identical_answer'
        for p in (1, 2, 3):
            record = journals[src].get((qid, p))
            if record is None or record['status'] != 'ok':
                raise RuntimeError(f'Seed {seed}: missing verdict {qid} pass {p} ({src})')
            latest[(qid, p)] = record
    target = folder(seed) / 'judge_gemini35'
    data = []
    for row in rows:
        labels = [latest[(row['query_id'], p)]['label'] for p in (1, 2, 3)]
        data.append({'query_id': row['query_id'], 'Type': row['type'],
                     **{f'pass{i+1}': label for i, label in enumerate(labels)},
                     'majority': judge_run.majority(labels), 'verdict_origin': origin[row['query_id']]})
    previous.write_csv(target / 'per_question.csv', ['query_id', 'Type', 'pass1', 'pass2', 'pass3', 'majority', 'verdict_origin'], data)
    counts = {k: sum(v == k for v in origin.values()) for k in ('C0_identical_answer', 'ST_identical_answer', 'F_identical_answer', 'judged_CE')}
    offline.write(target / 'summary.json', {'status': 'completed', 'n_questions': len(rows), 'passes': 3,
                                           'groups': judge_run.summarize(rows, latest, 3), 'verdict_origin': counts})
    offline.write(target / 'provenance.json', {'judge_model': JUDGE, 'composed_at': now(), 'verdict_origin': counts,
        'ce_answers_sha256': sha(folder(seed) / 'answers.csv'),
        'journals_sha256': {arm: sha(path) for arm, path in (
            ('C0', ST_OUT / f'test/C0_s{seed}/judge_gemini35/judge.jsonl'),
            ('ST', ST_OUT / f'test/ST_s{seed}/judge_delta/run/judge.jsonl'),
            ('F', previous.folder(seed) / 'judge_delta/run/judge.jsonl'), ('CE', delta / 'run/judge.jsonl')) if path.exists()}})
    return counts


def finish(a):
    offline.setup()
    try:
        for seed in SEEDS:
            while not (folder(seed) / 'answer_summary.json').exists():
                status('status_judge.json', step='waiting_for_answers', seed=seed)
                time.sleep(600)
            if not (folder(seed) / 'judge_gemini35/summary.json').exists():
                status('status_judge.json', step='judge_CE_delta', seed=seed)
                print(json.dumps({'seed': seed, 'verdict_origin': judge_seed(seed, a.rounds)}), flush=True)
        status('status_judge.json', step='report')
        report()
        checkpoint(OUT / 'status_judge.json', {'state': 'completed', 'at': now(), 'judge': JUDGE})
        return 0
    except BaseException as exc:
        checkpoint(OUT / 'status_judge.json', {'state': 'failed', 'at': now(), 'error_type': type(exc).__name__, 'message': str(exc)})
        raise


def report():
    from reproduce.mhr_st.report import paired, quality
    gold = {r['query_id']: r for r in records(DATA / 'processed/gold_chunks.jsonl')}
    queries = {r['query_id']: r['Question'] for r in test_rows()}
    result = {'judge': JUDGE, 'n': len(queries), 'seeds': {}, 'model_F_sha256': sha(previous.MODEL),
              'model_CE_sha256': sha(MODEL), 'sd_definition': 'sample SD between generation seeds',
              'primary_comparison': 'CE_vs_F',
              'registered_rule': 'ACC rises in all three seeds and exact McNemar CE_vs_F p < 0.05 in each',
              'by_type_exploratory_no_multiplicity_correction': True}
    by_seed = {a: [] for a in ('C0', 'ST', 'F', 'CE')}
    for seed in SEEDS:
        arms, judge_sd = {}, {}
        paths = {'C0': ST_OUT / f'test/C0_s{seed}/judge_gemini35', 'ST': ST_OUT / f'test/ST_s{seed}/judge_gemini35',
                 'F': previous.folder(seed) / 'judge_gemini35', 'CE': folder(seed) / 'judge_gemini35'}
        for arm, path in paths.items():
            s = read(path / 'summary.json')
            if s['status'] != 'completed' or s['n_questions'] != len(queries):
                raise RuntimeError(f'{arm} seed {seed}: judge incomplete')
            _, rows = read_csv(path / 'per_question.csv')
            arms[arm] = {r['query_id']: r for r in rows}
            if len(rows) != len(queries) or set(arms[arm]) != set(queries) or any(r['majority'] not in
                    ('accurate', 'error', 'neither', 'no_majority') for r in rows):
                raise ValueError('Incomplete, unpaired or invalid per-question labels')
            by_seed[arm].append(quality(rows))
            judge_sd[arm] = s['groups']['all']['pass_mean_sd']
        sr = {'quality': {a: quality(list(v.values())) for a, v in arms.items()},
              'pass_mean_sd_judge': judge_sd,
              'verdict_origin': read(paths['CE'] / 'summary.json')['verdict_origin'],
              'CE_vs_F': paired(arms['F'], arms['CE']), 'CE_vs_C0': paired(arms['C0'], arms['CE']), 'by_type': {}}
        for typ in ('comparison', 'inference', 'temporal', 'null'):
            sub = {a: {q: r for q, r in v.items() if r['Type'] == typ} for a, v in arms.items()}
            sr['by_type'][typ] = {'quality': {a: quality(list(v.values())) for a, v in sub.items()},
                                  'CE_vs_F': paired(sub['F'], sub['CE'])}
        old = {r['query']: r for r in records(ST_OUT / f'archive_c0/qa/s{seed}/C0_s{seed}/context.jsonl')}
        selected = {'C0': {qid: old.get(q, {}).get('chunk_ids', []) for qid, q in queries.items()}}
        for arm, path in (('ST', ST_OUT / f'test/ST_s{seed}/contexts.jsonl'),
                          ('F', previous.folder(seed) / 'contexts.jsonl'), ('CE', folder(seed) / 'contexts.jsonl')):
            selected[arm] = {r['query_id']: r['chunk_ids'] for r in records(path)}
            if set(selected[arm]) != set(queries):
                raise ValueError(f'Incomplete {arm} selections')
        recall = {}
        for arm, chosen in selected.items():
            hits = total = complete = n = 0
            macro = []
            for qid, ids in chosen.items():
                facts = gold[qid]['evidence']
                if not facts:
                    continue
                found = sum(bool(set(ids) & set(e['chunk_ids'])) for e in facts)
                hits += found
                total += len(facts)
                complete += found == len(facts)
                n += 1
                macro.append(found / len(facts))
            recall[arm] = {'n': n, 'facts_hit': hits, 'gold_facts': total, 'micro': hits / total,
                           'macro': statistics.mean(macro), 'full': complete / n}
        sr['evidence_recall'] = recall
        result['seeds'][str(seed)] = sr
    result['generation_mean_sd'] = {arm: {k: {'mean': statistics.mean(r[k] for r in values),
                                            'sd': statistics.stdev(r[k] for r in values)} for k in values[0]}
                                    for arm, values in by_seed.items()}
    decision_key = 'CE_vs_F_all_three_seeds_positive_and_significant'
    result[decision_key] = all(s['CE_vs_F']['up'] > s['CE_vs_F']['down'] and s['CE_vs_F']['p_exact'] < .05
                               for s in result['seeds'].values())
    manifest = read(INDEX / 'index_manifest.json')
    result['index_unchanged'] = all(sha(INDEX / p) == h for p, h in manifest['file_sha256'].items())
    if not result['index_unchanged']:
        raise RuntimeError('Index changed during experiment')
    text = ['MultiHopRAG: C0 vs +ST vs +F vs +CE', f'Qwen2.5-3B | Judge: {JUDGE}',
            f'{len(queries):,} questions; 3 generation seeds; 3 judge passes per answer.',
            'ACC/ERR/Neither use majority labels; +/- is sample SD between generation seeds.', '',
            '| Configuration | ACC mean +/- SD | ERR | Neither | No majority | Evidence recall (micro) | Full evidence |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for arm, name in (('C0', 'C0'), ('ST', 'C0+ST'), ('F', 'C0+ST+F'), ('CE', 'C0+ST+F+CE')):
        g = result['generation_mean_sd'][arm]
        rec = statistics.mean(s['evidence_recall'][arm]['micro'] for s in result['seeds'].values())
        full = statistics.mean(s['evidence_recall'][arm]['full'] for s in result['seeds'].values())
        text.append(f"| {name} | {g['accurate']['mean']:.2f} +/- {g['accurate']['sd']:.2f} | {g['error']['mean']:.2f} | "
                    f"{g['neither']['mean']:.2f} | {g['no_majority']['mean']:.2f} | {rec:.3f} | {100 * full:.1f}% |")
    for key, title in (('CE_vs_F', 'CE vs F (primary)'), ('CE_vs_C0', 'CE vs C0')):
        text += ['', title, '| Seed | Delta ACC (pp) | Wrong -> correct | Correct -> wrong | Exact McNemar p |',
                 '|---|---:|---:|---:|---:|']
        for seed, sr in result['seeds'].items():
            p = sr[key]
            text.append(f"| {seed} | {p['delta_acc_pp']:+.2f} | {p['up']} | {p['down']} | {p['p_exact']:.6g} |")
    text += ['', 'CE over F: positive and significant in all three seeds.' if result[decision_key] else
             'CE over F: improvement is not confirmed under the registered three-seed rule.', '',
             'By-type CE vs F is exploratory, without multiplicity correction; see RESULTS.json.',
             'Subgroups, ERR/Neither, verdict origins and judge variation: RESULTS.json.',
             'Frozen index checksums verified unchanged.']
    offline.write(OUT / 'RESULTS.json', result)
    offline.new_text(OUT / 'BANG_SO_SANH.txt', '\n'.join(text) + '\n')
    print('\n'.join(text))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('step', choices=('freeze', 'contexts', 'answers', 'execute', 'finish', 'report'))
    ap.add_argument('--seed', type=int, choices=SEEDS, default=101)
    ap.add_argument('--rounds', type=int, default=6)
    a = ap.parse_args()
    if a.step in ('freeze', 'contexts', 'report'):
        from reproduce.mhr_ce.local_runtime import install
        install()
        offline.guard()
    else:
        offline.setup()
    if a.step == 'report':
        report()
        return 0
    return {'freeze': freeze, 'contexts': contexts, 'answers': answers, 'execute': execute, 'finish': finish}[a.step](a)


if __name__ == '__main__':
    raise SystemExit(main())
