"""Operator-only SET answers and judging. Never executed by the offline task.

Every writer exclusively creates a file. Retries use fresh attempt directories;
successful records are composed once into the canonical immutable outputs.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import json
import os
import random
import subprocess
import sys
import time
import uuid

from reproduce.mhr_ce import run as ce
from reproduce.mhr_f import run as f
from reproduce.mhr_set import offline
from reproduce.mhr_set import run as r
from reproduce.mhr_st.common import JUDGE, OUT as ST_OUT, ROOT, SEEDS, read, records, sha


UNPARSED_CYCLES = 5  # DEVIATIONS.md, 2026-11-18 amendment


def fresh(parent):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / (r.now().replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex)
    path.mkdir()
    return path


def status(name='status.json', **kw):
    event = {'state': 'running', 'at': r.now(), 'judge': JUDGE, **kw}
    offline.write(fresh(r.OUT / (Path(name).stem + '_events')) / 'event.json', event)


def terminal(name, state, **kw):
    value = {'state': state, 'at': r.now(), 'judge': JUDGE, **kw}
    if state == 'completed' and not (r.OUT / name).exists():
        offline.write(r.OUT / name, value)
    else:
        offline.write(fresh(r.OUT / (Path(name).stem + '_events')) / 'event.json', value)


def pause(seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        time.sleep(min(60, max(0, deadline - time.monotonic())))


def configure_slm(phased, seed):
    # Exact +F/+CE serving validation; redirect its log and append helper in memory.
    original_out, original_append = f.OUT, f.append
    f.OUT, f.append = r.OUT, offline.event
    try:
        f.configure_slm(phased, seed)
    finally:
        f.OUT, f.append = original_out, original_append


def answers(a):
    offline.setup()
    import phased_qa as phased
    from minirag.base import QueryParam
    dest = r.folder(a.seed)
    summary = read(dest / 'context_summary.json')
    r.frozen()
    if (not summary['checks']['passed'] or summary['contexts_sha256'] != sha(dest / 'contexts.jsonl')
            or summary['set_params_sha256'] != sha(r.PARAMS)):
        raise ValueError('SET contexts not verified; refusing GPU calls')
    if (dest / 'answer_summary.json').exists():
        if read(dest / 'answer_summary.json')['answers_sha256'] != sha(dest / 'answers.jsonl'):
            raise ValueError('Completed SET answers changed')
        return 0
    rows = r.test_rows()
    contexts = {v['query_id']: v for v in records(dest / 'contexts.jsonl')}
    wanted = {v['query_id'] for v in rows}
    if set(contexts) != wanted:
        raise ValueError('Incomplete SET contexts')
    oldpath = ce.folder(a.seed) / 'answers.jsonl'
    old = {v['query_id']: v for v in records(oldpath)}
    old_summary = read(ce.folder(a.seed) / 'answer_summary.json')
    if set(old) != wanted or old_summary['answers_sha256'] != sha(oldpath):
        raise ValueError('Incomplete or changed frozen CE answers')
    done = {}
    for path in sorted((dest / 'answer_attempts').glob('*/*.jsonl')):
        if path.name not in ('answers.jsonl', 'reused.jsonl'):
            continue
        for value in records(path):
            q = value['query_id']
            if q not in wanted or (q in done and done[q] != value):
                raise ValueError('Conflicting immutable answer attempt')
            done[q] = value
    attempt = fresh(dest / 'answer_attempts')
    reused = [old[row['query_id']] | {'source': 'frozen_CE_identical_context'}
              for row in rows if row['query_id'] not in done and contexts[row['query_id']]['same_as_ce']]
    offline.jsonl(attempt / 'reused.jsonl', reused)
    done.update({v['query_id']: v for v in reused})
    todo = [row for row in rows if row['query_id'] not in done]
    if todo:
        phased.apply_env('C0', a.seed)
        configure_slm(phased, a.seed)

        async def generate():
            rag = phased.build(attempt / 'answer_work', llm_max_async=32)
            await phased.answer_phase(rag.llm_model_func, todo, contexts, 'mini', attempt / 'answers.jsonl',
                                      QueryParam().response_type)
        # Legacy writes are converted to exclusive creation by run.main().
        phased.traced(attempt, generate)
        done.update({v['query_id']: v for v in records(attempt / 'answers.jsonl')})
    if set(done) != wanted:
        raise ValueError('Incomplete SET answers')
    ordered = [done[row['query_id']] for row in rows]
    if not (dest / 'answers.jsonl').exists():
        offline.jsonl(dest / 'answers.jsonl', ordered)
    elif records(dest / 'answers.jsonl') != ordered:
        raise ValueError('Existing SET answers differ from immutable attempts')
    csv_rows = [{'Question': row['Question'], 'Gold Answer': row['Gold Answer'],
                 'minirag': done[row['query_id']]['answer']} for row in rows]
    if not (dest / 'answers.csv').exists():
        offline.csv_write(dest / 'answers.csv', ['Question', 'Gold Answer', 'minirag'], csv_rows)
    elif r.read_csv(dest / 'answers.csv')[1] != csv_rows:
        raise ValueError('Existing SET answer CSV differs')
    offline.write(dest / 'answer_summary.json', {
        'questions': len(rows), 'answers_sha256': sha(dest / 'answers.jsonl'),
        'answers_csv_sha256': sha(dest / 'answers.csv'), 'ce_answers_sha256': sha(oldpath),
        'contexts_sha256': sha(dest / 'contexts.jsonl'),
        'reused_identical_CE_context': sum(v.get('source') == 'frozen_CE_identical_context' for v in ordered),
        'context_status': {s: sum(v.get('context_status') == s for v in ordered)
                           for s in ('ok', 'no_context', 'early_exit', 'context_overflow')}})
    return 0


def child(step, seed):
    return [sys.executable, '-B', '-u', '-m', 'reproduce.mhr_set.run', step, '--seed', str(seed)]


def execute(a):
    r.frozen()
    if (r.OUT / 'status.json').exists() and read(r.OUT / 'status.json')['state'] == 'completed':
        return 0
    env = os.environ | {'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    try:
        failed = []
        for seed in SEEDS:
            status(step='contexts', seed=seed)
            dest = r.folder(seed)
            if (dest / 'context_summary.json').exists():
                s = read(dest / 'context_summary.json')
                if (not s['checks']['passed'] or s['contexts_sha256'] != sha(dest / 'contexts.jsonl')
                        or s['set_params_sha256'] != sha(r.PARAMS)):
                    raise ValueError('Completed SET contexts changed')
            else:
                subprocess.run(child('contexts', seed), cwd=ROOT, env=env, check=True)
            status(step='answers', seed=seed)
            if subprocess.run(child('answers', seed), cwd=ROOT, env=env).returncode:
                failed.append(seed)
        for _ in range(3):
            failed = [seed for seed in failed if subprocess.run(child('answers', seed), cwd=ROOT, env=env).returncode]
            if not failed:
                break
            pause(1800)
        if failed:
            raise RuntimeError(f'Answer step failed for seeds {failed}; rerun to resume')
        terminal('status.json', 'completed', step='contexts_and_answers_done')
        return 0
    except BaseException as exc:
        terminal('status.json', 'failed', error_type=type(exc).__name__, message=str(exc))
        raise


def loaded(seed):
    import judge_run
    from minirag.prompt import PROMPTS
    paths = {'C0': ST_OUT / f'test/C0_s{seed}/answers.csv', 'ST': ST_OUT / f'test/ST_s{seed}/answers.csv',
             'F': f.folder(seed) / 'answers.csv', 'CE': ce.folder(seed) / 'answers.csv',
             'SET': r.folder(seed) / 'answers.csv'}
    return {arm: {v['query_id']: v for v in judge_run.load_rows(ce.QUESTIONS, path, 'minirag', PROMPTS['fail_response'])}
            for arm, path in paths.items()}


def journals(seed):
    return {'C0': ST_OUT / f'test/C0_s{seed}/judge_gemini35/judge.jsonl',
            'ST': ST_OUT / f'test/ST_s{seed}/judge_delta/run/judge.jsonl',
            'F': f.folder(seed) / 'judge_delta/run/judge.jsonl',
            'CE': ce.folder(seed) / 'judge_delta/run/judge.jsonl',
            'SET': r.folder(seed) / 'judge_delta/run/judge.jsonl'}


def run_judge(questions, answers_path, run_dir, rounds):
    """CE retry policy with successful passes resumed through immutable attempts."""
    import judge_run
    run_dir = Path(run_dir)
    expected = {'questions_sha256': sha(questions), 'answers_sha256': sha(answers_path),
                'judge_model': JUDGE, 'passes': 3}
    if (run_dir / 'summary.json').exists():
        if read(run_dir / 'manifest.json') != expected:
            raise ValueError('Completed judge inputs changed')
        return
    attempts = run_dir.parent / (run_dir.name + '_attempts')
    prior = sorted(attempts.glob('*/run/judge.jsonl'))
    env = os.environ.copy()
    for key in ('GEMINI_API_KEY_ONLY', 'MINIRAG_SLM_SEED', 'MINIRAG_MAX_TOKENS', 'GEMINI_RPM'):
        env.pop(key, None)
    env.update(GEMINI_API_BASE='https://generativelanguage.googleapis.com/v1beta/openai/', GEMINI_TPM='3000',
               PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1')
    from minirag.prompt import PROMPTS
    rows = judge_run.load_rows(questions, answers_path, 'minirag', PROMPTS['fail_response'])
    wanted = {(v['query_id'], p) for v in rows for p in (1, 2, 3)}
    # Every attempt is bound to the same input files before its prior records are reused.
    for path in prior:
        if read(path.parent.parent / 'inputs.json') != expected:
            raise ValueError('Judge attempt inputs changed')
    latest, failed = {}, {}

    def absorb(path):
        for k, v in judge_run.read_records(path).items():
            if v['status'] == 'ok':
                latest[k] = v
            elif v['status'] == 'parse_failed':
                failed.setdefault(k, []).append(v)
    for path in prior:
        absorb(path)
    stubborn = lambda: [k for k in wanted - set(latest) if len(failed.get(k, ())) >= UNPARSED_CYCLES]
    for _ in range(rounds):
        if set(latest) == wanted or set(stubborn()) == wanted - set(latest):
            break
        attempt = fresh(attempts)
        offline.write(attempt / 'inputs.json', expected)
        cmd = [sys.executable, '-B', '-u', '-m', 'reproduce.mhr_set.judge_entry',
               '--questions', str(questions), '--answers', str(answers_path), '--run-dir', str(attempt / 'run')]
        for path in prior:
            cmd.extend(['--prior-journal', str(path)])
        with (attempt / 'judge_runner.log').open('x', encoding='utf-8') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        path = attempt / 'run/judge.jsonl'
        if path.exists():
            prior.append(path)
            absorb(path)
    # DEVIATIONS.md amendment 2026-11-18: still parse_failed after UNPARSED_CYCLES cycles -> 'unparsed' (not accurate).
    for k in stubborn():
        latest[k] = failed[k][-1] | {'status': 'unparsed', 'label': 'unparsed',
                                     'parse_failed_cycles': len(failed[k])}
    if set(latest) != wanted:
        raise RuntimeError(f'{run_dir}: judge incomplete after {rounds} rounds; rerun to resume')
    canonical = [latest[(v['query_id'], p)] for v in rows for p in (1, 2, 3)]
    if not (run_dir / 'judge.jsonl').exists():
        offline.jsonl(run_dir / 'judge.jsonl', canonical)
    elif records(run_dir / 'judge.jsonl') != canonical:
        raise ValueError('Canonical judge journal differs')
    if not (run_dir / 'manifest.json').exists():
        offline.write(run_dir / 'manifest.json', expected)
    elif read(run_dir / 'manifest.json') != expected:
        raise ValueError('Canonical judge manifest differs')
    offline.write(run_dir / 'summary.json', {'status': 'completed', 'n_questions': len(rows), 'passes': 3,
                  'groups': judge_run.summarize(rows, latest, 3),
                  'attempt_journals_sha256': {str(p.relative_to(r.OUT)): sha(p) for p in prior}, 'completed_at': r.now()})


def ensure_csv(path, fields, values):
    if Path(path).exists():
        old_fields, old = r.read_csv(path)
        if old_fields != fields or old != values:
            raise ValueError(f'Existing CSV changed: {path}')
    else:
        offline.csv_write(path, fields, values)


def judge_seed(seed, rounds):
    import judge_run
    arms = loaded(seed)
    delta = r.folder(seed) / 'judge_delta'
    qfields, questions = r.read_csv(ce.QUESTIONS)
    afields, answers_rows = r.read_csv(r.folder(seed) / 'answers.csv')
    # load_rows above verifies alignment; compare also before writing delta files.
    if len(questions) != len(answers_rows) or any(q['Question'] != a['Question'] or q['Gold Answer'] != a['Gold Answer']
                                                for q, a in zip(questions, answers_rows)):
        raise ValueError('SET answer CSV is not aligned')
    pick = [i for i, q in enumerate(questions) if arms['SET'][q['query_id']]['pred'] not in
            tuple(arms[arm][q['query_id']]['pred'] for arm in ('C0', 'ST', 'F', 'CE'))]
    qpath, apath = delta / 'questions.csv', delta / 'answers.csv'
    ensure_csv(qpath, qfields, [questions[i] for i in pick])
    ensure_csv(apath, afields, [answers_rows[i] for i in pick])
    if pick:
        run_judge(qpath, apath, delta / 'run', rounds)
    paths = journals(seed)
    source_records = {arm: judge_run.read_records(path) for arm, path in paths.items()}
    rows = [arms['SET'][q['query_id']] for q in questions]
    latest, origins, data = {}, {}, []
    for row in rows:
        qid, pred = row['query_id'], row['pred']
        source = next((arm for arm in ('C0', 'ST', 'F', 'CE') if pred == arms[arm][qid]['pred']), 'SET')
        origins[qid] = 'judged_SET' if source == 'SET' else f'{source}_identical_answer'
        labels = []
        for p in (1, 2, 3):
            value = source_records[source].get((qid, p))
            if value is None or value['status'] not in ('ok', 'unparsed'):
                raise RuntimeError(f'Seed {seed}: missing verdict {qid} pass {p} ({source})')
            latest[qid, p] = value
            labels.append(value['label'])
        data.append({'query_id': qid, 'Type': row['type'], **{f'pass{p}': labels[p-1] for p in (1, 2, 3)},
                     'majority': judge_run.majority(labels), 'verdict_origin': origins[qid]})
    counts = {k: sum(v == k for v in origins.values())
              for k in ('C0_identical_answer', 'ST_identical_answer', 'F_identical_answer', 'CE_identical_answer', 'judged_SET')}
    target = r.folder(seed) / 'judge_gemini35'
    ensure_csv(target / 'per_question.csv',
               ['query_id', 'Type', 'pass1', 'pass2', 'pass3', 'majority', 'verdict_origin'], data)
    if not (target / 'provenance.json').exists():
        offline.write(target / 'provenance.json', {'judge_model': JUDGE, 'composed_at': r.now(), 'verdict_origin': counts,
                      'set_answers_sha256': sha(r.folder(seed) / 'answers.csv'),
                      'journals_sha256': {a: sha(p) for a, p in paths.items() if p.exists()}})
    offline.write(target / 'summary.json', {'status': 'completed', 'n_questions': len(rows), 'passes': 3,
                  'groups': judge_run.summarize(rows, latest, 3), 'verdict_origin': counts})
    return counts


def drift(rounds):
    """Registered 200 CE seed-101 answers rejudged on the SET judging date."""
    dest = r.OUT / 'secondary_checks/ce_drift'
    qfields, questions = r.read_csv(ce.QUESTIONS)
    afields, old = r.read_csv(ce.folder(101) / 'answers.csv')
    if len(questions) != 2156 or len(old) != 2156:
        raise ValueError('Drift sample requires the frozen 2156 questions')
    indices = random.Random(13).sample(range(2156), 200)
    qpath, apath = dest / 'questions.csv', dest / 'answers.csv'
    ensure_csv(qpath, qfields, [questions[i] for i in indices])
    ensure_csv(apath, afields, [old[i] for i in indices])
    if not (dest / 'sample.json').exists():
        offline.write(dest / 'sample.json', {'seed': 13, 'indices': indices, 'ce_answers_sha256': sha(ce.folder(101) / 'answers.csv'),
                      'started_at': r.now(), 'purpose': 'Rejudge CE seed-101 on SET judging date'})
    run_judge(qpath, apath, dest / 'run', rounds)


def finish(a):
    offline.setup()
    try:
        for seed in SEEDS:
            while not (r.folder(seed) / 'answer_summary.json').exists():
                status('status_judge.json', step='waiting_for_answers', seed=seed)
                pause(600)
            if not (r.folder(seed) / 'judge_gemini35/summary.json').exists():
                status('status_judge.json', step='judge_SET_delta', seed=seed)
                print(json.dumps({'seed': seed, 'verdict_origin': judge_seed(seed, a.rounds)}), flush=True)
        status('status_judge.json', step='judge_CE_drift', seed=101)
        drift(a.rounds)
        status('status_judge.json', step='report')
        if not (r.OUT / 'RESULTS.json').exists():
            r.report()
        terminal('status_judge.json', 'completed', step='finished')
        return 0
    except BaseException as exc:
        terminal('status_judge.json', 'failed', error_type=type(exc).__name__, message=str(exc))
        raise
