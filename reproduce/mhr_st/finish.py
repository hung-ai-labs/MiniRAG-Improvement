"""Finish MultiHop-RAG C0 vs +ST with the pinned judge, then write the report.

Judge: gemini-3.5-flash-lite, the frozen strict-label judge_run.py (source/), 3 passes.

C0  every answer of every seed is judged; an existing journal is resumed, so
    verdicts already paid for are kept and only missing (query, pass) slots run.
ST  an ST answer whose text is identical to the C0 answer of the same query and
    generation seed shares C0's three verdicts: the judge would receive exactly
    the same input (question, gold answer, answer). This is the LiHua Table 1
    rule (one judgment per (id, prediction_sha256, pass)). Only answers that
    differ are judged, in ST_s<seed>/judge_delta/. ST_s<seed>/judge_gemini35/ is
    then composed from the two journals and records the origin of each verdict.

    python -m reproduce.mhr_st.finish            # resumable; rerun after any stop
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import JUDGE, OUT, ROOT, SEEDS, read, setup, sha, write

QUESTIONS = OUT / 'inputs/test.csv'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def status(**kw):
    write(OUT / 'status.json', {'state': 'running', 'at': now(), 'judge': JUDGE} | kw)


def run_judge(questions, answers, run_dir, rounds):
    """judge_run in a fresh process (key pool must not inherit SLM settings); resume until complete."""
    env = os.environ.copy()
    for key in ('GEMINI_API_KEY_ONLY', 'MINIRAG_SLM_SEED', 'MINIRAG_MAX_TOKENS', 'GEMINI_RPM'):
        env.pop(key, None)
    env['GEMINI_API_BASE'] = 'https://generativelanguage.googleapis.com/v1beta/openai/'
    env['GEMINI_TPM'] = '3000'
    log = Path(run_dir).parent / 'judge_runner.log'
    for attempt in range(1, rounds + 1):
        if (Path(run_dir) / 'summary.json').exists():
            return
        cmd = [sys.executable, '-u', '-m', 'reproduce.mhr_st.judge', '--questions', str(questions),
               '--answers', str(answers), '--run-dir', str(run_dir)]
        if (Path(run_dir) / 'manifest.json').exists():
            cmd.append('--resume')
        with log.open('a', encoding='utf-8') as f:
            f.write(f'=== finish.py round {attempt} {now()} ===\n')
            f.flush()
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    if not (Path(run_dir) / 'summary.json').exists():
        raise RuntimeError(f'{run_dir}: judge still incomplete after {rounds} rounds; rerun to resume')


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def differing(seed):
    """Query rows (in input order) whose ST answer text differs from C0's."""
    qfields, questions = read_csv(QUESTIONS)
    afields, c0 = read_csv(OUT / f'test/C0_s{seed}/answers.csv')
    _, st = read_csv(OUT / f'test/ST_s{seed}/answers.csv')
    assert len(questions) == len(c0) == len(st)
    out = []
    for q, a, b in zip(questions, c0, st):
        assert q['Question'] == a['Question'] == b['Question'] and q['Gold Answer'] == a['Gold Answer'] == b['Gold Answer']
        if a['minirag'] != b['minirag']:
            out.append((q, b))
    return qfields, afields, out


def judge_st_delta(seed, rounds):
    qfields, afields, rows = differing(seed)
    delta = OUT / f'test/ST_s{seed}/judge_delta'
    qpath, apath = delta / 'questions.csv', delta / 'answers.csv'
    if not qpath.exists():
        write_csv(qpath, qfields, [q for q, _ in rows])
        write_csv(apath, afields, [a for _, a in rows])
    run_judge(qpath, apath, delta / 'run', rounds)
    return len(rows)


def compose_st(seed):
    import judge_run
    from minirag.prompt import PROMPTS
    target = OUT / f'test/ST_s{seed}/judge_gemini35'
    c0_dir = OUT / f'test/C0_s{seed}/judge_gemini35'
    delta_dir = OUT / f'test/ST_s{seed}/judge_delta/run'
    rows = judge_run.load_rows(QUESTIONS, OUT / f'test/ST_s{seed}/answers.csv', 'minirag', PROMPTS['fail_response'])
    c0_rows = {r['query_id']: r for r in judge_run.load_rows(
        QUESTIONS, OUT / f'test/C0_s{seed}/answers.csv', 'minirag', PROMPTS['fail_response'])}
    c0 = judge_run.read_records(c0_dir / 'judge.jsonl')
    new = judge_run.read_records(delta_dir / 'judge.jsonl')
    latest, origin = {}, {}
    for r in rows:
        shared = r['pred'] == c0_rows[r['query_id']]['pred']
        origin[r['query_id']] = 'C0_identical_answer' if shared else 'judged_ST'
        for p in (1, 2, 3):
            rec = (c0 if shared else new).get((r['query_id'], p))
            if rec is None or rec['status'] != 'ok':
                raise RuntimeError(f'seed {seed}: missing verdict {r["query_id"]} pass {p}')
            latest[(r['query_id'], p)] = rec
    target.mkdir(parents=True, exist_ok=True)
    with (target / 'per_question.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['query_id', 'Type', 'pass1', 'pass2', 'pass3', 'majority', 'verdict_origin'])
        for r in rows:
            labels = [latest[(r['query_id'], p)]['label'] for p in (1, 2, 3)]
            w.writerow([r['query_id'], r['type'], *labels, judge_run.majority(labels), origin[r['query_id']]])
    counts = {k: sum(v == k for v in origin.values()) for k in ('C0_identical_answer', 'judged_ST')}
    write(target / 'summary.json', {'status': 'completed', 'n_questions': len(rows), 'passes': 3,
                                    'groups': judge_run.summarize(rows, latest, 3), 'verdict_origin': counts})
    write(target / 'provenance.json', {
        'judge_model': JUDGE, 'composed_at': now(), 'rule': __doc__.split('ST  ')[1].split('\n\n')[0],
        'verdict_origin': counts,
        'questions_sha256': sha(QUESTIONS), 'st_answers_sha256': sha(OUT / f'test/ST_s{seed}/answers.csv'),
        'c0_answers_sha256': sha(OUT / f'test/C0_s{seed}/answers.csv'),
        'c0_journal_sha256': sha(c0_dir / 'judge.jsonl'), 'delta_journal_sha256': sha(delta_dir / 'judge.jsonl')})
    return counts


def amendment():
    path = OUT / 'amendment_20261113_move_and_shared_verdicts.json'
    if not path.exists():
        write(path, {
            'date': '2026-11-13',
            'authorization': 'User asked to finish judging C0 + ST with gemini-3.5-flash-lite and move the '
                             'results into MiniRAG-Improvement.',
            'move': {'from': 'MiniRAG/runs/multihop_rag/st_20261112', 'to': 'MiniRAG-Improvement/review/mhr_st_20261112',
                     'left_behind': 'snapshots/ (dev worker index copies, 1.8 GB)',
                     'copied': ['MiniRAG/indexes/multihop_rag-qwen-full609 -> MultiHop-RAG-qwen-full609 (sha256 verified)',
                                'dataset/MultiHop-RAG (processed, splits, raw)',
                                'x20-x20/work/phase3/test_run/qa/s*/C0_s*/{answers,context,requests}.jsonl -> archive_c0/']},
            'judge_change': 'ST answers identical to the C0 answer (same query, same seed) share C0 verdicts '
                            'instead of being judged again; only differing ST answers are judged. Unchanged: '
                            'judge model, prompt, label contract, passes, C0 judging.',
            'state_before': 'C0_s101 judge journal 6263 records (6259 ok slots); nothing else judged.',
        })


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--rounds', type=int, default=3, help='judge_run resume rounds per cohort')
    args = ap.parse_args()
    setup()
    amendment()
    try:
        for seed in args.seeds:
            status(step='judge_C0', seed=seed)
            run_judge(QUESTIONS, OUT / f'test/C0_s{seed}/answers.csv', OUT / f'test/C0_s{seed}/judge_gemini35', args.rounds)
            status(step='judge_ST_delta', seed=seed)
            n = judge_st_delta(seed, args.rounds)
            counts = compose_st(seed)
            print(json.dumps({'seed': seed, 'st_differing_answers': n, 'verdict_origin': counts}), flush=True)
        if sorted(args.seeds) == sorted(SEEDS):
            status(step='report')
            from .report import report
            report()
            write(OUT / 'status.json', {'state': 'completed', 'at': now(), 'judge': JUDGE})
    except BaseException as exc:
        write(OUT / 'status.json', {'state': 'failed', 'at': now(), 'error_type': type(exc).__name__,
                                    'message': str(exc)})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
