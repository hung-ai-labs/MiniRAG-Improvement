"""Oracle selection headroom: Qwen2.5-3B generates once, Gemini judges 3 passes.

Generation is identical to the QA20 pilot (reproduce/evidence_selector/lihua_qa.py):
rag_response template, response_type "Multiple Paragraphs", max_tokens 1024,
server-default sampling, seed 13 + question id (same across arms). Identical
request bodies share one cached Qwen response, so arms whose context equals
the baseline get the baseline's answer instead of a fresh sample.

Judge: gemini-flash-lite-latest, Step_2 JUDGE_PROMPT and label parsing,
3 independent passes, each in its own shuffled order, blind to arm. One
verdict per (question, prediction, pass) is shared across arms. A failed
judge call is never turned into a verdict: the run stops and resumes.

    .venv/bin/python -m reproduce.oracle_headroom.run generate
    .venv/bin/python -m reproduce.oracle_headroom.run judge
    .venv/bin/python -m reproduce.oracle_headroom.run report
"""
import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
import json
import os
from pathlib import Path
import random
import runpy
import statistics
import threading

from reproduce.graph_ablation.study import ROOT, read, lines, write, append, sha
from reproduce.graph_ablation.api import judge_template

OUT = ROOT / 'review/oracle_headroom_20261111'
ARMS = ('BASELINE', 'ORACLE_POOL', 'ORACLE_INJECT')
LABELS = ('accurate', 'error', 'neither')
JUDGE_MODEL = 'gemini-flash-lite-latest'
PASSES = (1,)  # user, 11/11: one judge pass is enough for this diagnostic (speed)
RESPONSE_TYPE = 'Multiple Paragraphs'


def contexts():
    build = read(OUT / 'build.json')
    assert sha((OUT / 'contexts.jsonl').read_bytes()) == build['contexts_sha256'], 'contexts changed after build'
    return lines(OUT / 'contexts.jsonl')


def generate(workers):
    from reproduce.evidence_selector.lihua_qa import Qwen, MODEL
    rows = contexts()
    template = runpy.run_path(str(ROOT / 'minirag/prompt.py'))['PROMPTS']['rag_response']
    spec = {'generator': MODEL, 'generation_rounds': 1, 'max_generation_tokens': 1024,
            'sampling': 'server defaults; seed 13+question id, same in every arm',
            'response_type': RESPONSE_TYPE, 'template_sha256': sha(template),
            'contexts_sha256': read(OUT / 'build.json')['contexts_sha256'],
            'runner_sha256': sha(Path(__file__).read_bytes())}
    if (OUT / 'generation_manifest.json').exists():
        old = read(OUT / 'generation_manifest.json')
        assert {k: v for k, v in old.items() if k != 'runner_sha256'} == \
               {k: v for k, v in spec.items() if k != 'runner_sha256'}, 'generation settings changed'
    else:
        write(OUT / 'generation_manifest.json', spec)
    done = {(r['id'], r['arm']) for r in lines(OUT / 'answers.jsonl')}
    todo = [r for r in rows if (r['id'], r['arm']) not in done]
    # baseline first, so identical oracle contexts reuse its cached response
    todo.sort(key=lambda r: (ARMS.index(r['arm']), r['id']))
    lock = threading.Lock()
    qwen = Qwen(OUT)

    def one(r):
        v = qwen.call(r, template, RESPONSE_TYPE)
        a = {**v, 'id': r['id'], 'arm': r['arm'], 'question': r['question'], 'gold': r['gold'],
             'type': r['type'], 'context_sha256': r['context_sha256'], 'generation_round': 1}
        with lock:
            append(OUT / 'answers.jsonl', a)
        return a

    for arm in ARMS:  # arm by arm: identical bodies hit the cache instead of racing
        batch = [r for r in todo if r['arm'] == arm]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, r) for r in batch]
            for n, f in enumerate(as_completed(futures), 1):
                f.result()
                if n % 25 == 0 or n == len(batch):
                    print(f'{arm}: {n}/{len(batch)}', flush=True)
    print(json.dumps({'answers': len(lines(OUT / 'answers.jsonl'))}))


def parse(text):
    """Step_2_evaluate.judge label parsing, unchanged."""
    v = (text or '').strip().lower()
    for label in LABELS:
        if label in v:
            return label
    return 'neither'


async def judge_all(concurrency):
    for name in ('GEMINI_API_BASE', 'GEMINI_API_KEY_ONLY', 'MINIRAG_SLM_MODEL', 'MINIRAG_MAX_TOKENS'):
        os.environ.pop(name, None)  # the judge must reach Gemini, never the SLM endpoint
    os.environ.setdefault('GEMINI_RPM', '13')
    os.environ.setdefault('GEMINI_TPM', '3000')
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    from minirag.llm.gemini import gemini_complete_if_cache
    prompt_t = judge_template()
    answers = {(r['id'], r['arm']): r for r in lines(OUT / 'answers.jsonl')}
    planned = {(r['id'], r['arm']) for r in contexts()}
    assert set(answers) == planned, f'generation incomplete: {len(answers)}/{len(planned)}'
    unique = {}
    for a in answers.values():
        unique.setdefault((a['id'], a['prediction_sha256']), a)
    done = {(r['id'], r['prediction_sha256'], r['judge_pass']) for r in lines(OUT / 'judgments.jsonl')}
    jobs = []
    for p in PASSES:
        keys = sorted(unique)
        random.Random(1308 + p).shuffle(keys)
        jobs += [(p, k) for k in keys if (k[0], k[1], p) not in done]
    print(f'judge calls to make: {len(jobs)} (unique predictions {len(unique)})', flush=True)
    sem = asyncio.Semaphore(concurrency)
    failures = []

    async def one(p, key):
        a = unique[key]
        prompt = prompt_t.format(question=a['question'], gold=a['gold'], pred=a['prediction'])
        for attempt in range(6):
            async with sem:
                try:
                    raw = await gemini_complete_if_cache(JUDGE_MODEL, prompt)
                    break
                except Exception as e:  # noqa: BLE001
                    err = f'{type(e).__name__}: {e}'[:300]
            if attempt == 5:
                failures.append({'id': a['id'], 'pass': p, 'error': err})
                return
            await asyncio.sleep(60 * (attempt + 1))
        append(OUT / 'judgments.jsonl', {'id': a['id'], 'prediction_sha256': a['prediction_sha256'],
               'judge_pass': p, 'verdict': parse(raw), 'raw': (raw or '')[:64], 'judge_model': JUDGE_MODEL})

    for start in range(0, len(jobs), 200):
        await asyncio.gather(*(one(p, k) for p, k in jobs[start:start + 200]))
        print(f'judged {min(start + 200, len(jobs))}/{len(jobs)}', flush=True)
        if failures:
            break
    if failures:
        write(OUT / 'judge_failures.json', failures)
        raise SystemExit(f'{len(failures)} judge calls failed after retries; no verdict recorded for them')


def exact_p(b, c):
    n = b + c
    return min(1., 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.


def report():
    rows = contexts()
    meta = {(r['id'], r['arm']): r for r in rows}
    answers = {(r['id'], r['arm']): r for r in lines(OUT / 'answers.jsonl')}
    verdict = {}
    for j in lines(OUT / 'judgments.jsonl'):
        verdict[j['id'], j['prediction_sha256'], j['judge_pass']] = j['verdict']
    ids = sorted({r['id'] for r in rows})
    complete = [i for i in ids if all((i, a) in answers and all(
        (i, answers[i, a]['prediction_sha256'], p) in verdict for p in PASSES) for a in ARMS)]
    label = lambda i, a, p: verdict[i, answers[i, a]['prediction_sha256'], p]

    def majority(i, a):
        c = Counter(label(i, a, p) for p in PASSES)
        top, n = c.most_common(1)[0]
        return top if len(PASSES) == 1 or n >= 2 else 'neither'

    def rates(subset):
        out = {}
        for a in ARMS:
            per = {l: [100 * sum(label(i, a, p) == l for i in subset) / len(subset) for p in PASSES] for l in LABELS}
            out[a] = {l: {'mean': statistics.mean(v), 'sd': statistics.stdev(v) if len(v) > 1 else None}
                      for l, v in per.items()}
            out[a]['n'] = len(subset)
        return out

    def mcnemar(x, y, subset, lab):
        up = sum(majority(i, x) != lab and majority(i, y) == lab for i in subset)
        down = sum(majority(i, x) == lab and majority(i, y) != lab for i in subset)
        return {'became': up, 'ceased': down, 'net': up - down, 'p_exact': exact_p(up, down)}

    groups = {'all': complete,
              'Single': [i for i in complete if meta[i, 'BASELINE']['type'] == 'Single'],
              'Multi': [i for i in complete if meta[i, 'BASELINE']['type'] == 'Multi'],
              'baseline_full_evidence': [i for i in complete if meta[i, 'BASELINE']['full_evidence']],
              'baseline_missing_evidence': [i for i in complete if not meta[i, 'BASELINE']['full_evidence']],
              'pool_reorder_changes_context': [i for i in complete
                  if meta[i, 'ORACLE_POOL']['context_sha256'] != meta[i, 'BASELINE']['context_sha256']]}
    pairs = [('BASELINE', 'ORACLE_POOL'), ('BASELINE', 'ORACLE_INJECT'), ('ORACLE_POOL', 'ORACLE_INJECT')]
    result = {'status': 'COMPLETE' if len(complete) == len(ids) else 'PARTIAL',
              'questions': len(complete), 'planned': len(ids), 'judge': JUDGE_MODEL, 'judge_passes': len(PASSES),
              'generation_rounds': 1,
              'groups': {g: {'n': len(s), 'rates': rates(s) if len(s) > 1 else None} for g, s in groups.items()},
              'mcnemar_majority': {f'{x}->{y}': {g: {lab: mcnemar(x, y, s, lab) for lab in ('accurate', 'error')}
                                                 for g, s in groups.items() if s} for x, y in pairs},
              'full_evidence': read(OUT / 'build.json')['full_evidence']}
    write(OUT / 'results.json', result)
    print(json.dumps({g: {a: round(v['rates'][a]['accurate']['mean'], 2) for a in ARMS}
                      for g, v in result['groups'].items() if v['rates']}, indent=1))
    print(json.dumps({k: v['all'] for k, v in result['mcnemar_majority'].items()}, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('generate', 'judge', 'report'))
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--concurrency', type=int, default=8)
    args = ap.parse_args()
    if args.step == 'generate':
        generate(args.workers)
    elif args.step == 'judge':
        asyncio.run(judge_all(args.concurrency))
    else:
        report()
