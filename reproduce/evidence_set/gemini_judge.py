"""Gemini judge for the Table 1 ladder (CLAUDE.md protocol): gemini-flash-lite-latest, 3 passes.

Exactly the Step_2_evaluate.py path: its JUDGE_PROMPT, gemini_complete_if_cache (12-key pool,
token/min limiter, free tier), its label parsing. Blind and arm-agnostic: the judge sees
question, gold and answer only; calls are shuffled across arms.

Differences from Step_2, both bookkeeping only:
  - one judgment per unique (id, prediction_sha256, pass), so identical answers in two arms
    share a verdict (as the DeepSeek judge does);
  - a call that fails 6 times is NOT written (Step_2 writes JUDGE_FAILED); the run stays
    resumable and report() refuses to score a round with a missing verdict.
Output: <ladder>/judgments_gemini.jsonl {id, prediction_sha256, judge_pass, verdict, raw, judge_model}
"""
import asyncio
import os
import random

# the judge must reach Google, never the SLM endpoint the reader uses
for _k in ('GEMINI_API_BASE', 'GEMINI_API_KEY_ONLY', 'MINIRAG_MAX_TOKENS', 'MINIRAG_SLM_SEED'):
    os.environ.pop(_k, None)

from reproduce.graph_ablation.api import judge_template          # noqa: E402  Step_2 JUDGE_PROMPT
from reproduce.graph_ablation.study import lines, append          # noqa: E402
from reproduce.oracle_headroom.run import parse                   # noqa: E402  Step_2 label parsing

MODEL = 'gemini-flash-lite-latest'
PASSES = (1, 2, 3)


def judge_answers(answers, out_path, workers=4):
    from minirag.llm.gemini import gemini_complete_if_cache
    unique = {}
    for a in answers:
        unique.setdefault((a['id'], a['prediction_sha256']), a)
    done = {(j['id'], j['prediction_sha256'], j['judge_pass']) for j in lines(out_path)}
    todo = [(k, p) for k in sorted(unique) for p in PASSES if (*k, p) not in done]
    random.Random(1309).shuffle(todo)
    print(f'Gemini judge: {len(unique)} unique answers x {len(PASSES)} passes, {len(todo)} calls to make', flush=True)
    if not todo:
        return
    template = judge_template()
    failed = []

    async def one(sem, item):
        (i, h), p = item
        a = unique[i, h]
        prompt = template.format(question=a['question'], gold=a['gold'], pred=a['prediction'])
        for attempt in range(6):
            async with sem:
                try:
                    raw = await gemini_complete_if_cache(MODEL, prompt)
                    break
                except Exception as e:  # noqa: BLE001  Step_2 policy: wait and retry, never invent a verdict
                    err = e
            await asyncio.sleep(60 * (attempt + 1))
        else:
            failed.append((i, h, p, repr(err)[:200]))
            return
        append(out_path, {'id': i, 'prediction_sha256': h, 'judge_pass': p, 'verdict': parse(raw),
                          'raw': (raw or '')[:64], 'judge_model': MODEL})

    async def main():
        sem = asyncio.Semaphore(workers)
        for s in range(0, len(todo), 200):
            await asyncio.gather(*(one(sem, t) for t in todo[s:s + 200]))
            print(f'judged {min(s + 200, len(todo))}/{len(todo)}', flush=True)

    asyncio.run(main())
    if failed:
        raise SystemExit(f'{len(failed)} Gemini judge calls failed after 6 attempts; rerun to resume. First: {failed[0]}')
