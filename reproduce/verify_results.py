"""Kiểm chứng offline rằng Bảng 1 (ladder C0 -> baseline -> +F -> +CE -> +SET) là kết quả chạy thật.

Không gọi API nào, không ghi đè file kết quả nào (mọi bước tính lại chạy trong thư mục tạm).

    python -m reproduce.verify_results                 # ~2-5 phút, cả Qwen và Phi
    python -m reproduce.verify_results --recapture 20  # thêm: chạy lại truy hồi MiniRAG từ index
                                                       # cho 20 câu đầu (cần torch, tải all-MiniLM-L6-v2)

Với mỗi SLM (qwen, phi_ropefix), trên 435 câu ngoài dev:

  [1] TABLE     tính lại ACC/ERR/McNemar từ answers.jsonl + judgments_deepseek.jsonl
                -> phải trùng results.json đã lưu.
  [2] CONTEXT   dựng lại context của 5 cấu hình từ index + pool truy hồi + scorer
                -> sha256 của contexts.jsonl phải trùng build.json.
  [3] ANSWER    với từng câu trả lời: (a) context_sha256 khớp contexts.jsonl; (b) tính lại sha256
                của request body (prompt rag_response + context + câu hỏi + seed) -> phải trùng
                body_sha256; (c) raw.json là phản hồi gốc của server vLLM (id chatcmpl-*, model,
                timestamp created, usage) và chứa đúng câu trả lời.
  [4] JUDGE     với từng phán quyết: tính lại sha256 request gửi DeepSeek (prompt chấm + câu hỏi +
                gold + câu trả lời) -> phải có hoá đơn ok=True trong deepseek/calls.jsonl
                (usage token, chi phí do API trả về).
  [5] SCORER    huấn luyện lại scorer chỉ trên dev200 -> hệ số phải trùng model đã dùng cho non-dev.
  [6] RECAPTURE (tuỳ chọn) chạy lại truy hồi gốc của MiniRAG trên index của SLM, câu trả lời phân
                tích từ khoá lấy từ cache (LLM bị cấm gọi) -> context phải trùng từng byte.
"""
import argparse
import json
import os
import pickle
import runpy
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLMS = ('qwen', 'phi_ropefix')


def ok(name, cond, detail=''):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)
    return bool(cond)


def child(recapture):
    os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(ROOT / '.cache/minirag/tiktoken'))
    from reproduce.graph_ablation.study import lines, sha
    from reproduce.graph_ablation.api import judge_template
    from reproduce.oracle_headroom.run import RESPONSE_TYPE
    from reproduce.evidence_set import pipeline
    from reproduce.evidence_set.slm import SLM, SPEC, GEN_MODEL, QUICK
    from reproduce.evidence_set.deepseek_judge import LEDGER, MODEL as JMODEL
    split = 'nondev'
    real = pipeline.ladder(split)
    print(f"\n=== {SPEC['name']}  (index {SPEC['index']}, kết quả {real.relative_to(ROOT)})", flush=True)
    good = True
    with tempfile.TemporaryDirectory(prefix='verify-') as tmp:
        tmp = Path(tmp)
        (tmp / 'ladder').mkdir()
        for f in ('answers.jsonl', 'contexts.jsonl', 'judgments_deepseek.jsonl', 'build.json'):
            shutil.copy2(real / f, tmp / 'ladder' / f)
        pipeline.SPLIT[split] = {**pipeline.SPLIT[split], 'dir': tmp}

        # [1] TABLE
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.report(split)
        a, b = json.loads((tmp / 'ladder/results.json').read_text()), json.loads((real / 'results.json').read_text())
        t = a['table']['all']
        good &= ok('[1] TABLE tính lại == results.json', a == b,
                   ', '.join(f"{arm} {t[arm]['accurate']['mean']:.2f}" for arm in pipeline.ARMS))

        # [2] CONTEXT
        stored = json.loads((real / 'build.json').read_text())['contexts_sha256']
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.build(split)
        got = json.loads((tmp / 'ladder/build.json').read_text())['contexts_sha256']
        good &= ok('[2] CONTEXT dựng lại, sha256 == build.json', got == stored, got[:16])

    # [3] ANSWER
    template = runpy.run_path(str(ROOT / 'minirag/prompt.py'))['PROMPTS']['rag_response']
    ctx = {(r['id'], r['arm']): r for r in lines(real / 'contexts.jsonl')}
    cache = real.parent / 'qwen_cache'
    answers = lines(real / 'answers.jsonl')
    bad, created, served, fail = Counter(), [], Counter(), 0
    for r in answers:
        c = ctx[r['id'], r['arm']]
        if r['context_sha256'] != c['context_sha256']:
            bad['context_sha256'] += 1
        if c['context'] is None:
            fail += 1
            continue
        k = r['id'] + 100000 * (r['round'] - 1)
        body = {'model': GEN_MODEL, 'messages': [
            {'role': 'system', 'content': template.format(context_data=c['context'], response_type=RESPONSE_TYPE)},
            {'role': 'user', 'content': r['question']}], 'max_tokens': 1024, 'seed': 13 + k}
        h = sha(json.dumps(body, sort_keys=True, ensure_ascii=False))
        if h != r['body_sha256']:
            bad['body_sha256'] += 1
            continue
        d = cache / h
        resp = json.loads((d / 'response.json').read_text())
        raw = json.loads((d / 'raw.json').read_text())
        if resp['prediction'] != r['prediction'] or sha(r['prediction']) != r['prediction_sha256']:
            bad['prediction'] += 1
        if raw['choices'][0]['message']['content'] != r['prediction'] or not raw['id'].startswith('chatcmpl-'):
            bad['raw'] += 1
        served[raw['model']] += 1
        created.append(raw['created'])
    span = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    rounds = sorted({r['round'] for r in answers})
    good &= ok('[3] ANSWER context -> request body -> phản hồi server', not bad,
               f"{len(answers)} câu trả lời ({len(rounds)} lượt sinh), {len(set(r['body_sha256'] for r in answers if r['body_sha256']))} "
               f"request khác nhau, model server {dict(served)}, created {span(min(created))} -> {span(max(created))}"
               + (f", {fail} fail_response không gọi LLM" if fail else '') + (f', LỖI {dict(bad)}' if bad else ''))

    # [4] JUDGE
    jt = judge_template()
    receipts = {}
    for x in lines(LEDGER / 'calls.jsonl'):
        if x.get('ok'):
            receipts.setdefault(x['request_sha256'], x)
    pred = {(r['id'], r['prediction_sha256']): r for r in answers}
    judg = lines(real / 'judgments_deepseek.jsonl')
    miss, cost, tok = 0, 0., 0
    for j in judg:
        a = pred[j['id'], j['prediction_sha256']]
        body = {'model': JMODEL, 'messages': [{'role': 'user', 'content': jt.format(
            question=a['question'], gold=a['gold'], pred=a['prediction'])}], 'max_tokens': 16,
                'thinking': {'type': 'disabled'}}
        rc = receipts.get(sha(json.dumps(body)))
        if rc is None:
            miss += 1
        else:
            cost += rc['cost_usd']
            tok += rc['usage']['total_tokens']
    passes = Counter(j['judge_pass'] for j in judg)
    # identical prompts (same answer judged in 2 passes) share one request hash; count distinct receipts
    good &= ok('[4] JUDGE mỗi phán quyết có hoá đơn DeepSeek', miss == 0,
               f"{len(judg)} phán quyết (pass {dict(sorted(passes.items()))}), "
               f"{len({(j['id'], j['prediction_sha256']) for j in judg})} câu trả lời khác nhau"
               + (f', THIẾU {miss}' if miss else ''))

    # [5] SCORER
    from reproduce.evidence_set import quick_scorer
    with tempfile.TemporaryDirectory(prefix='verify-scorer-') as tmp:
        quick_scorer.QUICK = Path(tmp)
        argv, sys.argv = sys.argv, ['quick_scorer', '--fit-all']
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            quick_scorer.main()
        sys.argv = argv
        import numpy as np
        diffs = {}
        for m in ('F', 'CE0+F'):
            new = pickle.load(open(Path(tmp) / f'models/{m}_allfolds.pkl', 'rb'))
            old = pickle.load(open(QUICK / f'models/{m}_allfolds.pkl', 'rb'))
            lr_n, lr_o = new.steps[-1][1], old.steps[-1][1]
            diffs[m] = float(max(np.abs(lr_n.coef_ - lr_o.coef_).max(), np.abs(lr_n.intercept_ - lr_o.intercept_).max()))
        good &= ok('[5] SCORER huấn luyện lại trên dev200 == model đã dùng', all(v < 1e-6 for v in diffs.values()),
                   ', '.join(f'{m} max|Δhệ số|={v:.1e}' for m, v in diffs.items()))

    # [6] RECAPTURE
    if recapture:
        good &= recapture_check(recapture)
    return good


def recapture_check(limit):
    """Native MiniRAG retrieval on the SLM's own index, keyword replies from cache, LLM forbidden."""
    import asyncio
    from reproduce.graph_ablation.study import lines
    from reproduce.evidence_set import pool
    from reproduce.path.path_common import forbidden_llm
    import minirag.llm.gemini as gem
    good, src = True, pool.OUT
    for profile, tag in (('baseline', ''), ('c0', '_c0')):
        stored = {}
        for k in range(4):
            stored.update({r['id']: r for r in lines(src / f'pools_nondev{tag}_{k}of4.jsonl')})
        with tempfile.TemporaryDirectory(prefix='verify-pool-') as tmp:
            tmp = Path(tmp)
            for k in range(4):
                shutil.copy2(src / f'kw_cache_nondev_{k}of4.jsonl', tmp)
            # one process here (shard 0/1) reads the cache of all 4 original shards
            (tmp / 'kw_cache_nondev_0of1.jsonl').write_text(
                ''.join((tmp / f'kw_cache_nondev_{k}of4.jsonl').read_text() for k in range(4)))
            pool.OUT = tmp
            pool.slm_parser_env = lambda: None
            gem.gemini_complete = forbidden_llm  # a cache miss would raise instead of calling the SLM
            os.environ['MINIRAG_KW_CACHE_ONLY'] = '1'
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                asyncio.run(pool.run(limit, 'nondev', (0, 1), profile))
            new = {r['id']: r for r in lines(tmp / f'pools_nondev{tag}_0of1.jsonl')}
        same = sum(new[i]['context_sha256'] == stored[i]['context_sha256'] and
                   new[i]['baseline_kept'] == stored[i]['baseline_kept'] for i in new)
        good &= ok(f'[6] RECAPTURE {profile}: truy hồi lại từ index == pool đã lưu', same == len(new) == limit,
                   f'{same}/{len(new)} câu trùng từng byte')
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recapture', type=int, default=0, metavar='N', help='re-run native retrieval for the first N questions')
    ap.add_argument('--slm', choices=SLMS, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.slm:
        sys.exit(0 if child(a.recapture) else 1)
    codes = []
    for slm in SLMS:
        env = {**os.environ, 'MINIRAG_EVSET_SLM': slm}
        cmd = [sys.executable, '-m', 'reproduce.verify_results', '--slm', slm, '--recapture', str(a.recapture)]
        codes.append(subprocess.run(cmd, cwd=ROOT, env=env).returncode)
    print('\nKẾT LUẬN:', 'TẤT CẢ KIỂM TRA ĐỀU ĐẠT' if not any(codes) else 'CÓ KIỂM TRA KHÔNG ĐẠT (xem FAIL ở trên)')
    sys.exit(1 if any(codes) else 0)


if __name__ == '__main__':
    main()
