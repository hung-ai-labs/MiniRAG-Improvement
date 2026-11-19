"""Cross-encoder logits for every (question, pool chunk) pair of the saved +F pools.

    python -m reproduce.mhr_ce.scores --split dev     # dev400 pools (seed 101)
    python -m reproduce.mhr_ce.scores --split test    # union of the test pools of seeds 101/202/303

Writes review/mhr_ce_20261116/ce_logits_<split>.jsonl ({query_id, logits: {chunk_id: logit}}),
resumable. A CE score only depends on (question, chunk text), so one score serves every seed.
Before the GPU run, 32 pairs are also scored on the local CPU; the max |GPU - CPU| gap is logged
in ce_check_<split>.json and must stay below 1e-3.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from reproduce.mhr_st.common import INDEX, OUT as ST_OUT, ROOT, read, records, sha, write
from reproduce.mhr_f.pool import OUT as F_OUT, hf_cache

OUT = ROOT / 'review' / 'mhr_ce_20261116'
SEEDS = (101, 202, 303)
PER_CALL = 24


def pools(split):
    paths = [F_OUT / 'dev/s101/pools.jsonl'] if split == 'dev' else [F_OUT / f'test/s{s}/pools.jsonl' for s in SEEDS]
    need = {}
    for p in paths:
        for r in records(p):
            need.setdefault(r['query_id'], dict.fromkeys([]))
            need[r['query_id']].update(dict.fromkeys(r['final_full']))
    return {q: list(v) for q, v in need.items()}, {str(p.relative_to(ROOT)): sha(p) for p in paths}


def questions(split):
    with (ST_OUT / f'inputs/{split}.csv').open(encoding='utf-8-sig', newline='') as f:
        return {r['query_id']: r['Question'] for r in csv.DictReader(f)}


def cpu_scores(pairs):
    os.environ['HF_HUB_CACHE'] = str(hf_cache())
    import torch
    from sentence_transformers import CrossEncoder
    from reproduce.mhr_ce.modal_ce import CE_MODEL, REVISION
    m = CrossEncoder(CE_MODEL, revision=REVISION, device='cpu', max_length=512)
    return m.predict(pairs, batch_size=32, show_progress_bar=False, activation_fn=torch.nn.Identity()).ravel().tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', choices=('dev', 'test'), required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    need, pool_sha = pools(a.split)
    qtext = questions(a.split)
    assert set(need) == set(qtext), 'pools and questions differ'
    raw = read(INDEX / 'kv_store_text_chunks.json')
    text_path = OUT / 'chunks_text.json'
    if not text_path.exists():
        write(text_path, {c: r['content'] for c, r in raw.items()})
    text = read(text_path)
    assert all(text[c] == raw[c]['content'] for c in raw), 'chunk texts changed'
    dest = OUT / f'ce_logits_{a.split}.jsonl'
    done = {r['query_id']: r['logits'] for r in records(dest)} if dest.exists() else {}
    todo = [q for q in sorted(need) if not set(need[q]) <= set(done.get(q, {}))]
    print(json.dumps({'split': a.split, 'questions': len(need), 'pairs': sum(map(len, need.values())),
                      'todo_questions': len(todo)}), flush=True)

    import modal
    from reproduce.mhr_ce import modal_ce
    Scorer = modal_ce.build(str(text_path))
    first = next(q for q in sorted(need) if need[q])
    probe = [(qtext[first], text[c]) for c in need[first][:32]]
    with modal.enable_output(), modal_ce.app.run():
        scorer = Scorer()
        gpu = scorer.score.remote([(first, qtext[first], need[first][:32])])[0][1]
        cpu = cpu_scores(probe)
        gap = max(abs(gpu[c] - v) for c, v in zip(need[first][:32], cpu))
        write(OUT / f'ce_check_{a.split}.json', {'query_id': first, 'pairs': len(cpu), 'max_abs_gpu_cpu_gap': gap,
                                                 'model': modal_ce.CE_MODEL, 'revision': modal_ce.REVISION,
                                                 'gpu': modal_ce.GPU, 'pools_sha256': pool_sha})
        print(json.dumps({'gpu_cpu_gap': gap}), flush=True)
        if gap > 1e-3:
            raise RuntimeError(f'GPU and CPU logits differ by {gap}')
        batches = [[(q, qtext[q], need[q]) for q in todo[i:i + PER_CALL]] for i in range(0, len(todo), PER_CALL)]
        n = 0
        with dest.open('a', encoding='utf-8', newline='\n') as fh:
            for out in scorer.score.map(batches, order_outputs=False):
                for qid, logits in out:
                    if set(logits) != set(need[qid]):
                        raise RuntimeError(f'{qid}: scored {len(logits)}/{len(need[qid])} chunks')
                    fh.write(json.dumps({'query_id': qid, 'logits': logits}) + '\n')
                fh.flush()
                n += len(out)
                if n % 240 < PER_CALL:
                    print(json.dumps({'done_questions': n, 'of': len(todo)}), flush=True)
    got = {r['query_id']: r['logits'] for r in records(dest)}
    assert all(set(need[q]) <= set(got[q]) for q in need), 'missing scores'
    write(OUT / f'ce_logits_{a.split}_summary.json', {'questions': len(need), 'pairs': sum(map(len, need.values())),
          'sha256': sha(dest), 'pools_sha256': pool_sha, 'chunks_text_sha256': sha(text_path)})
    print('ok', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
