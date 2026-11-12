"""Extra Table 1 rows with the FINE-TUNED scorer (train_scorer.py; Kaggle cross-fit, published 11/11/2026 23:04).

The fine-tuned arm `CE+F`: MiniLM-L6 cross-encoder, all encoder weights trained on the dev200
Evidence labels, plus a linear head on the same 18 standardized features; long chunks are read
as overlapping 512-token windows (stride 128), max over windows; Platt-calibrated.

Rows (same 435 non-dev questions, same contexts / answers / verdicts for the shared rows):
  BASELINE   C0 + source-time + A1@4000
  CE0F       zero-shot CE + F, sort by p -> A1          (Table 1 row 4)
  FT         fine-tuned CE + F, sort by p -> A1
  CE0F_SET   zero-shot CE + F, budgeted set selection    (Table 1 row 5)
  FT_SET     fine-tuned CE + F, budgeted set selection
Nothing is tuned on non-dev: the all-fold checkpoint (models/all_folds/CE+F, trained on dev200
only) and (beta, tau) chosen on its cross-fitted dev scores (scores/CE+F.jsonl).
Generation and judging reuse pipeline.py (body-hash Qwen cache; verdict store copied first),
so only FT and FT_SET cost new Qwen / DeepSeek calls.

    .venv/bin/python -m reproduce.evidence_set.finetuned all --rounds 3
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from reproduce.graph_ablation.study import ROOT, sha, write, lines
from reproduce.evidence_set import pipeline as P
from reproduce.evidence_set.slm import SLM, BASE

FT_DIR = BASE / 'models/all_folds/CE+F'          # written by train_scorer.py (Qwen-index dev200)
FT_DEV_SCORES = BASE / 'scores/CE+F.jsonl'        # its cross-fitted dev200 p
ARMS = ('BASELINE', 'CE0F', 'FT', 'CE0F_SET', 'FT_SET')
LABEL = {'BASELINE': 'C0 + source-time (baseline)', 'CE0F': 'scorer, zero-shot CE + F (sort by p)',
         'FT': 'scorer, FINE-TUNED CE + F (sort by p)', 'CE0F_SET': 'zero-shot CE + F + set selection',
         'FT_SET': 'FINE-TUNED CE + F + set selection'}


def out_dir(split):
    return P.SPLIT[split]['dir'] / 'ladder_ft'


def ft_scores(split, base_rows):
    """p for every (question, pool chunk) from the all-fold fine-tuned checkpoint; cached."""
    cache = P.SPLIT[split]['dir'] / 'ft_scores.json'
    ckpt_sha = sha((FT_DIR / 'model.pt').read_bytes())
    if cache.exists():
        got = json.loads(cache.read_text())
        assert got['checkpoint_sha256'] == ckpt_sha
        return got['p']
    import torch
    from reproduce.evidence_set import train_scorer as T
    from reproduce.evidence_set.data import corpus
    from reproduce.evidence_set.features import featurize, pool_of
    chunks, _, _ = corpus()
    ckpt = torch.load(FT_DIR / 'model.pt', map_location='cpu', weights_only=False)
    assert ckpt['arm'] == 'CE+F' and ckpt['base_revision'] == T.REVISION
    feats = ckpt['features']
    rows = [{'id': i, 'question': base_rows[i]['question'],
             'candidates': [{**c, 'label': 0} for c in featurize(base_rows[i], pool_of(base_rows[i], P.POOL))]}
            for i in sorted(base_rows)]
    snapshot = Path.home() / '.cache/huggingface/hub' / ('models--' + T.MODEL.replace('/', '--')) / 'snapshots' / T.REVISION
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = T.TextScorer(snapshot, len(feats))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    tok = T.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    pairs = T.Pairs(rows, chunks, tok, feats, T.GRID['max_length'], T.GRID['document_stride'])
    raw = T.predict(model, rows, pairs, np.asarray(ckpt['mean']), np.asarray(ckpt['scale']), device)
    platt = json.loads((FT_DIR / 'calibration.json').read_text())['platt']
    prob = T.calibrate(raw, platt)
    p = {str(r['id']): {c['chunk_id']: float(v) for c, v in zip(r['candidates'], prob[r['id']])} for r in rows}
    cache.write_text(json.dumps({'checkpoint_sha256': ckpt_sha, 'device': device, 'p': p}))
    return p


def build(split):
    assert split == 'nondev', 'dev200 fine-tuned rows would need the cross-fitted scores only; not wired'
    from reproduce.evidence_set.data import corpus, full, gold
    from reproduce.evidence_set.features import pool_of
    from reproduce.evidence_set.nondev import dev_beta_tau
    from reproduce.evidence_set.qa import with_sources
    from reproduce.evidence_set.select import facets, select, rerank
    chunks, _, _ = corpus()
    base, bctx = P.load(split, 'baseline')
    old = {(r['id'], r['arm']): r for r in P.contexts(split)}
    p = ft_scores(split, base)
    beta, tau = dev_beta_tau(None, FT_DEV_SCORES)[0]
    out = []
    for i in sorted(base):
        sets = gold(base[i]['question'])[1]
        pool = [c for c in pool_of(base[i], P.POOL)]
        pi = p[str(i)]
        sel = {'FT': rerank(pool, pi), 'FT_SET': select(pool, pi, facets(base[i], pool), beta=beta, tau=tau)[0]}
        for arm in ARMS:
            if arm in sel:
                ids = sel[arm]
                assert sum(chunks[c]['tokens'] for c in ids) <= 4000
                ctx = with_sources(bctx[i]['context'], [chunks[c]['content'] for c in ids])
                r = {**old[i, 'BASELINE'], 'arm': arm, 'context': ctx, 'context_sha256': sha(ctx), 'chunk_ids': ids,
                     'full_evidence': full(ids, sets) if sets else None}
            else:
                r = old[i, arm]
            out.append(r)
    d = out_dir(split)
    d.mkdir(parents=True, exist_ok=True)
    (d / 'contexts.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out))
    by = lambda a: [x for x in out if x['arm'] == a]
    summary = {'split': split, 'slm': SLM, 'questions': len(base), 'arms': list(ARMS), 'pool': P.POOL,
               'scores': 'all-fold models fit on dev200 only (FT: fine-tuned checkpoint, deployment refit)',
               'ft_checkpoint_sha256': sha((FT_DIR / 'model.pt').read_bytes()),
               'ft_beta_tau_from_dev': [beta, tau],
               'labelled': sum(x['full_evidence'] is not None for x in by('BASELINE')),
               'full_evidence': {a: sum(bool(x['full_evidence']) for x in by(a)) for a in ARMS},
               'identical_to_baseline': {a: sum(x['context_sha256'] == b['context_sha256']
                                                for x, b in zip(by(a), by('BASELINE'))) for a in ARMS},
               'contexts_sha256': sha((d / 'contexts.jsonl').read_bytes()),
               'builder_sha256': sha(Path(__file__).read_bytes()), 'api_calls': 0}
    write(d / 'build.json', summary)
    # reuse every verdict already paid for (keyed by question id + answer hash)
    src = P.ladder(split) / 'judgments_deepseek.jsonl'
    if src.exists() and not (d / 'judgments_deepseek.jsonl').exists():
        shutil.copy(src, d / 'judgments_deepseek.jsonl')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('build', 'generate', 'judge', 'report', 'all'))
    ap.add_argument('--split', default='nondev')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--judge-workers', type=int, default=32)
    a = ap.parse_args()
    P.ARMS, P.LABEL = ARMS, LABEL
    ladder_of_ladder = P.ladder
    P.ladder = out_dir
    steps = ('build', 'generate', 'judge', 'report') if a.step == 'all' else (a.step,)
    for s in steps:
        print(f'== {s} ({a.split}, fine-tuned rows)', flush=True)
        if s == 'build':
            P.ladder = ladder_of_ladder  # read the main ladder's contexts
            P.ARMS = ('C0', 'BASELINE', 'F', 'CE0F', 'CE0F_SET')
            build(a.split)
            P.ARMS, P.ladder = ARMS, out_dir
        elif s == 'generate':
            P.generate(a.split, a.rounds, a.workers)
        elif s == 'judge':
            P.judge(a.split, a.judge_workers)
        else:
            P.report(a.split)
