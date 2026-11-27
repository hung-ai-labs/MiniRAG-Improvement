"""Full LiHua-World table (635 questions) = dev200 (outer-CV) + nondev435 (held out), one SLM.

Both splits must have the same ladder rows, the same generation rounds and the same judge.
The dev rows use cross-fitted scores (5 folds by conversation, beta/tau nested per fold);
the nondev rows use the all-fold models with dev-chosen beta/tau. Nothing is re-run here:
this only reads answers.jsonl + judgments_<judge>.jsonl of both ladders.

ACC/ERR = mean over rounds x passes on the pooled 635 questions; ± = sd between rounds.
McNemar on the per-round majority verdict, pooled over both splits.

    MINIRAG_EVSET_SLM=qwen .venv/bin/python -m reproduce.evidence_set.merge_full
"""
import json
import statistics
from collections import defaultdict

from reproduce.graph_ablation.study import lines, write
from reproduce.evidence_set.deepseek_judge import PASSES
from reproduce.evidence_set.pipeline import (ARMS, LABEL, LABELS, JUDGE, JUDGE_MODEL, SPLIT, contexts, exact_p,
                                             ladder, majority)
from reproduce.evidence_set.slm import SLM, SPEC, OUT


def load(split):
    d = ladder(split)
    meta = {(split, r['id'], r['arm']): r for r in contexts(split)}
    verdict = defaultdict(dict)
    for j in lines(d / f'judgments_{JUDGE}.jsonl'):
        verdict[j['id'], j['prediction_sha256']][j['judge_pass']] = j['verdict']
    ans = defaultdict(dict)
    for a in lines(d / 'answers.jsonl'):
        v = verdict.get((a['id'], a['prediction_sha256']), {})
        if all(p in v for p in PASSES):
            ans[a['round']][split, a['id'], a['arm']] = [v[p] for p in PASSES]
    build = json.loads((d / 'build.json').read_text())
    return meta, ans, build


def main():
    meta, ans, builds = {}, defaultdict(dict), {}
    for split in SPLIT:
        m, a, b = load(split)
        meta.update(m)
        builds[split] = b
        for k, v in a.items():
            ans[k].update(v)
    ids = sorted({(s, i) for s, i, _ in meta})
    qs = {meta[s, i, 'BASELINE']['question'] for s, i in ids}
    assert len(qs) == len(ids), 'a question appears in both splits'
    rounds = sorted(k for k in ans if all((s, i, a) in ans[k] for s, i in ids for a in ARMS))
    assert rounds, 'no round complete on both splits'
    groups = {'all': ids, **{t: [x for x in ids if meta[(*x, 'BASELINE')]['type'] == t] for t in ('Single', 'Multi', 'Null')}}
    maj = {k: {key: majority(v) for key, v in ans[k].items()} for k in rounds}
    res = {'slm': SLM, 'questions': len(ids), 'per_split': {s: sum(x[0] == s for x in ids) for s in SPLIT},
           'rounds': rounds, 'judge': JUDGE_MODEL, 'judge_passes': list(PASSES),
           'scores': {s: builds[s]['scores'] for s in SPLIT}, 'table': {}}
    for g, s in groups.items():
        res['table'][g] = {}
        for j, arm in enumerate(ARMS):
            cell = {l: [[100 * sum(ans[k][(*x, arm)][p] == l for x in s) / len(s) for p in range(len(PASSES))]
                        for k in rounds] for l in LABELS}
            row = {}
            for l, m in cell.items():
                per_round = [statistics.mean(x) for x in m]
                row[l] = {'mean': round(statistics.mean(per_round), 2), 'per_round': [round(x, 2) for x in per_round],
                          'sd_rounds': round(statistics.stdev(per_round), 2) if len(per_round) > 1 else None}
            for ref in ('BASELINE', ARMS[j - 1] if j else None):
                if ref is None or ref == arm:
                    continue
                row[f'vs_{ref}'] = []
                for k in rounds:
                    m = {}
                    for l in ('accurate', 'error'):
                        up = sum(maj[k][(*x, ref)] != l and maj[k][(*x, arm)] == l for x in s)
                        dn = sum(maj[k][(*x, ref)] == l and maj[k][(*x, arm)] != l for x in s)
                        m[l] = {'became': up, 'ceased': dn, 'p_exact': exact_p(up, dn)}
                    row[f'vs_{ref}'].append(m)
            res['table'][g][arm] = row
    fe = {a: sum(builds[s]['full_evidence'][a] for s in SPLIT) for a in ARMS}
    lab = sum(builds[s]['labelled'] for s in SPLIT)
    out = OUT / 'qa_full635'
    out.mkdir(exist_ok=True)
    write(out / f'results_{JUDGE}.json', res)
    sd = lambda x: f" ± {x['sd_rounds']:.2f}" if x['sd_rounds'] is not None else ''
    cmp = lambda r, key, l: '; '.join(f"{m[l]['became']}/{m[l]['ceased']}, p={m[l]['p_exact']:.2g}"
                                      for m in r[key]) if key in r else '—'
    md = [f"# Table 1 — {SPEC['name']}, full LiHua-World ({len(ids)} questions)", '',
          f"dev200 (outer-CV: {builds['dev200']['scores']}) + nondev435 (held out: {builds['nondev']['scores']}). "
          f"{len(rounds)} generation round(s) × {len(PASSES)} {JUDGE_MODEL} judge passes. "
          "ACC/ERR = mean over rounds × passes, ± = sd between generation rounds. McNemar on the per-round "
          "majority verdict, pooled over both splits, one entry per round (became / ceased, exact p).", '',
          '**The main result stays the 435 held-out questions.** The method was designed by looking at dev200; '
          'outer-CV removes leakage from training and tuning, not from design choices.', '',
          '| # | Configuration | Full evidence | ACC | ERR | Neither | ACC vs baseline | ACC vs row above | ERR vs row above |',
          '|---|---|---:|---:|---:|---:|---|---|---|']
    for j, arm in enumerate(ARMS):
        r = res['table']['all'][arm]
        above = 'vs_' + ARMS[j - 1] if j else None
        md.append(f"| {j + 1} | {LABEL[arm]} | {fe[arm]}/{lab} | {r['accurate']['mean']:.2f}{sd(r['accurate'])} | "
                  f"{r['error']['mean']:.2f}{sd(r['error'])} | {r['neither']['mean']:.2f} | "
                  f"{cmp(r, 'vs_BASELINE', 'accurate')} | {cmp(r, above, 'accurate') if above else '—'} | "
                  f"{cmp(r, above, 'error') if above else '—'} |")
    md += ['', '## ACC by question type', '',
           '| Configuration | ' + ' | '.join(f'{g} ({len(s)})' for g, s in groups.items() if g != 'all') + ' |',
           '|---|' + '---:|' * 3]
    for arm in ARMS:
        md.append(f'| {LABEL[arm]} | ' + ' | '.join(f"{res['table'][g][arm]['accurate']['mean']:.2f}"
                                                   for g in groups if g != 'all') + ' |')
    (out / f'REPORT{"" if JUDGE == "deepseek" else "_" + JUDGE}.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
