"""Recompute final paired quality and Sources recall from completed journals."""
import csv
from collections import Counter
import statistics

from .common import ARCHIVE, DATA, INDEX, JUDGE, OUT, SEEDS, read, records, sha, write


def quality(rows):
    counts = Counter(r['majority'] for r in rows)
    return {k: 100 * counts[k] / len(rows) for k in ('accurate', 'error', 'neither', 'no_majority')}


def paired(a, b):
    from scipy.stats import binomtest
    import numpy as np
    up = sum(a[k]['majority'] != 'accurate' and b[k]['majority'] == 'accurate' for k in a)
    down = sum(a[k]['majority'] == 'accurate' and b[k]['majority'] != 'accurate' for k in a)
    differences = np.array([int(b[k]['majority'] == 'accurate') - int(a[k]['majority'] == 'accurate') for k in a])
    rng = np.random.default_rng(13)
    bootstrap = np.concatenate([rng.choice(differences, size=(250, len(a))).mean(axis=1) for _ in range(40)])
    return {'up': up, 'down': down, 'delta_acc_pp': 100 * (up - down) / len(a),
            'p_exact': binomtest(up, up + down).pvalue if up + down else 1.0,
            'paired_bootstrap_ci95_pp': (100 * np.quantile(bootstrap, [.025, .975])).tolist()}


def report():
    result = {'judge': JUDGE, 'n': 2156, 'seeds': {}, 'sd_definition': 'sample SD between generation seeds'}
    gold = {r['query_id']: r for r in records(DATA / 'processed/gold_chunks.jsonl')}
    by_seed = {'C0': [], 'ST': []}
    for seed in SEEDS:
        arms = {}
        judge_sd = {}
        for arm in ('C0', 'ST'):
            path = OUT / f'test/{arm}_s{seed}/judge_gemini35'
            summary = read(path / 'summary.json')
            if summary['status'] != 'completed' or summary['n_questions'] != 2156:
                raise RuntimeError('Judge incomplete')
            with (path / 'per_question.csv').open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            if len(rows) != 2156 or any(r['majority'] not in ('accurate', 'error', 'neither', 'no_majority') for r in rows):
                raise ValueError('Incomplete or invalid per-question labels')
            arms[arm] = {r['query_id']: r for r in rows}
            by_seed[arm].append(quality(rows))
            judge_sd[arm] = summary['groups']['all']['pass_mean_sd']
        if set(arms['C0']) != set(arms['ST']):
            raise ValueError('Unpaired query IDs')
        seed_result = {'quality': {a: quality(list(v.values())) for a, v in arms.items()},
                       'pass_mean_sd_judge': judge_sd,
                       'ST_vs_C0': paired(arms['C0'], arms['ST']), 'by_type': {}}
        for typ in ('comparison', 'inference', 'temporal', 'null'):
            pair = {a: {k: r for k, r in v.items() if r['Type'] == typ} for a, v in arms.items()}
            seed_result['by_type'][typ] = {'quality': {a: quality(list(v.values())) for a, v in pair.items()},
                                            'paired': paired(pair['C0'], pair['ST'])}
        # Facts can occur in either chunk of an overlap; one matching chunk is
        # enough. The frozen mapper has no facts spanning chunk boundaries.
        old_logs = {r['query']: r for r in records(ARCHIVE / f'qa/s{seed}/C0_s{seed}/context.jsonl')}
        with (OUT / 'inputs/test.csv').open(encoding='utf-8', newline='') as f:
            queries = {r['query_id']: r['Question'] for r in csv.DictReader(f)}
        selections = {'C0': {k: old_logs.get(queries[k], {}).get('chunk_ids', []) for k in queries},
                      'ST': {r['query_id']: r['chunk_ids'] for r in records(OUT / f'test/ST_s{seed}/contexts.jsonl')}}
        recall = {}
        for arm, selected in selections.items():
            hits = total = complete = n = 0
            macro = []
            for qid, chunks in selected.items():
                facts = gold[qid]['evidence']
                if not facts:
                    continue
                found = sum(bool(set(chunks) & set(e['chunk_ids'])) for e in facts)
                hits += found
                total += len(facts)
                complete += found == len(facts)
                n += 1
                macro.append(found / len(facts))
            recall[arm] = {'n': n, 'facts_hit': hits, 'gold_facts': total,
                           'micro': hits / total, 'macro': statistics.mean(macro), 'full': complete / n}
        seed_result['evidence_recall'] = recall
        result['seeds'][str(seed)] = seed_result
    result['generation_mean_sd'] = {arm: {metric: {
        'mean': statistics.mean(r[metric] for r in values), 'sd': statistics.stdev(r[metric] for r in values)}
        for metric in values[0]} for arm, values in by_seed.items()}
    result['all_three_seeds_positive_and_significant'] = all(
        s['ST_vs_C0']['up'] > s['ST_vs_C0']['down'] and s['ST_vs_C0']['p_exact'] < .05 for s in result['seeds'].values())
    manifest = read(INDEX / 'index_manifest.json')
    result['index_unchanged'] = all(sha(INDEX / p) == h for p, h in manifest['file_sha256'].items())
    if not result['index_unchanged']:
        raise RuntimeError('Index changed during experiment')
    write(OUT / 'RESULTS.json', result)
    table = ['MultiHopRAG: C0 vs +ST', f'Qwen2.5-3B | Judge: {JUDGE}',
             '2,156 questions; 3 generation seeds; 3 judge passes per answer.',
             'ACC/ERR/Neither below use majority labels; +/- is SD between generation seeds.', '',
             '| Configuration | ACC mean +/- SD | ERR | Neither | No majority |',
             '|---|---:|---:|---:|---:|']
    for arm in ('C0', 'ST'):
        m = result['generation_mean_sd'][arm]
        table.append(f"| {arm} | {m['accurate']['mean']:.2f} +/- {m['accurate']['sd']:.2f} | "
                     f"{m['error']['mean']:.2f} | {m['neither']['mean']:.2f} | {m['no_majority']['mean']:.2f} |")
    table += ['', '| Seed | Delta ACC (pp) | Wrong -> correct | Correct -> wrong | Exact McNemar p |',
              '|---|---:|---:|---:|---:|']
    for seed, value in result['seeds'].items():
        p = value['ST_vs_C0']
        table.append(f"| {seed} | {p['delta_acc_pp']:+.2f} | {p['up']} | {p['down']} | {p['p_exact']:.6g} |")
    decision = ('Positive and significant in all three seeds.' if result['all_three_seeds_positive_and_significant']
                else 'Improvement is not confirmed under the registered three-seed rule.')
    table += ['', decision, '', 'Judge variation per seed and subgroup results: RESULTS.json.',
              'Frozen index checksums verified unchanged.']
    (OUT / 'BANG_SO_SANH.txt').write_text('\n'.join(table) + '\n', encoding='utf-8')
    return result
