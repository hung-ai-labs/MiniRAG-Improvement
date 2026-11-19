"""Deferred five-arm report, registered secondary metrics and provenance. No APIs."""
from __future__ import annotations

from collections import Counter
import json
import random
import re
import statistics

from reproduce.mhr_ce import run as ce
from reproduce.mhr_f import run as f
from reproduce.mhr_f import study as base
from reproduce.mhr_set import offline
from reproduce.mhr_set import run as r
from reproduce.mhr_st.common import DATA, INDEX, JUDGE, OUT as ST_OUT, SEEDS, read, records, sha

ARMS = ('C0', 'ST', 'F', 'CE', 'SET')
NAMES = {'C0': 'C0', 'ST': 'C0+ST', 'F': 'C0+ST+F', 'CE': 'C0+ST+F+CE', 'SET': 'C0+ST+F+CE+set'}


def verdict_path(arm, seed):
    return {'C0': ST_OUT / f'test/C0_s{seed}', 'ST': ST_OUT / f'test/ST_s{seed}',
            'F': f.folder(seed), 'CE': ce.folder(seed), 'SET': r.folder(seed)}[arm] / 'judge_gemini35'


def read_verdicts(arm, seed, wanted):
    path = verdict_path(arm, seed)
    summary = read(path / 'summary.json')
    _, rows = r.read_csv(path / 'per_question.csv')
    out = {v['query_id']: v for v in rows}
    if summary['status'] != 'completed' or summary['n_questions'] != len(wanted):
        raise ValueError(f'{arm} seed {seed}: incomplete judge summary')
    if len(rows) != len(wanted) or set(out) != set(wanted):
        raise ValueError('Incomplete, duplicate or unpaired verdicts')
    for v in rows:
        labels = [v[f'pass{p}'] for p in (1, 2, 3)]
        if any(x not in ('accurate', 'error', 'neither', 'unparsed') for x in labels):
            raise ValueError('Invalid pass label')
        label, count = Counter(labels).most_common(1)[0]
        if v['majority'] != (label if count >= 2 else 'no_majority'):
            raise ValueError('Invalid majority label')
    return out, summary


def binary_pair(reference, changed):
    if len(reference) != len(changed) or not reference:
        raise ValueError('Unpaired or empty binary metric')
    up = sum(b and not a for a, b in zip(reference, changed))
    down = sum(a and not b for a, b in zip(reference, changed))
    return {'n': len(reference), 'up': up, 'down': down, 'p_exact': base.exact_p(up, down),
            'delta_acc_pp': 100 * (up - down) / len(reference)}


def normalize(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', text.lower())).strip()


def entity_metric(seed, questions):
    # Exact rule from review/mhr_ce_20261116/secondary_checks/check.py.
    entity = [i for i, q in enumerate(questions) if q['Type'] != 'null' and normalize(q['Gold Answer']) not in ('yes', 'no')]
    outcomes = {}
    for arm, path in (('F', f.folder(seed)), ('CE', ce.folder(seed)), ('SET', r.folder(seed))):
        _, answers = r.read_csv(path / 'answers.csv')
        if len(answers) != len(questions) or any(q['Question'] != a['Question'] or q['Gold Answer'] != a['Gold Answer']
                                                for q, a in zip(questions, answers)):
            raise ValueError(f'{arm}: entity metric answer alignment differs')
        outcomes[arm] = [f" {normalize(questions[i]['Gold Answer'])} " in f" {normalize(answers[i]['minirag'])} "
                         for i in entity]
    return {'n': len(entity), 'rule': 'Normalized gold word-bounded substring; type != null and gold not yes/no',
            'accuracy_percent': {a: 100 * sum(v) / len(v) for a, v in outcomes.items()},
            'SET_vs_CE': binary_pair(outcomes['CE'], outcomes['SET']),
            'SET_vs_F': binary_pair(outcomes['F'], outcomes['SET']),
            'limitation': 'Yes/no questions remain judge-only'}


def kappa(rows):
    categories = ('accurate', 'error', 'neither')
    votes = [[row[f'pass{p}'] for p in (1, 2, 3)] for row in rows]
    excluded = sum('unparsed' in v for v in votes)  # DEVIATIONS.md 2026-11-18
    votes = [v for v in votes if 'unparsed' not in v]
    if not votes:
        return {'n': 0, 'fleiss_kappa': None, 'unanimous': 0}
    if any(label not in categories for row in votes for label in row):
        raise ValueError('Fleiss kappa requires three valid labels')
    n = len(votes)
    observed = sum((sum(row.count(c) ** 2 for c in categories) - 3) / 6 for row in votes) / n
    proportions = [sum(row.count(c) for row in votes) / (3 * n) for c in categories]
    expected = sum(p * p for p in proportions)
    return {'n': n, 'fleiss_kappa': (observed - expected) / (1 - expected) if expected < 1 else None,
            'unanimous': sum(len(set(row)) == 1 for row in votes), 'excluded_unparsed_rows': excluded,
            'observed_agreement': observed, 'chance_agreement': expected}


def drift_report(questions):
    path = r.OUT / 'secondary_checks/ce_drift'
    summary, sample = read(path / 'run/summary.json'), read(path / 'sample.json')
    indices = random.Random(13).sample(range(2156), 200)
    if sample['indices'] != indices or sample['ce_answers_sha256'] != sha(ce.folder(101) / 'answers.csv'):
        raise ValueError('CE drift sample changed')
    if summary['status'] != 'completed' or summary['n_questions'] != 200:
        raise ValueError('CE drift rejudging incomplete')
    chosen = [questions[i]['query_id'] for i in indices]
    old, _ = read_verdicts('CE', 101, {q['query_id'] for q in questions})
    latest = {}
    for v in records(path / 'run/judge.jsonl'):
        key = v['query_id'], v['pass']
        if v['status'] not in ('ok', 'unparsed') or key in latest:
            raise ValueError('Invalid CE drift journal')
        latest[key] = v
    if set(latest) != {(q, p) for q in chosen for p in (1, 2, 3)}:
        raise ValueError('CE drift verdict coverage differs')
    majority_equal = pass_equal = 0
    counts = Counter()
    for q in chosen:
        labels = [latest[q, p]['label'] for p in (1, 2, 3)]
        label, n = Counter(labels).most_common(1)[0]
        majority = label if n >= 2 else 'no_majority'
        majority_equal += majority == old[q]['majority']
        pass_equal += sum(labels[p-1] == old[q][f'pass{p}'] for p in (1, 2, 3))
        counts[old[q]['majority'] + ' -> ' + majority] += 1
    return {'n': 200, 'sample_seed': 13, 'majority_agreement_count': majority_equal,
            'majority_agreement': majority_equal / 200, 'pass_agreement_count': pass_equal,
            'pass_agreement': pass_equal / 600, 'transitions': dict(counts),
            'started_at': sample['started_at'], 'completed_at': summary['completed_at'],
            'journal_sha256': sha(path / 'run/judge.jsonl')}


def report():
    from reproduce.mhr_st.report import paired, quality
    if any((r.OUT / name).exists() for name in ('RESULTS.json', 'BANG_SO_SANH.txt')):
        raise FileExistsError('ADD-ONLY: report output already exists')
    r.frozen()
    questions = r.test_rows()
    queries = {v['query_id']: v['Question'] for v in questions}
    if len(questions) != 2156 or len(queries) != 2156:
        raise ValueError('Registered test2156 required')
    gold = {v['query_id']: v for v in records(DATA / 'processed/gold_chunks.jsonl')}
    chunks, _, _ = base.corpus()
    result = {'judge': JUDGE, 'n': len(queries), 'seeds': {}, 'model_F_sha256': sha(f.MODEL),
              'model_CE_sha256': sha(ce.MODEL), 'set_params_sha256': sha(r.PARAMS),
              'sd_definition': 'sample SD between generation seeds',
              'primary_comparison': 'SET_vs_CE', 'secondary_comparison': 'SET_vs_F',
              'registered_rule': 'ACC rises in all three seeds and exact McNemar SET_vs_CE p < 0.05 in each',
              'by_type_exploratory_no_multiplicity_correction': True,
              'sources_tokens_definition': 'Sum of frozen tiktoken chunk-content counts; all test questions',
              'secondary_checks': {'judge_free_entity': {}, 'fleiss_kappa_SET': {}},
              'input_sha256': {}}
    by_seed = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        arms, summaries = {}, {}
        for arm in ARMS:
            arms[arm], summaries[arm] = read_verdicts(arm, seed, queries)
            if any(v['Type'] != next(q['Type'] for q in questions if q['query_id'] == k) for k, v in arms[arm].items()):
                raise ValueError('Question types differ from frozen inputs')
            by_seed[arm].append(quality(list(arms[arm].values())))
            p = verdict_path(arm, seed) / 'per_question.csv'
            result['input_sha256'][str(p.relative_to(r.ROOT))] = sha(p)
        sr = {'quality': {a: quality(list(v.values())) for a, v in arms.items()},
              'pass_mean_sd_judge': {a: summaries[a]['groups']['all']['pass_mean_sd'] for a in ARMS},
              'verdict_origin': summaries['SET']['verdict_origin'],
              'SET_vs_CE': paired(arms['CE'], arms['SET']), 'SET_vs_F': paired(arms['F'], arms['SET']),
              'SET_vs_C0': paired(arms['C0'], arms['SET']), 'by_type': {}}
        for typ in ('comparison', 'inference', 'temporal', 'null'):
            sub = {a: {q: v for q, v in rows.items() if v['Type'] == typ} for a, rows in arms.items()}
            sr['by_type'][typ] = {'quality': {a: quality(list(v.values())) for a, v in sub.items()},
                                  'SET_vs_CE': paired(sub['CE'], sub['SET'])}
        c0path = ST_OUT / f'archive_c0/qa/s{seed}/C0_s{seed}/context.jsonl'
        old = {v['query']: v for v in records(c0path)}
        if any(q not in old for q in queries.values()):
            raise ValueError('Incomplete C0 context archive')
        selected = {'C0': {qid: old[q]['chunk_ids'] for qid, q in queries.items()}}
        context_paths = {'ST': ST_OUT / f'test/ST_s{seed}/contexts.jsonl',
                         'F': f.folder(seed) / 'contexts.jsonl', 'CE': ce.folder(seed) / 'contexts.jsonl',
                         'SET': r.folder(seed) / 'contexts.jsonl'}
        for arm, path in context_paths.items():
            rows = records(path)
            selected[arm] = {v['query_id']: v['chunk_ids'] for v in rows}
            if len(rows) != len(queries) or set(selected[arm]) != set(queries):
                raise ValueError(f'Incomplete {arm} selections')
        for path in [c0path, *context_paths.values()]:
            result['input_sha256'][str(path.relative_to(r.ROOT))] = sha(path)
        recall, tokens = {}, {}
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
                           'macro': statistics.mean(macro), 'full': complete / n, 'full_evidence': complete}
            tokens[arm] = r.token_stats(sum(chunks[c]['tokens'] for c in ids) for ids in chosen.values())
        sr['evidence_recall'], sr['sources_tokens'] = recall, tokens
        result['seeds'][str(seed)] = sr
        result['secondary_checks']['judge_free_entity'][str(seed)] = entity_metric(seed, questions)
        result['secondary_checks']['fleiss_kappa_SET'][str(seed)] = {
            'all': kappa(list(arms['SET'].values())),
            'newly_judged_only': kappa([v for v in arms['SET'].values() if v['verdict_origin'] == 'judged_SET'])}
    result['secondary_checks']['CE_seed101_judge_drift'] = drift_report(questions)
    result['secondary_checks']['known_limitation'] = 'Yes/no answers are scored only by the LLM judge'
    result['generation_mean_sd'] = {
        a: {k: {'mean': statistics.mean(v[k] for v in values), 'sd': statistics.stdev(v[k] for v in values)}
            for k in values[0]} for a, values in by_seed.items()}
    for ref in ('CE', 'F', 'C0'):
        key = f'SET_vs_{ref}'
        result[key + '_all_three_seeds_positive_and_significant'] = all(
            sr[key]['up'] > sr[key]['down'] and sr[key]['p_exact'] < .05 for sr in result['seeds'].values())
    manifest = read(INDEX / 'index_manifest.json')
    result['index_unchanged'] = all(sha(INDEX / p) == h for p, h in manifest['file_sha256'].items())
    if not result['index_unchanged']:
        raise RuntimeError('Index changed during experiment')
    text = ['MultiHopRAG: C0 vs +ST vs +F vs +CE vs +set', f'Qwen2.5-3B | Judge: {JUDGE}',
            '2,156 questions; 3 generation seeds; 3 judge passes per answer.',
            'ACC/ERR/Neither use majority labels; +/- is sample SD between generation seeds.', '',
            '| Configuration | ACC mean +/- SD | ERR | Neither | No majority | Evidence recall (micro) | Full evidence | Mean Sources tokens |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        quality_values = result['generation_mean_sd'][arm]
        evidence = statistics.mean(sr['evidence_recall'][arm]['micro'] for sr in result['seeds'].values())
        full = statistics.mean(sr['evidence_recall'][arm]['full'] for sr in result['seeds'].values())
        tokens = statistics.mean(sr['sources_tokens'][arm]['mean'] for sr in result['seeds'].values())
        text.append(f"| {NAMES[arm]} | {quality_values['accurate']['mean']:.2f} +/- {quality_values['accurate']['sd']:.2f} | "
                    f"{quality_values['error']['mean']:.2f} | {quality_values['neither']['mean']:.2f} | "
                    f"{quality_values['no_majority']['mean']:.2f} | {evidence:.3f} | {100 * full:.1f}% | {tokens:.1f} |")
    for key, title in (('SET_vs_CE', 'SET vs CE (primary, registered)'), ('SET_vs_F', 'SET vs F (secondary)'),
                       ('SET_vs_C0', 'SET vs C0')):
        text += ['', title, '| Seed | Delta ACC (pp) | Wrong -> correct | Correct -> wrong | Exact McNemar p |',
                 '|---|---:|---:|---:|---:|']
        for seed, sr in result['seeds'].items():
            value = sr[key]
            text.append(f"| {seed} | {value['delta_acc_pp']:+.2f} | {value['up']} | {value['down']} | {value['p_exact']:.6g} |")
    decision = result['SET_vs_CE_all_three_seeds_positive_and_significant']
    text += ['', 'SET over CE: positive and significant in all three seeds.' if decision else
             'SET over CE: improvement is not confirmed under the registered three-seed rule.',
             'SET vs F is secondary because CE was not confirmed over F.',
             'By-type SET vs CE is exploratory without multiplicity correction.',
             'Entity-answer substring metric, CE judge drift, Fleiss kappa, ERR/Neither and origins: RESULTS.json.',
             'Yes/no answers are scored only by the LLM judge.', 'Frozen index checksums verified unchanged.']
    offline.write(r.OUT / 'RESULTS.json', result)
    offline.new_text(r.OUT / 'BANG_SO_SANH.txt', '\n'.join(text) + '\n')
    print('\n'.join(text))
    return result
