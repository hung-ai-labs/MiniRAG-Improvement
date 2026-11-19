"""Registered judge-drift check (PROTOCOL.md): re-judge 200 F seed-101 test answers,
random.Random(13).sample(range(2156), 200), on the CE judging date; compare majorities with the
stored F verdicts. Run from MiniRAG-Improvement with MINIRAG_DOTENV set. Resumable."""
import collections, csv, json, os, random, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(os.environ['MINIRAG_DOTENV'], override=False)
from reproduce.mhr_ce import run
R = Path('review'); OUT = R / 'mhr_ce_20261116/secondary_checks/judge_drift'; OUT.mkdir(parents=True, exist_ok=True)
q = list(csv.DictReader(open(R / 'mhr_st_20261112/inputs/test.csv', encoding='utf-8')))
F = list(csv.DictReader(open(R / 'mhr_f_20261114/test/F_s101/answers.csv', encoding='utf-8')))
assert len(q) == len(F) == 2156 and all(a['Question'] == b['Question'] for a, b in zip(q, F))
idx = sorted(random.Random(13).sample(range(len(q)), 200))
if not (OUT / 'questions.csv').exists():
    with open(OUT / 'questions.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(q[0])); w.writeheader(); w.writerows(q[i] for i in idx)
    with open(OUT / 'answers.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(F[0])); w.writeheader(); w.writerows(F[i] for i in idx)
run.run_judge(OUT / 'questions.csv', OUT / 'answers.csv', OUT / 'run', rounds=10)
lab = collections.defaultdict(dict)
for l in open(OUT / 'run/judge.jsonl', encoding='utf-8'):
    r = json.loads(l)
    if r['status'] == 'ok': lab[r['query_id']][r['pass']] = r['label']
def maj(ls):
    c = collections.Counter(ls).most_common(1)[0]; return c[0] if c[1] >= 2 else 'no_majority'
old = {r['query_id']: r['majority'] for r in csv.DictReader(open(R / 'mhr_f_20261114/test/F_s101/judge_gemini35/per_question.csv', encoding='utf-8'))}
ids = [q[i]['query_id'] for i in idx]
new = {k: maj(list(lab[k].values())) for k in ids}
agree = sum(old[k] == new[k] for k in ids)
flips = collections.Counter(f'{old[k]}->{new[k]}' for k in ids if old[k] != new[k])
res = {'sample': '200 F seed-101 test answers, random.Random(13).sample(range(2156), 200)',
       'majority_agreement': f'{agree}/200 ({agree/2:.1f}%)',
       'old_counts': dict(collections.Counter(old[k] for k in ids)), 'new_counts': dict(collections.Counter(new.values())),
       'flips_old_to_new': dict(flips),
       'net_accurate_change': sum(new[k] == 'accurate' for k in ids) - sum(old[k] == 'accurate' for k in ids)}
(OUT / 'RESULT.json').write_text(json.dumps(res, indent=2), encoding='utf-8'); print(json.dumps(res, indent=2))
