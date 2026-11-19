"""Registered secondary checks for row 4 (PROTOCOL.md, 'Secondary checks'): judge-free entity metric
(CE vs F McNemar per seed) and Fleiss kappa of CE verdicts. Run from review/. Read-only on inputs."""
import csv, re, collections
from scipy.stats import binomtest
q = list(csv.DictReader(open('mhr_st_20261112/inputs/test.csv', encoding='utf-8')))
n = lambda s: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', s.lower())).strip()
yn = {i for i, r in enumerate(q) if n(r['Gold Answer']) in ('yes', 'no')}
ent = [i for i, r in enumerate(q) if r['Type'] != 'null' and i not in yn]
def ans(p): return [r['minirag'] for r in csv.DictReader(open(p, encoding='utf-8'))]
def ver(p): return {r['query_id']: r for r in csv.DictReader(open(p, encoding='utf-8'))}
def hit(gold, pred): return f' {n(gold)} ' in f' {n(pred)} '
def mc(a, b):
    up = sum(x and not y for x, y in zip(a, b)); dn = sum(y and not x for x, y in zip(a, b))
    return up, dn, (binomtest(up, up + dn, .5).pvalue if up + dn else 1.0)
def kappa(rows):
    cats = ('accurate', 'error', 'neither')
    rows = [x for x in rows if all(l in cats for l in x)]; N = len(rows)
    P = sum((sum(x.count(c) ** 2 for c in cats) - 3) / 6 for x in rows) / N
    pj = [sum(x.count(c) for x in rows) / (3 * N) for c in cats]; Pe = sum(p * p for p in pj)
    return (P - Pe) / (1 - Pe), N, sum(len(set(x)) == 1 for x in rows)
print(f'entity-answer questions (gold not yes/no, type not null): {len(ent)} of {len(q)}; yes/no {len(yn)} (judge-only)')
for s in (101, 202, 303):
    AF = ans(f'mhr_f_20261114/test/F_s{s}/answers.csv'); AC = ans(f'mhr_ce_20261116/test/CE_s{s}/answers.csv')
    VF = ver(f'mhr_f_20261114/test/F_s{s}/judge_gemini35/per_question.csv'); VC = ver(f'mhr_ce_20261116/test/CE_s{s}/judge_gemini35/per_question.csv')
    MF = [hit(q[i]['Gold Answer'], AF[i]) for i in ent]; MC = [hit(q[i]['Gold Answer'], AC[i]) for i in ent]
    up, dn, p = mc(MC, MF)
    jf = [VF[q[i]['query_id']]['majority'] == 'accurate' for i in ent]; jc = [VC[q[i]['query_id']]['majority'] == 'accurate' for i in ent]
    ju, jd, jp = mc(jc, jf)
    print(f'\nseed {s}: rule F {100*sum(MF)/len(ent):.2f}  CE {100*sum(MC)/len(ent):.2f}  delta {100*(sum(MC)-sum(MF))/len(ent):+.2f} pp  CE vs F +{up}/-{dn} p={p:.3g}')
    print(f'          judge on same subset: F {100*sum(jf)/len(ent):.2f}  CE {100*sum(jc)/len(ent):.2f}  +{ju}/-{jd} p={jp:.3g}')
    allrows = [[r['pass1'], r['pass2'], r['pass3']] for r in VC.values()]
    own = [[r['pass1'], r['pass2'], r['pass3']] for r in VC.values() if r['verdict_origin'] == 'judged_CE']
    k1, n1, u1 = kappa(allrows); k2, n2, u2 = kappa(own)
    print(f'          Fleiss kappa CE verdicts: all {k1:.3f} (N={n1}, unanimous {u1}); newly judged only {k2:.3f} (N={n2}, unanimous {u2})')
    print('          verdict origins:', dict(collections.Counter(r['verdict_origin'] for r in VC.values())))
