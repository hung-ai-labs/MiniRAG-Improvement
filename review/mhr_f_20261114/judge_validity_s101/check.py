import csv, re, sys, collections, math
from scipy.stats import binomtest
B='mhr_st_20261112/test/'; F='mhr_f_20261114/test/F_s101/'
q=list(csv.DictReader(open('mhr_st_20261112/inputs/test.csv',encoding='utf-8')))
def ans(p): return [r['minirag'] for r in csv.DictReader(open(p,encoding='utf-8'))]
A={'C0':ans(B+'C0_s101/answers.csv'),'ST':ans(B+'ST_s101/answers.csv'),'F':ans(F+'answers.csv')}
V={k:{r['query_id']:r for r in csv.DictReader(open(p,encoding='utf-8'))} for k,p in
   {'C0':B+'C0_s101/judge_gemini35/per_question.csv','ST':B+'ST_s101/judge_gemini35/per_question.csv','F':F+'judge_gemini35/per_question.csv'}.items()}
n=lambda s: re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',s.lower())).strip()
def hit(gold,pred,typ):
    g,p=n(gold),n(pred)
    if typ=='null': return bool(re.search(r'insufficient|not enough|cannot|unable|no information|do not have|does not (provide|contain|mention)',p))
    if g in('yes','no'):
        w=re.findall(r'\b(yes|no)\b',p); return bool(w) and w[0]==g
    return f' {g} ' in f' {p} '
M={k:[hit(r['Gold Answer'],a,r['Type']) for r,a in zip(q,A[k])] for k in A}
print('rule-based contains-match ACC (all 2156):')
for k in M: print(f'  {k}: {100*sum(M[k])/len(q):.2f}')
nn=[i for i,r in enumerate(q) if r['Type']!='null']
print(f'  excluding null ({len(nn)}):',{k:round(100*sum(M[k][i] for i in nn)/len(nn),2) for k in M})
def mc(a,b):
    up=sum(x and not y for x,y in zip(a,b)); dn=sum(y and not x for x,y in zip(a,b))
    return up,dn,binomtest(up,up+dn,.5).pvalue
for x,y in(('F','ST'),('F','C0'),('ST','C0')): print(f'  McNemar {x} vs {y}: +{mc(M[x],M[y])[0]} / -{mc(M[x],M[y])[1]}  p={mc(M[x],M[y])[2]:.2g}')
# judge vs rule agreement
for k in V:
    J=[V[k][r['query_id']]['majority']=='accurate' if r['query_id'] in V[k] else None for r in q]
    pr=[(j,m) for j,m in zip(J,M[k]) if j is not None]
    print(f'  {k}: judge-vs-rule agreement {sum(j==m for j,m in pr)}/{len(pr)} = {100*sum(j==m for j,m in pr)/len(pr):.1f}%')
# Fleiss kappa over 3 passes
for k in V:
    rows=[[r['pass1'],r['pass2'],r['pass3']] for r in V[k].values()]
    rows=[x for x in rows if all(l in('accurate','error','neither') for l in x)]
    N=len(rows); cats=('accurate','error','neither')
    P=sum((sum(x.count(c)**2 for c in cats)-3)/6 for x in rows)/N
    pj=[sum(x.count(c) for x in rows)/(3*N) for c in cats]; Pe=sum(p*p for p in pj)
    print(f'  {k}: Fleiss kappa 3 passes = {(P-Pe)/(1-Pe):.3f} (N={N}); unanimous {sum(len(set(x))==1 for x in rows)}/{N}')
import random
print('\n=== disagreement breakdown (F) by type: judge_acc&rule_miss / judge_not&rule_hit')
c=collections.Counter()
for r,a,m in zip(q,A['F'],M['F']):
    j=V['F'][r['query_id']]['majority']=='accurate'
    if j!=m: c[(r['Type'],'J+R-' if j else 'J-R+')]+=1
print(sorted(c.items()))
print('\n=== F gains by judge over ST (ST not accurate, F accurate): rule says')
g=[i for i,r in enumerate(q) if V['F'][r['query_id']]['majority']=='accurate' and V['ST'][r['query_id']]['majority']!='accurate']
print(len(g), 'rule F hit', sum(M['F'][i] for i in g), 'rule ST hit', sum(M['ST'][i] for i in g))
random.seed(1)
for i in random.sample([i for i in g if not M['F'][i]],6):
    print('---',q[i]['Type'],'| GOLD:',q[i]['Gold Answer'],'| ST judge:',V['ST'][q[i]['query_id']]['majority'])
    print('Q:',q[i]['Question'][:200]); print('ST:',A['ST'][i][:300].replace('\n',' ')); print('F :',A['F'][i][:300].replace('\n',' '))
print('\n=== subsets')
yn=[i for i,r in enumerate(q) if n(r['Gold Answer']) in('yes','no')]
ent=[i for i,r in enumerate(q) if r['Type']!='null' and i not in set(yn)]
print('yes/no',len(yn),'entity',len(ent),'null',sum(r['Type']=='null' for r in q))
for name,S in(('entity',ent),('yes/no',yn)):
    for k in A:
        J=sum(V[k][q[i]['query_id']]['majority']=='accurate' for i in S)
        print(f'  {name} {k}: rule {100*sum(M[k][i] for i in S)/len(S):.2f}  judge {100*J/len(S):.2f}')
    u,d,p=mc([M['F'][i] for i in S],[M['ST'][i] for i in S]); print(f'  {name} rule F vs ST +{u}/-{d} p={p:.2g}')
    Jf=[V['F'][q[i]['query_id']]['majority']=='accurate' for i in S]; Js=[V['ST'][q[i]['query_id']]['majority']=='accurate' for i in S]
    u,d,p=mc(Jf,Js); print(f'  {name} judge F vs ST +{u}/-{d} p={p:.2g}')
    print(f'  {name} judge-vs-rule agreement F', sum((V['F'][q[i]['query_id']]['majority']=='accurate')==M['F'][i] for i in S),'/',len(S))
