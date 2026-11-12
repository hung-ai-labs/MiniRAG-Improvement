"""One-pass blinded judging, exact-context cache reuse and staged selection."""
from collections import Counter
import json
import os
from pathlib import Path
import runpy

from .study import ROOT,OUT,FROZEN,COMPARISONS,read,lines,write,append,sha,csvfile,metrics,guard
from .api import API,StopAPI,DailyQuota,judge_template,MODEL

LABELS=('accurate','error','neither')
ARMS=('G0','B','P','TS','TE','D','TD','TDV')


def historical():
 # Validate model/prompt/settings before any reuse. No cherry-picking passes.
 dirs=[ROOT/'logs/p0_qa/20261109_dev200',ROOT/'logs/p0_qa/20261110_c123_extension']
 template=judge_template();prompts=runpy.run_path(str(ROOT/'minirag/prompt.py'))['PROMPTS']
 answers={};judgments={}
 for d in dirs:
  plan=read(d/'manifest.json')
  if plan.get('generator')!=MODEL or plan.get('judge')!=MODEL:continue
  if 'judge_prompt_sha256' in plan and plan['judge_prompt_sha256']!=sha(template):raise ValueError('Historical judge prompt mismatch')
  if 'system_template_sha256' in plan and plan['system_template_sha256']!=sha(prompts['rag_response']):raise ValueError('Historical generator template mismatch')
  for r in lines(d/'answers.jsonl'):
   if r.get('generation_round')!=1:continue
   key=(r['question'],r['context_sha256']);value=r['prediction']
   if key in answers and answers[key]!=value:raise ValueError('Conflicting generation cache; no picking answer')
   answers[key]=value
  for r in lines(d/'judgments.jsonl'):
   if r.get('judge_pass')!=1 or r.get('generation_round')!=1 or r['verdict'] not in LABELS:continue
   key=(r['question'],r['prediction_sha256'])
   if key in judgments and judgments[key]!=r['verdict']:raise ValueError('Conflicting first-pass judge cache')
   judgments[key]=r['verdict']
 return answers,judgments,prompts['rag_response'],template


def rates(rows):
 n=len(rows)
 return {k:100*sum(r['verdict']==label for r in rows)/n if n else None for k,label in [('ACC','accurate'),('ERR','error'),('Neither','neither')]}|{'n':n}


def gate(subset,evidence,arms,*,allow_evidence_only=False):
 selected=[];base=rates([r for r in subset if r['arm']=='G0'])
 by={(r['id'],r['arm']):r['verdict'] for r in subset}
 for arm in arms:
  if arm=='G0':continue
  rs=[r for r in subset if r['arm']==arm];m=rates(rs)
  evidence_ok=evidence[arm]['kept_gold_pairs']>=evidence['G0']['kept_gold_pairs']
  loss=sum(by[(r['id'],'G0')]=='accurate' and r['verdict']!='accurate' for r in rs)
  promising_retrieval=(allow_evidence_only and evidence[arm]['kept_gold_pairs']>evidence['G0']['kept_gold_pairs'] and any(r['context_changed'] for r in rs))
  if (m['n']==base['n'] and m['ACC']>=base['ACC'] and m['ERR']<=base['ERR'] and
      (m['ACC']>base['ACC'] or m['ERR']<base['ERR'] or promising_retrieval) and loss<=max(1,int(.2*m['n'])) and evidence_ok):selected.append(arm)
 return selected


def run_phase(spec,traces,ids,arms,phase,api_holder):
 if not read(OUT/'regression_tests.json')['passed']:raise ValueError('Regression gate not passed')
 old_answers,old_judges,system_template,jprompt=historical()
 answers={**old_answers,**{(r['question'],r['context_sha256']):r['prediction'] for r in lines(OUT/'answers.jsonl')}}
 judgments={**old_judges,**{(r['question'],r['prediction_sha256']):r['verdict'] for r in lines(OUT/'judgments.jsonl') if r['judge']==MODEL}}
 records=[r for r in traces if r['id'] in ids and r['arm'] in arms]
 saved={(r['id'],r['arm'],r['phase']):r for r in lines(OUT/'evaluated.jsonl')}
 def call(*args,**kwargs):
  if api_holder[0] is None:api_holder[0]=API(allow_daily_retry=os.environ.get('MINIRAG_RESUME_CONFIRMED_HTTP')=='1')
  return api_holder[0].call(*args,**kwargs)
 # Materialize valid cached results first; an unavailable new generation must
 # not erase already-complete controls or fabricate the missing comparisons.
 for r in records:
  sk=(r['id'],r['arm'],phase);ak=(r['question'],r['context_sha256'])
  if sk in saved or ak not in answers:continue
  pk=(r['question'],sha(answers[ak]))
  if pk not in judgments:continue
  cached={'id':r['id'],'question':r['question'],'arm':r['arm'],'phase':phase,'judge':MODEL,
   'verdict':judgments[pk],'context_sha256':r['context_sha256'],'prediction_sha256':pk[1],
   'answer_reused':ak in old_answers,'judge_reused':True,
   'context_changed':r['context_sha256']!=r['baseline_context_sha256']}
  append(OUT/'evaluated.jsonl',cached);saved[sk]=cached
 # Finish all generations before judging. A fallback judge cannot rescue missing
 # generations from the fixed model; never switch generation model mid-study.
 for r in records:
  key=(r['question'],r['context_sha256'])
  if key not in answers:
   system=system_template.format(context_data=r['context'],response_type='Multiple Paragraphs')
   pred=call(r['question'],system=system,role='generation')
   answers[key]=pred
   append(OUT/'answers.jsonl',{'question':r['question'],'context_sha256':r['context_sha256'],'prediction':pred,'model':MODEL,'generation_round':1,'phase':phase})
 for r in records:
  key=(r['id'],r['arm'],phase)
  if key in saved:continue
  pred=answers[(r['question'],r['context_sha256'])];pk=(r['question'],sha(pred))
  reused=pk in judgments
  if not reused:
   gold=spec['questions'][r['id']]['Gold Answer']
   label=call(jprompt.format(question=r['question'],gold=gold,pred=pred),role='judge').strip().lower()
   judgments[pk]=label
   append(OUT/'judgments.jsonl',{'question':r['question'],'prediction_sha256':pk[1],'verdict':label,'judge':MODEL,'judge_pass':1,'phase':phase})
  record={'id':r['id'],'question':r['question'],'arm':r['arm'],'phase':phase,'judge':MODEL,
   'verdict':judgments[pk],'context_sha256':r['context_sha256'],'prediction_sha256':pk[1],
   'answer_reused':(r['question'],r['context_sha256']) in old_answers,'judge_reused':reused,
   'context_changed':r['context_sha256']!=r['baseline_context_sha256']}
  append(OUT/'evaluated.jsonl',record);saved[key]=record
 print('PHASE',phase,'complete',len(ids),'questions',','.join(arms),flush=True)
 guard(spec)
 return [saved[(r['id'],r['arm'],phase)] for r in records]


def report(spec,state,best=None,error=None):
 evaluated=lines(OUT/'evaluated.jsonl');api=lines(OUT/'api_calls.jsonl')
 # Never pool different phases or judges into one denominator. Prefer a complete
 # full200 result, then validation20, then canary6 for each displayed arm.
 order=['full200','validation20','canary6'];rows=[]
 for arm in ARMS:
  chosen=[];phase=None
  for p in order:
   sub=[r for r in evaluated if r['arm']==arm and r['phase']==p]
   required=200 if p=='full200' else 20 if p=='validation20' else 6
   if len(sub)==required:chosen=sub;phase=p;break
  m=rates(chosen);base=[r for r in evaluated if r['arm']=='G0' and r['phase']==phase]
  b=rates(base)
  verdict='CONTROL' if arm=='G0' else 'INCOMPLETE'
  if chosen and arm!='G0':
   if not any(r['context_changed'] for r in chosen):verdict='NO_RETRIEVAL_EFFECT'
   elif m['ACC']>b['ACC'] and m['ERR']<=b['ERR'] or m['ERR']<b['ERR'] and m['ACC']>=b['ACC']:verdict='PROMISING_EXPLORATORY'
   elif m['ACC']<b['ACC'] or m['ERR']>b['ERR']:verdict='REGRESSION'
   else:verdict='NEUTRAL'
  rows.append({'arm':arm,**m,'phase':phase,'judge':MODEL,'Delta_ACC':m['ACC']-b['ACC'] if chosen and base else None,'Delta_ERR':m['ERR']-b['ERR'] if chosen and base else None,'Verdict':verdict})
 csvfile(OUT/'ablation_results.csv',evaluated,['id','question','arm','phase','judge','verdict','context_sha256','prediction_sha256','answer_reused','judge_reused','context_changed'])
 csvfile(OUT/'combination_results.csv',[*rows,*([dict(next(r for r in rows if r['arm']==best),arm='BEST',underlying_arm=best)] if best else [])],['arm','n','ACC','ERR','Neither','Delta_ACC','Delta_ERR','phase','judge','Verdict','underlying_arm'])
 paid_estimate=sum(r.get('conservative_peak_cost_usd',0) for r in api)
 usage={'new_requests':len(api),'successes':sum(bool(r.get('success')) for r in api),'input_tokens':sum((r.get('usage') or {}).get('prompt_tokens',0) for r in api),'output_tokens':sum((r.get('usage') or {}).get('completion_tokens',0) for r in api),
  'estimated_paid_usd':paid_estimate,'provider_invoiced_cost':None,'cost_basis':'Peak-price upper estimate for DeepSeek, configured free tier for Gemini; invoice not returned','deepseek_requests':sum(r.get('provider')=='deepseek' for r in api),'local_new_extraction_requests':0,
  'old_answer_reuse_count':sum(r['answer_reused'] for r in evaluated),'judge_reuse_count':sum(r['judge_reused'] for r in evaluated)}
 write(OUT/'api_usage.json',usage)
 status={'state':state,'best':best,'error':error,'evaluated_question_arm_pairs':len(evaluated),'unique_questions':len({r['id'] for r in evaluated}),'usage':usage,'rows':rows,'judge_passes':1,'configuration_gain_confirmed':False}
 write(OUT/'status.json',status)
 text=['# Original MiniRAG graph: eight-arm study','',f'Status: **{state}**. BEST: **{best or "NOT_SELECTED"}**.',
  '','All arms keep original nx.Graph/topology/weights and C3-P0; original production index untouched.',
  f'Generator/judge: {MODEL}, pass 1 only. Dev has been inspected in prior research; single generation and judge are exploratory.',
  '', '| Arm | n | Phase | ACC | ERR | Neither | Delta ACC | Delta ERR | Verdict |','|---|---:|---|---:|---:|---:|---:|---:|---|']
 for r in rows:
  fmt=lambda x:'—' if x is None else f'{x:.2f}'
  text.append(f"| {r['arm']} | {r['n']} | {r['phase'] or '—'} | {fmt(r['ACC'])} | {fmt(r['ERR'])} | {fmt(r['Neither'])} | {fmt(r['Delta_ACC'])} | {fmt(r['Delta_ERR'])} | {r['Verdict']} |")
 text+=['', 'Rows with different Phase/n are not directly comparable. Each delta uses G0 on that exact phase.',
  'FULL = TDV. BEST is selected only after the frozen stage gates. Identical contexts reuse identical predictions and first-pass verdicts.',
  f'New API usage: {json.dumps(usage)}.', '',
  'B: cached formatting recovery cannot be safely transplanted into an already merged graph with missing raw history; no fake graph changes.',
  'D: no externally verified direction; UNKNOWN falls back. TDV: no semantic trust reviews; no deletion of all UNKNOWN evidence.',
  'No evidence precision or answer-supporting path accuracy is claimed from incomplete gold chunk labels. Exact source validity is not predicate correctness.']
 if error:text+=['', 'Stop reason: '+error]
 (OUT/'SUMMARY.md').write_text('\n'.join(text)+'\n')
 return status


def evaluate(spec,traces):
 api=[None];best=None
 try:
  canary=run_phase(spec,traces,spec['canary_indices'],ARMS,'canary6',api)
  em=metrics(spec,traces,spec['canary_indices']);write(OUT/'canary_evidence.json',em)
  promising=gate(canary,em,ARMS,allow_evidence_only=True)
  if not promising:return report(spec,'NO_CANARY_CANDIDATE',error='No arm passed predefined ACC/ERR and evidence gates; do not spend on 200-question QA')
  # Collapse duplicate context policies before choosing/expanding; keep simpler arm.
  signatures={a:tuple(r['context_sha256'] for r in traces if r['arm']==a) for a in ARMS}
  unique=[]
  for a in promising:
   if not any(signatures[a]==signatures[b] for b in unique):unique.append(a)
  arms=['G0',*unique]
  validation=run_phase(spec,traces,spec['validation_indices'],arms,'validation20',api)
  ve=metrics(spec,traces,spec['validation_indices']);write(OUT/'validation_evidence.json',ve)
  finalists=gate(validation,ve,arms)
  if not finalists:return report(spec,'NO_VALIDATION_CANDIDATE',error='Canary improvement did not pass fixed validation20 gate; stop before 200')
  best=max(finalists,key=lambda a:(rates([r for r in validation if r['arm']==a])['ACC'],-rates([r for r in validation if r['arm']==a])['ERR'],-ARMS.index(a)))
  run_phase(spec,traces,list(range(200)),['G0',best],'full200',api)
  return report(spec,'GRAPH_200_COMPLETE',best)
 except (StopAPI,DailyQuota) as exc:
  # A fallback cannot fill missing generations from the fixed model. Assess
  # feasibility BEFORE spending the <=8-request DeepSeek budget.
  write(OUT/'fallback_decision.json',{'provider':MODEL,'status':'STOPPED_WITH_CHECKPOINT','reason':'Fixed-provider comparison incomplete; no mixed-judge or unpaired baseline-only spend','cap':spec.get('deepseek_budget',spec['limits']),'error_type':type(exc).__name__})
  return report(spec,'INCOMPLETE_API',best,str(exc))
