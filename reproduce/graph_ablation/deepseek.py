"""Independent DeepSeek fallback study. No Gemini verdict/generation pooling.

Default limits remain the user's 8 requests/24k input/16k output/$0.50.
Larger limits require an explicitly approved budget file before sending.
"""
import argparse
import fcntl
import json
from pathlib import Path
import runpy
import time
import urllib.error
import urllib.request

from .api import StopAPI,keys,judge_template
from .study import ROOT,OUT as GEMINI_OUT,read,write,lines,sha

OUT=GEMINI_OUT/'deepseek'
MODEL='deepseek-flash'
DEFAULT={'max_requests':8,'max_input_tokens':24000,'max_output_tokens':16000,'max_usd':0.50}
PRICE={'input_miss_per_million':0.30,'input_hit_per_million':0.006,'output_per_million':1.20}


def peak_cost(inp,out,hit=0):
 return (max(0,inp-hit)*PRICE['input_miss_per_million']+hit*PRICE['input_hit_per_million']+out*PRICE['output_per_million'])/1000000


def proposal():
 from .study import metrics
 traces=lines(GEMINI_OUT/'retrieval_traces.jsonl');spec=read(GEMINI_OUT/'question_manifest.json')
 system=runpy.run_path(str(ROOT/'minirag/prompt.py'))['PROMPTS']['rag_response']
 enc=__import__('tiktoken').get_encoding('o200k_base')
 chosen={(r['question'],r['context_sha256']):r for r in traces if r['arm'] in ('G0','TS')}
 n=len(chosen);input_proxy=sum(len(enc.encode(system.format(context_data=r['context'],response_type='Multiple Paragraphs')+r['question']))+64 for r in chosen.values())
 # Judge upper proxy includes a max-length generated answer and unchanged rubric.
 judge_proxy=sum(len(enc.encode(judge_template().format(question=r['question'],gold=spec['questions'][r['id']]['Gold Answer'],pred='')))+64+1024 for r in chosen.values())
 plan={'status':'AWAITING_BUDGET_APPROVAL','study':'Separate DeepSeek generation AND judging for all compared arms; no pooled Gemini metrics',
  'model':MODEL,'thinking':'disabled','questions':200,'judge_passes':1,'stages':['canary6 all8','validation20 promising','full200 only passing candidates'],
  'G0_TS_unique_generation_inputs':n,'G0_TS_max_individual_judgments':n,'G0_TS_requests_upper':2*n,
  'full200_input_tiktoken_proxy':input_proxy+judge_proxy,'full200_max_output_tokens':n*(1024+16),
  'peak_cost_proxy_max_output_usd':peak_cost(input_proxy+judge_proxy,n*(1024+16)),
  'offpeak_cost_proxy_max_output_usd':peak_cost(input_proxy+judge_proxy,n*(1024+16))/2,
  'proxy_is_not_provider_tokenizer_or_invoice':True,'prices':PRICE,
  'price_sources':['https://api-docs.deepseek.com/quick_start/pricing/','https://api-docs.deepseek.com/api/create-chat-completion/'],
  'default_authorized_caps':DEFAULT,'proposed_caps':{'max_requests':600,'max_input_tokens':2000000,'max_output_tokens':300000,'max_usd':1.00},
  'cost_policy':'Reserve each request at conservative UTF8-byte input bound and peak miss price; release unused reservation only after valid actual usage. Stop before any cap.',
  'new_requests_sent':0,'same_fixed_questions':True,'fixed_profile':'C3-P0','budget_requested_not_granted':True}
 OUT.mkdir(parents=True,exist_ok=True)
 if not (OUT/'proposed_plan.json').exists():write(OUT/'proposed_plan.json',plan)
 if (OUT/'approved_budget.json').exists() and read(OUT/'approved_budget.json').get('approved'):
  plan.update(status='APPROVED',budget_requested_not_granted=False,approved_caps=read(OUT/'approved_budget.json'))
  if (OUT/'budget_ledger.json').exists():plan['new_requests_sent']=read(OUT/'budget_ledger.json')['requests']
  write(OUT/'execution_plan.json',plan)
 return plan


class DeepSeekAPI:
 def __init__(self,*,allow_daily_retry=False,transport_retry_hashes=()):
  self.transport_retry_hashes=frozenset(transport_retry_hashes)
  budget_path=OUT/'approved_budget.json'
  self.budget=read(budget_path) if budget_path.exists() else {**DEFAULT,'approved':True,'scope':'Default user-attached cap only'}
  if not self.budget.get('approved'):raise StopAPI('Paid budget is not approved')
  self.key=keys().get('DEEPSEEK_API_KEY','')
  if not self.key:raise StopAPI('No configured DEEPSEEK_API_KEY')
  self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  self.resolved={r['role']:r['resolved_model'] for r in lines(OUT/'api_calls.jsonl') if r.get('success')}
  self.ledger=OUT/'budget_ledger.json'
  if not self.ledger.exists():write(self.ledger,{'requests':0,'input_tokens':0,'output_tokens':0,'cost_peak_estimate_usd':0.,'intents':[]})

 def call(self,prompt,system=None,role='judge'):
  max_tokens=1024 if role=='generation' else 16
  messages=([{'role':'system','content':system}] if system else [])+[{'role':'user','content':prompt}]
  body={'model':MODEL,'messages':messages,'max_tokens':max_tokens,'thinking':{'type':'disabled'}}
  request_sha=sha(json.dumps(body,ensure_ascii=False,sort_keys=True));cache_root=OUT/'api_cache'/request_sha
  if (cache_root/'response.json').exists():return read(cache_root/'response.json')['content']
  cache=cache_root;attempt=1
  if (cache_root/'intent.json').exists():
   # Retry ONLY a fully received, usage-accounted truncated judge. The exact
   # same model/prompt/settings are sent; ambiguous transport never qualifies.
   history=[r for r in lines(OUT/'api_calls.jsonl') if r.get('request_sha256')==request_sha]
   received=[x for x in read(self.ledger)['intents'] if x.get('request_sha256')==request_sha]
   safe_judge_retry=(role=='judge' and history and history[-1].get('success') is False
       and history[-1].get('finish_reason')=='length'
       and received and all(x.get('state')=='RECEIVED' for x in received))
   # Explicitly reviewed one-time reconnect after a network-route repair.
   # Keep the entire old PENDING reservation/cost: do not assume it was free.
   reviewed_reconnect=(request_sha in self.transport_retry_hashes and role=='generation'
       and len(received)==1 and received[0].get('state')=='PENDING'
       and history and history[-1].get('success') is False
       and history[-1].get('error_type')=='URLError')
   if not (safe_judge_retry or reviewed_reconnect):
    raise StopAPI('DeepSeek request already reserved without validated success; no unreviewed retry')
   attempt=len(received)+1
   if attempt>3:raise StopAPI('Three confirmed truncated judge attempts; no further automatic retry')
   cache=cache_root/('retry_'+str(attempt))
   if cache.exists():raise StopAPI('Ambiguous retry cache exists; never resend')
  # UTF8 bytes bound text tokens; 2048 additional tokens conservatively bound
  # message/model framing. Tool/vision/search are never sent.
  bound=sum(len(m['content'].encode()) for m in messages)+2048
  reserve=peak_cost(bound,max_tokens)
  lock=OUT/'budget.lock';lock.parent.mkdir(parents=True,exist_ok=True)
  with lock.open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);ledger=read(self.ledger)
   if ledger['requests']+1>self.budget['max_requests'] or ledger['input_tokens']+bound>self.budget['max_input_tokens'] or ledger['output_tokens']+max_tokens>self.budget['max_output_tokens'] or ledger['cost_peak_estimate_usd']+reserve>self.budget['max_usd']:
    raise StopAPI('DeepSeek budget preflight stopped before exceeding a cap')
   ledger['requests']+=1;ledger['input_tokens']+=bound;ledger['output_tokens']+=max_tokens;ledger['cost_peak_estimate_usd']+=reserve
   ledger['intents'].append({'request_sha256':request_sha,'attempt':attempt,'role':role,'input_bound':bound,'output_bound':max_tokens,'reserved_usd':reserve,'state':'PENDING'})
   write(self.ledger,ledger);cache.mkdir(parents=True,exist_ok=True)
   write(cache/'intent.json',ledger['intents'][-1])
  start=time.perf_counter();url='https://api.deepseek.com/chat/completions'
  req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
  from .study import append
  try:
   with self.opener.open(req,timeout=180) as response:raw=response.read()
  except urllib.error.HTTPError as exc:
   raw=exc.read().replace(self.key.encode(),b'[REDACTED]');(cache/'error_body.json').write_bytes(raw)
   append(OUT/'api_calls.jsonl',{'provider':'deepseek','role':role,'request_sha256':request_sha,'success':False,'status':exc.code,'reservation_retained':True})
   raise StopAPI('DeepSeek HTTP '+str(exc.code)+'; raw saved, reservation retained') from None
  except Exception as exc:
   append(OUT/'api_calls.jsonl',{'provider':'deepseek','role':role,'request_sha256':request_sha,'success':False,'error_type':type(exc).__name__,'reservation_retained':True})
   raise StopAPI('DeepSeek transport failure; reservation retained') from None
  raw=raw.replace(self.key.encode(),b'[REDACTED]');(cache/'raw.json').write_bytes(raw)
  try:
   result=json.loads(raw);choice=result['choices'][0];text=choice['message']['content'];usage=result['usage'];inp=usage['prompt_tokens'];out=usage['completion_tokens'];hit=usage.get('prompt_cache_hit_tokens',0)
   if not isinstance(inp,int) or not isinstance(out,int) or not 0<=inp<=bound or not 0<=out<=max_tokens or not 0<=hit<=inp:raise ValueError('Usage outside preflight bounds')
  except (ValueError,KeyError,IndexError,TypeError):
   append(OUT/'api_calls.jsonl',{'provider':'deepseek','role':role,'request_sha256':request_sha,'success':False,'error_type':'SchemaOrUsageError','reservation_retained':True})
   raise StopAPI('DeepSeek raw/usage invalid; no budget release') from None
  model=result.get('model') or MODEL;cost=peak_cost(inp,out,hit)
  with lock.open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);ledger=read(self.ledger)
   ledger['input_tokens']-=bound-inp;ledger['output_tokens']-=max_tokens-out;ledger['cost_peak_estimate_usd']-=reserve-cost
   for item in ledger['intents']:
    if item['request_sha256']==request_sha and item.get('attempt',1)==attempt:item.update(state='RECEIVED',actual_input_tokens=inp,actual_output_tokens=out,cost_peak_estimate_usd=cost)
   write(self.ledger,ledger)
  valid=isinstance(text,str) and bool(text.strip()) and (role!='judge' or text.strip().lower() in ('accurate','error','neither') and choice.get('finish_reason')=='stop')
  valid=valid and self.resolved.get(role,model)==model
  receipt={'provider':'deepseek','role':role,'request_sha256':request_sha,'attempt':attempt,'success':bool(valid),'resolved_model':model,'usage':usage,'finish_reason':choice.get('finish_reason'),'latency_s':time.perf_counter()-start,'conservative_peak_cost_usd':cost,'invoice_cost':None}
  append(OUT/'api_calls.jsonl',receipt)
  if not valid:raise StopAPI('DeepSeek invalid label/response/model; never convert failure to neither')
  self.resolved[role]=model;write(cache/'response.json',{'content':text,'receipt':receipt})
  if cache!=cache_root:write(cache_root/'response.json',{'content':text,'receipt':receipt,'successful_attempt':attempt})
  print('DEEPSEEK',role,'OK',flush=True)
  return text


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-api',action='store_true');args=p.parse_args();plan=proposal()
 if not args.run_api:print(json.dumps(plan,ensure_ascii=False,indent=2));return
 budget=read(OUT/'approved_budget.json') if (OUT/'approved_budget.json').exists() else DEFAULT
 # Valid fixed canary needs 9 unique generations and <=9 judgments. Do not
 # burn an 8-request budget on a knowingly unfinishable fixed comparison.
 if budget['max_requests']<18:raise SystemExit('Default 8-request budget cannot complete frozen six-question comparison (up to18 calls). No DeepSeek request sent; see proposed_plan.json.')
 from . import evaluate
 from .study import freeze
 spec=freeze();spec={**spec,'generator':MODEL,'judge':MODEL,'scope':'Independent DeepSeek fallback; Gemini results not pooled','deepseek_budget':budget}
 write(OUT/'question_manifest.json',spec)
 for n in ('regression_tests.json','native_validation.json','offline_validation.json'):
  write(OUT/n,read(GEMINI_OUT/n))
 traces=lines(GEMINI_OUT/'retrieval_traces.jsonl')
 evaluate.OUT=OUT;evaluate.MODEL=MODEL;evaluate.API=DeepSeekAPI
 status=evaluate.evaluate(spec,traces)
 print(json.dumps(status,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
