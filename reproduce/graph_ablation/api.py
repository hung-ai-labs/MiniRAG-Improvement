"""Single-key public Gemini transport. No key rotation, paid generation or hidden retry."""
import ast
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .study import ROOT,OUT,append,read,sha,write,lines

MODEL='gemini-flash-lite-latest'
class StopAPI(RuntimeError):pass
class DailyQuota(StopAPI):pass


def keys():
 # Selectively parse keys; do not load/log unrelated malformed .env lines.
 env=dict(os.environ)
 for line in (ROOT/'.env').read_text().splitlines():
  if '=' not in line or line.lstrip().startswith('#'):continue
  k,v=line.split('=',1);k=k.strip();v=v.strip().strip('\"\'')
  if (k.startswith('GEMINI_API_KEY') or k in ('GOOGLE_API_KEY','DEEPSEEK_API_KEY')) and k not in env:env[k]=v
 return env


def judge_template():
 tree=ast.parse((ROOT/'reproduce/Step_2_evaluate.py').read_text())
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JUDGE_PROMPT' for t in n.targets):return ast.literal_eval(n.value)
 raise ValueError('Judge prompt not found')


def quota_is_daily(body):
 if isinstance(body,dict):
  if 'PerDay' in str(body.get('quotaId','')) or 'per_day' in str(body.get('quotaMetric','')).lower():return True
  return any(quota_is_daily(v) for v in body.values())
 if isinstance(body,list):return any(map(quota_is_daily,body))
 return False


class API:
 def __init__(self,*,allow_daily_retry=False):
  env=keys();pairs=sorted((k,v) for k,v in env.items() if k.startswith('GEMINI_API_KEY') and v.startswith('AIza'))
  if not pairs and env.get('GOOGLE_API_KEY','').startswith('AIza'):pairs=[('GOOGLE_API_KEY',env['GOOGLE_API_KEY'])]
  if not pairs:raise StopAPI('No configured Gemini key')
  self.key_name,self.key=pairs[0];self.next_at=0;self.allow_daily_retry=allow_daily_retry
  self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  self.resolved={}
  for r in lines(OUT/'api_calls.jsonl'):
   if r.get('success') and r['provider']=='gemini':self.resolved.setdefault(r['role'],r['resolved_model'])
  self.tokenizer=__import__('tiktoken').get_encoding('o200k_base')

 def call(self,prompt,system=None,role='judge'):
  # Match the historical protocol, max_tokens and provider-default sampling.
  max_tokens=1024 if role=='generation' else 16
  messages=([{'role':'system','content':system}] if system else [])+[{'role':'user','content':prompt}]
  body={'model':MODEL,'messages':messages,'max_tokens':max_tokens}
  task_sha=sha(json.dumps(body,sort_keys=True,ensure_ascii=False))
  cache_root=OUT/'api_cache'/task_sha
  if (cache_root/'response.json').exists():return read(cache_root/'response.json')['content']
  attempts=([cache_root] if (cache_root/'intent.json').exists() else [])+sorted(cache_root.glob('retry_*'))
  if attempts:
   last=attempts[-1]
   if not (last/'error_body.json').exists():
    raise StopAPI('Ambiguous prior request without an HTTP failure receipt; do not resend automatically')
   history=[r for r in lines(OUT/'api_calls.jsonl') if r.get('request_sha256')==task_sha]
   if not history or (history[-1].get('daily_quota') and not self.allow_daily_retry):
    raise DailyQuota('Confirmed daily quota checkpoint; use explicit resume only after quota resets, never rotate keys')
   if history[-1].get('status') not in (429,500,502,503,504):
    raise StopAPI('Non-transient HTTP failure retained; no automatic retry')
   if len(attempts)>=3:raise StopAPI('Three same-key transient HTTP attempts failed; checkpoints retained')
  cache=cache_root if not attempts else cache_root/f'retry_{len(attempts)+1}'
  if len(lines(OUT/'api_calls.jsonl'))>=600:raise StopAPI('Global request cap reached')
  wait=max(0,self.next_at-time.monotonic())
  if wait:time.sleep(wait)
  cache.mkdir(parents=True,exist_ok=True)
  write(cache/'intent.json',{'role':role,'provider':'gemini','requested_model':MODEL,'request_sha256':task_sha,'key_variable':self.key_name,'no_key_rotation':True})
  input_bound=len(self.tokenizer.encode(''.join(m['content'] for m in messages)))+64
  self.next_at=time.monotonic()+60*(input_bound+max_tokens)/3000
  url='https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
  req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
  start=time.perf_counter()
  try:
   with self.opener.open(req,timeout=180) as response:raw=response.read()
  except urllib.error.HTTPError as exc:
   raw=exc.read().replace(self.key.encode(),b'[REDACTED]');(cache/'error_body.json').write_bytes(raw)
   try:detail=json.loads(raw)
   except ValueError:detail={}
   daily=quota_is_daily(detail)
   record={'provider':'gemini','role':role,'success':False,'status':exc.code,'daily_quota':daily,'request_sha256':task_sha,'key_variable':self.key_name,'latency_s':time.perf_counter()-start}
   append(OUT/'api_calls.jsonl',record)
   if daily:raise DailyQuota('Confirmed Gemini per-day quota; stopped without key rotation') from None
   if exc.code in (429,500,502,503,504) and len(attempts)<2:
    self.next_at=max(self.next_at,time.monotonic()+60*(len(attempts)+1))
    return self.call(prompt,system=system,role=role)
   raise StopAPI('Gemini HTTP '+str(exc.code)+'; saved raw error, no fabricated verdict') from None
  except Exception as exc:
   append(OUT/'api_calls.jsonl',{'provider':'gemini','role':role,'success':False,'error_type':type(exc).__name__,'request_sha256':task_sha})
   raise StopAPI('Gemini transport failure: '+type(exc).__name__) from None
  raw=raw.replace(self.key.encode(),b'[REDACTED]')
  (cache/'raw.json').write_bytes(raw)
  try:
   result=json.loads(raw);choice=result['choices'][0];content=choice['message']['content'];resolved=result.get('model') or MODEL
  except (ValueError,KeyError,IndexError,TypeError):
   append(OUT/'api_calls.jsonl',{'provider':'gemini','role':role,'success':False,'http_success':True,
      'request_sha256':task_sha,'error_type':'ProviderSchemaError','latency_s':time.perf_counter()-start})
   raise StopAPI('Provider schema invalid; raw retained, request counted') from None
  invalid=None
  if self.resolved.get(role,resolved)!=resolved:invalid='Resolved provider model changed within run'
  self.resolved[role]=resolved
  if not content or (role=='judge' and choice.get('finish_reason')!='stop'):invalid='Empty/truncated judge response, saved before parse'
  if role=='judge' and isinstance(content,str) and content.strip().lower() not in ('accurate','error','neither'):invalid='Invalid judge label; never map failure to neither'
  if invalid:
   append(OUT/'api_calls.jsonl',{'provider':'gemini','role':role,'success':False,'http_success':True,
      'request_sha256':task_sha,'error_type':'UnusableResponse','reason':invalid,'usage':result.get('usage'),
      'resolved_model':resolved,'latency_s':time.perf_counter()-start})
   raise StopAPI(invalid)
  record={'provider':'gemini','role':role,'success':True,'request_sha256':task_sha,'resolved_model':resolved,'usage':result.get('usage'),'finish_reason':choice.get('finish_reason'),
          'latency_s':time.perf_counter()-start,'estimated_paid_cost_usd':0,'cost_basis':'Configured public free-tier endpoint; provider invoice not returned'}
  append(OUT/'api_calls.jsonl',record);write(cache/'response.json',{'content':content,'receipt':record})
  if cache!=cache_root:write(cache_root/'response.json',{'content':content,'receipt':record,'successful_attempt':len(attempts)+1})
  print('API',role,'OK',flush=True)
  return content
