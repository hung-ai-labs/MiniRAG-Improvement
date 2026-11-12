"""Fresh Qwen3B generation + one blinded DeepSeek judge pass on fixed LiHua20.

Both arms share frozen C0+source-time pool. No RECOMP/RRF in either arm.
Existing shared DeepSeek budget is never reset or increased. No historical
DeepSeek generations are pooled into this Qwen comparison.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from math import comb
from pathlib import Path
import random
import runpy
import socket
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from reproduce.graph_ablation.study import ROOT,read,lines,write,append,sha,csvfile
from reproduce.graph_ablation import deepseek
from reproduce.graph_ablation.api import judge_template,StopAPI
from . import lihua_transfer as transfer

ARMS=('BASELINE','MLP_NEIGHBORS')
MODEL='Qwen/Qwen2.5-3B-Instruct'
LABELS=('accurate','error','neither')


def exact_p(b,c):
    n=b+c
    return min(1.,2*sum(comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.


class Qwen:
    def __init__(self,out):
        config=Path.home()/'.config/minirag';base=(config/'slm_url').read_text().strip().rstrip('/')
        if not base.endswith('/v1'):base+='/v1'
        self.url=base+'/chat/completions';self.key=(config/'slm_key').read_text().strip();self.out=out
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self,r,template,response_type):
        body={'model':MODEL,'messages':[{'role':'system','content':template.format(context_data=r['context'],response_type=response_type)},
              {'role':'user','content':r['question']}],'max_tokens':1024,'seed':13+r['id']}
        digest=sha(json.dumps(body,sort_keys=True,ensure_ascii=False));cache=self.out/'qwen_cache'/digest
        if (cache/'response.json').exists():return read(cache/'response.json')
        if (cache/'intent.json').exists():raise StopAPI('Ambiguous prior Qwen intent; no automatic duplicate request')
        cache.mkdir(parents=True,exist_ok=True);write(cache/'intent.json',{'body_sha256':digest,'context_sha256':r['context_sha256'],
                'requested_model':MODEL,'endpoint_sha256':sha(self.url),'seed':13+r['id']})
        request=urllib.request.Request(self.url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
        started=time.perf_counter()
        try:
            with self.opener.open(request,timeout=180) as response:raw=response.read()
        except Exception as exc:
            write(cache/'failure.json',{'error_type':type(exc).__name__,'http_status':getattr(exc,'code',None),
                'outcome_unknown':True,'no_prediction_or_judge_fabricated':True})
            raise StopAPI('Qwen request failed: '+type(exc).__name__) from None
        raw=raw.replace(self.key.encode(),b'[REDACTED]');(cache/'raw.json').write_bytes(raw)
        value=json.loads(raw);choice=value['choices'][0];text=choice['message']['content']
        if not isinstance(text,str) or not text.strip() or text.strip()=='Error':raise StopAPI('Empty/invalid Qwen output')
        if choice['finish_reason'] not in ('stop','length'):raise StopAPI('Unsupported Qwen finish reason')
        if value.get('model') not in (MODEL,'minirag-slm'):raise StopAPI('Unexpected Qwen resolved model')
        result={'prediction':text,'prediction_sha256':sha(text),'requested_model':MODEL,'resolved_model':value.get('model'),
            'finish_reason':choice['finish_reason'],'usage':value.get('usage'),'latency_s':time.perf_counter()-started,
            'body_sha256':digest,'seed':13+r['id'],'provider_cost_not_reported':True}
        write(cache/'response.json',result);return result


def summary(out,spec,contexts,error=None):
    answers={(r['id'],r['arm']):r for r in lines(out/'answers.jsonl')}
    judged={(r['id'],r['arm']):r for r in lines(out/'judgments.jsonl')}
    assert len(answers)==len(lines(out/'answers.jsonl')) and len(judged)==len(lines(out/'judgments.jsonl'))
    ids=[i for i in spec['indices'] if all((i,a) in judged for a in ARMS)]
    rates={}
    for arm in ARMS:
        counts={label:sum(judged[i,arm]['verdict']==label for i in ids) for label in LABELS}
        rates[arm]={'n':len(ids),'counts':counts,**{key:100*counts[label]/len(ids) if ids else None
            for key,label in [('ACC','accurate'),('ERR','error'),('neither','neither')]}}
    paired={}
    for label in ('accurate','error'):
        up=sum(judged[i,ARMS[0]]['verdict']!=label and judged[i,ARMS[1]]['verdict']==label for i in ids)
        down=sum(judged[i,ARMS[0]]['verdict']==label and judged[i,ARMS[1]]['verdict']!=label for i in ids)
        paired[label]={'became_label':up,'ceased_label':down,'net':up-down,'p_exact':exact_p(up,down)}
    previous=0.
    for j,label in enumerate(sorted(paired,key=lambda l:paired[l]['p_exact'])):
        previous=max(previous,min(1.,paired[label]['p_exact']*(2-j)));paired[label]['p_holm_two']=previous
    start=read(out/'budget_start.json');ledger=read(deepseek.OUT/'budget_ledger.json')
    receipts=lines(deepseek.OUT/'api_calls.jsonl')[start['receipt_count']:]
    result={'status':'COMPLETE' if len(ids)==20 else 'INCOMPLETE','planned_questions':20,'matched_questions':len(ids),
        'generator':MODEL,'judge':deepseek.MODEL,'generation_rounds':1,'judge_passes':1,'rates':rates,'paired':paired,
        'generation_outputs':len(answers),'judgments':len(judged),'new_judge_receipts':len(receipts),
        'new_judge_peak_estimate_usd':sum(r.get('conservative_peak_cost_usd',0.) for r in receipts),
        'global_DeepSeek_budget':{k:ledger[k] for k in ('requests','input_tokens','output_tokens','cost_peak_estimate_usd')},
        'Qwen_serving_cost_not_provided_by_endpoint':True,'RRF':False,'RECOMP':False,'error':error,
        'scope':'Exploratory fixed20, one Qwen generation and one DeepSeek judge, not final paper benchmark'}
    write(out/'qa_report.json',result)
    meta={r['id']:r for r in contexts};table=[]
    for i in ids:
        row={'id':i,'question':meta[i]['question'],'type':meta[i]['type'],'gold':meta[i]['gold']}
        for a in ARMS:row[a+'_verdict']=judged[i,a]['verdict'];row[a+'_prediction']=answers[i,a]['prediction']
        table.append(row)
    csvfile(out/'paired_results.csv',table,['id','question','type','gold',*[f'{a}_{x}' for a in ARMS for x in ('verdict','prediction')]])
    md=['# LiHua20: model MiniLM-MLP vs baseline không RECOMP/RRF','',
        f"Trạng thái: {result['status']}, {len(ids)}/20 cặp. Qwen2.5-3B sinh mới cả hai arm, DeepSeek-flash chấm1lượt.",'',
        '| Arm | Đúng | Sai | Neither | ACC | ERR | Neither |','|---|---:|---:|---:|---:|---:|---:|']
    for a,r in rates.items():
        if not r['n']:continue
        c=r['counts'];md.append(f"| {a} | {c['accurate']} | {c['error']} | {c['neither']} | {r['ACC']:.1f}% | {r['ERR']:.1f}% | {r['neither']:.1f}% |")
    p=paired;md+=['',f"ACC: {p['accurate']['became_label']} lên/{p['accurate']['ceased_label']} xuống, p={p['accurate']['p_exact']:.6g}.",
        f"ERR: {p['error']['ceased_label']} lỗi được sửa/{p['error']['became_label']} lỗi mới, p={p['error']['p_exact']:.6g}.",'',
        'Cùng pool C0+source-time, P0/parser/Entities, prompt, Qwen settings và4000 token nội dung Sources. '
        'Model seed13 train FriendsQA, không train/tune trên LiHua. Chỉ thay phần chọn nội dung bằng MLP+ngữ cảnh liền kề. '
        'Nguồn context từ retrieval đã đóng băng; baseline20 khớp native lịch sử. '
        'Câu hỏi20 đã xem nhiều, không phải holdout sạch; một lượt sinh/chấm còn nhiễu. Không gộp output DeepSeek sinh trước đó. '
        'Đây là kiểm tra chuyển miền/pipeline tổng thể, chưa cô lập gain của learned head với packing/adjacency, chưa chứng minh contribution graph.',
        f"DeepSeek mới={len(receipts)} receipts, peak estimate=${result['new_judge_peak_estimate_usd']:.6f}. Ledger chung không reset hoặc tăng trần. Chi phí Qwen phục vụ không được endpoint báo."]
    if error:md+=['','Dừng: '+error]
    (out/'REPORT.md').write_text('\n'.join(md)+'\n');return result


def run(root):
    transfer_spec=read(root/'manifest.json');transfer.guard(transfer_spec)
    assert sha((root/'contexts200.jsonl').read_bytes())==read(root/'integrity.json')['sha256']
    allrows=lines(root/'contexts200.jsonl');ids=transfer_spec['qa20_indices']
    import csv
    with (ROOT/'logs/devset.csv').open() as f:qa={r['Question']:r for r in csv.DictReader(f)}
    rows=[{**r,'gold':qa[r['question']]['Gold Answer'],'type':qa[r['question']]['Type']} for r in allrows if r['id'] in ids and r['arm'] in ARMS]
    assert len(rows)==40
    out=root/'qa_qwen20';out.mkdir(exist_ok=True)
    template=runpy.run_path(str(ROOT/'minirag/prompt.py'))['PROMPTS']['rag_response'];judge=judge_template()
    spec={'indices':ids,'arms':list(ARMS),'generator':MODEL,'judge':deepseek.MODEL,'generation_rounds':1,'judge_passes':1,
        'max_generation_tokens':1024,'Qwen_sampling':'server defaults; seed13+question_id identical across arms',
        'judge_max_tokens':16,'judge_thinking':'disabled','response_type':'Multiple Paragraphs','seed_primary':13,
        'template_sha256':sha(template),'judge_template_sha256':sha(judge),'parent_manifest_sha256':sha((root/'manifest.json').read_bytes()),
        'runner_sha256':sha(Path(__file__).read_bytes()),'qa_csv_sha256':sha((ROOT/'logs/devset.csv').read_bytes()),'context_sha256':{f"{r['id']}:{r['arm']}":r['context_sha256'] for r in rows}}
    if (out/'manifest.json').exists():assert read(out/'manifest.json')==spec
    else:write(out/'manifest.json',spec)
    if not (out/'budget_start.json').exists():write(out/'budget_start.json',{'ledger':read(deepseek.OUT/'budget_ledger.json'),
        'receipt_count':len(lines(deepseek.OUT/'api_calls.jsonl'))})
    budget=read(deepseek.OUT/'approved_budget.json');ledger=read(deepseek.OUT/'budget_ledger.json')
    judgments={(r['id'],r['arm']):r for r in lines(out/'judgments.jsonl')}
    if budget['max_requests']-ledger['requests']<40-len(judgments):raise StopAPI('Remaining shared judge budget insufficient')
    write(out/'preflight.json',{'judge_calls_remaining_upper':40-len(judgments),
        'global_requests_left':budget['max_requests']-ledger['requests'],'input_tokens_left':budget['max_input_tokens']-ledger['input_tokens'],
        'USD_left':budget['max_usd']-ledger['cost_peak_estimate_usd'],'no_budget_increase':True})
    lookup=socket.getaddrinfo
    socket.getaddrinfo=lambda host,port,family=0,type=0,proto=0,flags=0:lookup(host,port,socket.AF_INET,type,proto,flags)
    answers={(r['id'],r['arm']):r for r in lines(out/'answers.jsonl')};by={(r['id'],r['arm']):r for r in rows}
    lock=threading.Lock();stop=threading.Event();order=list(ids);random.Random(2026111121).shuffle(order)
    def question(n,i):
        generator=Qwen(out);client=deepseek.DeepSeekAPI()
        tls=ssl.create_default_context();tls.maximum_version=ssl.TLSVersion.TLSv1_2
        client.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=tls))
        arms=list(ARMS if n%2==0 else ARMS[::-1])
        for arm in arms:
            if (i,arm) in answers:continue
            if stop.is_set():return
            r=by[i,arm];value=generator.call(r,template,spec['response_type'])
            a={**value,'id':i,'arm':arm,'question':r['question'],'gold':r['gold'],'type':r['type'],
                'context_sha256':r['context_sha256'],'generation_round':1}
            with lock:append(out/'answers.jsonl',a);answers[i,arm]=a
        random.Random(2026111122+i).shuffle(arms)
        for arm in arms:
            if (i,arm) in judgments:continue
            if stop.is_set():return
            r=by[i,arm];a=answers[i,arm];prompt=judge.format(question=r['question'],gold=r['gold'],pred=a['prediction'])
            for attempt in range(3):
                try:verdict=client.call(prompt,role='judge').strip().lower();break
                except StopAPI:
                    body={'model':deepseek.MODEL,'messages':[{'role':'user','content':prompt}],'max_tokens':16,'thinking':{'type':'disabled'}}
                    key=sha(json.dumps(body,ensure_ascii=False,sort_keys=True))
                    history=[r for r in lines(deepseek.OUT/'api_calls.jsonl') if r.get('request_sha256')==key]
                    if attempt==2 or not history or history[-1].get('finish_reason')!='length':raise
            if verdict not in LABELS:raise StopAPI('Invalid label; never convert failure to neither')
            j={'id':i,'arm':arm,'question':r['question'],'verdict':verdict,'prediction_sha256':a['prediction_sha256'],
                'judge_model':deepseek.MODEL,'judge_pass':1}
            with lock:append(out/'judgments.jsonl',j);judgments[i,arm]=j
    error=None
    with ThreadPoolExecutor(max_workers=3) as workers:
        tasks=[workers.submit(question,n,i) for n,i in enumerate(order)]
        for task in as_completed(tasks):
            try:task.result()
            except Exception as exc:stop.set();error=type(exc).__name__+': '+str(exc)
            with lock:
                result=summary(out,spec,rows,error);write(out/'status.json',{'state':'STOPPED' if error else 'RUNNING','completed_pairs':result['matched_questions'],'error':error})
            print(json.dumps({'Qwen_paired_questions':result['matched_questions'],'planned':20,'error':error}),flush=True)
    transfer.guard(transfer_spec);result=summary(out,spec,rows,error)
    write(out/'status.json',{'state':result['status'] if not error else 'STOPPED','completed_pairs':result['matched_questions'],'error':error})
    print(json.dumps(result),flush=True)
    if error:raise SystemExit(error)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=transfer.OUT);args=p.parse_args();run(args.output)
