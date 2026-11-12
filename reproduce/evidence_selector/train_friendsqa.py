"""FriendsQA answer-utterance ranking with frozen MiniLM + 98,433-parameter MLP.

Task labels locate annotated answer utterances, not all sufficient evidence.
Only official train labels update weights; dev selects epochs; test never selects.
No LiHua, Qwen generation, LLM labels, Judge, RRF or production-index writes.
"""
import argparse
from collections import Counter
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATA_REVISION='9ce6c98a6acf13c87d96b6811e4409b5de43bdfe'
DATA_BASE=f'https://raw.githubusercontent.com/emorynlp/FriendsQA/{DATA_REVISION}/'
ENCODER='sentence-transformers/all-MiniLM-L6-v2'
ENCODER_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
DATA_FILES={
 'train':('friendsqa_trn.json','d63685693b30cee5d33b7cf2e6408cd284c05e49be1b3b06954c4d01cfe1c91b'),
 'dev':('friendsqa_dev.json','3a85d3fcb175589ffd050fb4bf431da77a6e12589c8ec03f1d54060ce33ec703'),
 'test':('friendsqa_tst.json','a98f24d4b04ac9f8d9468088febf304bd90eada134537277c1a425e31c0372bc')}
LICENSE_SHA='35a15bf1a8802e225106045dc4bb6c668a376e445b89f783471156fa26cc16b4'


def digest(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.writing')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


@dataclass
class Config:
    data_dir:str='data/friendsqa'
    output_dir:str='output/friendsqa_selector'
    cache_dir:str='cache/friendsqa_selector'
    seeds:tuple=(13,37,73)
    epochs:int=15
    patience:int=3
    batch_size:int=64
    encode_batch_size:int=64
    hidden:int=64
    lr:float=0.001
    weight_decay:float=0.0001
    max_length:int=256
    device:str='auto'
    threads:int=4
    bootstrap:int=2000
    smoke_questions:int=0
    local_files_only:bool=False


def download_data(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    files=[('dat/'+name,expected) for name,expected in DATA_FILES.values()]+[('LICENSE.txt',LICENSE_SHA)]
    for name,expected in files:
        target=directory/Path(name).name
        if not target.exists():
            with urllib.request.urlopen(DATA_BASE+name,timeout=90) as response:data=response.read()
            if digest(data)!=expected:raise ValueError('Dataset download SHA mismatch: '+name)
            target.write_bytes(data)
        if digest(target.read_bytes())!=expected:raise ValueError('Dataset cache SHA mismatch: '+name)
    return directory


def parse_data(raw,split):
    """Accept the author's literal 'utterances:' key and uid-based answer labels."""
    rows=[];scene_ids=set();qids=set()
    for item in raw['data']:
        title=item['title'];scene_ids.add(title)
        for pi,paragraph in enumerate(item['paragraphs']):
            units=paragraph.get('utterances:',paragraph.get('utterances'))
            if not units:raise ValueError('Missing dialogue utterances')
            uids=[u['uid'] for u in units]
            if len(set(uids))!=len(uids):raise ValueError('Duplicate utterance uid')
            positions={uid:i for i,uid in enumerate(uids)}
            texts=[', '.join(u['speakers'])+': '+u['utterance'] for u in units]
            for q in paragraph['qas']:
                if q['id'] in qids:raise ValueError('Duplicate QA id')
                qids.add(q['id'])
                if not q.get('answers') or not q['question'].strip():raise ValueError('Missing annotated answer/question')
                if any(a['utterance_id'] not in positions for a in q['answers']):raise ValueError('Unknown answer uid')
                gold=sorted({positions[a['utterance_id']] for a in q['answers']})
                rows.append({'id':q['id'],'scene_id':title,'dialogue_id':f'{title}/p{pi}',
                    'split':split,'question':q['question'],'utterance_ids':uids,
                    'utterances':texts,'gold_positions':gold,
                    'speaker_answer':any(a.get('is_speaker',False) for a in q['answers'])})
    return rows,scene_ids


def load_splits(directory):
    result={};scene_sets={};manifest={}
    for split,(filename,expected) in DATA_FILES.items():
        p=Path(directory)/filename
        if digest(p.read_bytes())!=expected:raise ValueError('Unpinned dataset split')
        result[split],scene_sets[split]=parse_data(json.loads(p.read_text()),split)
        manifest[split]={'filename':filename,'sha256':expected,'scenes':len(scene_sets[split]),
            'questions':len(result[split]),'speaker_answer_questions':sum(r['speaker_answer'] for r in result[split])}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        if scene_sets[a]&scene_sets[b]:raise ValueError('Scene overlap: '+a+'/'+b)
    return result,manifest


def feature_vectors(q,u):
    return torch.cat((q,u,torch.abs(q-u),q*u),dim=-1)


class Head(nn.Module):
    def __init__(self,dim=384,hidden=64):
        super().__init__();self.network=nn.Sequential(nn.Linear(4*dim,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,q,u):return self.network(feature_vectors(q,u)).squeeze(-1)


def ranking_loss(scores,valid,gold):
    """Multi-positive listwise NLL: retrieving any annotated answer location suffices.

    Other turns are ranking distractors for answer location, not verified
    negatives for all semantic evidence/context relevance.
    """
    if not torch.all((gold&valid).any(dim=1)):raise ValueError('Each query needs a valid gold location')
    if torch.any(gold&~valid):raise ValueError('Gold on padding')
    return (torch.logsumexp(scores.masked_fill(~valid,-torch.inf),dim=1)
            -torch.logsumexp(scores.masked_fill(~gold,-torch.inf),dim=1)).mean()


def metric_rows(rows,scores):
    output=[]
    for row,values in zip(rows,scores):
        if len(values)!=len(row['utterances']) or not np.isfinite(values).all():raise ValueError('Invalid scores')
        order=np.argsort(-np.asarray(values),kind='stable').tolist();gold=set(row['gold_positions'])
        first=min(order.index(i)+1 for i in gold)
        output.append({'id':row['id'],'scene_id':row['scene_id'],'gold_utterance_ids':[row['utterance_ids'][i] for i in sorted(gold)],
            'ranked_utterance_ids':[row['utterance_ids'][i] for i in order],
            'hit1':int(first<=1),'hit3':int(first<=3),'hit5':int(first<=5),'rr':1/first,
            'annotated_answer_location_recall3':len(set(order[:3])&gold)/len(gold)})
    return output


def aggregate(records):
    if not records:raise ValueError('No evaluation questions')
    return {'questions':len(records),'Hit@1':float(np.mean([r['hit1'] for r in records])),
        'Hit@3':float(np.mean([r['hit3'] for r in records])),
        'Hit@5':float(np.mean([r['hit5'] for r in records])),
        'MRR':float(np.mean([r['rr'] for r in records])),
        'annotated_answer_location_recall@3':float(np.mean([r['annotated_answer_location_recall3'] for r in records]))}


def comparison(baseline,treatment,bootstrap=2000,seed=13):
    if [r['id'] for r in baseline]!=[r['id'] for r in treatment]:raise ValueError('Unmatched evaluation rows')
    b=sum(not x['hit1'] and y['hit1'] for x,y in zip(baseline,treatment))
    c=sum(x['hit1'] and not y['hit1'] for x,y in zip(baseline,treatment));n=b+c
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.
    groups={}
    for x,y in zip(baseline,treatment):groups.setdefault(x['scene_id'],[]).append(y['hit1']-x['hit1'])
    values=list(groups.values());rng=np.random.default_rng(seed);deltas=[]
    sums=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values])
    for _ in range(bootstrap):
        idx=rng.integers(0,len(values),size=len(values));deltas.append(float(sums[idx].sum()/counts[idx].sum()))
    return {'Hit@1_up':b,'Hit@1_down':c,'net':b-c,'exact_McNemar_p':p,
        'paired_delta_Hit@1':float(sum(sums)/sum(counts)),
        'scene_cluster_bootstrap_95_CI':np.quantile(deltas,[.025,.975]).tolist() if deltas else None,
        'bootstrap_replicates':bootstrap,'McNemar_question_independence_limitation':True}


def text_index(splits):
    texts=[];index={};packed={}
    def put(text):
        if text not in index:index[text]=len(texts);texts.append(text)
        return index[text]
    for split,rows in splits.items():
        packed[split]=[dict(r,qv=put(r['question']),uv=[put(u) for u in r['utterances']]) for r in rows]
    return texts,packed


def encode_texts(texts,cfg,device):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    directory=Path(snapshot_download(ENCODER,revision=ENCODER_REVISION,
        allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json'],
        local_files_only=cfg.local_files_only))
    model_hash=digest((directory/'model.safetensors').read_bytes())
    spec={'model':ENCODER,'revision':ENCODER_REVISION,'weights_sha256':model_hash,
        'pooling':'attention-mask mean including special tokens; L2 normalized',
        'max_length':cfg.max_length,'texts_sha256':digest(json.dumps(texts,ensure_ascii=False)),
        'dim':384,'frozen':True}
    cache=Path(cfg.cache_dir);cache.mkdir(parents=True,exist_ok=True)
    key=digest(json.dumps(spec,sort_keys=True));npfile=cache/(key+'.npy')
    if npfile.exists():
        vectors=np.load(npfile,allow_pickle=False)
        if vectors.shape!=(len(texts),384) or not np.isfinite(vectors).all():raise ValueError('Bad embedding cache')
        return vectors,{**spec,'cache_hit':True,'seconds':0.,'truncated_inputs':None}
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    model=AutoModel.from_config(AutoConfig.from_pretrained(directory,local_files_only=True))
    state=load((directory/'model.safetensors').read_bytes());state.pop('embeddings.position_ids',None)
    model.load_state_dict(state,strict=True);model.to(device).eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    parameter_count=sum(p.numel() for p in model.parameters())
    vectors=np.empty((len(texts),384),dtype=np.float32);truncated=0;start=time.perf_counter()
    # Stable length buckets improve throughput without changing input content.
    lengths=tokenizer(texts,add_special_tokens=True,truncation=False,verbose=False)['input_ids']
    order=sorted(range(len(texts)),key=lambda i:len(lengths[i]))
    truncated=sum(len(v)>cfg.max_length for v in lengths)
    for begin in range(0,len(order),cfg.encode_batch_size):
        idx=order[begin:begin+cfg.encode_batch_size]
        batch=tokenizer([texts[i] for i in idx],padding=True,truncation=True,max_length=cfg.max_length,return_tensors='pt').to(device)
        with torch.inference_mode():
            hidden=model(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
            embedding=(hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1)
            embedding=F.normalize(embedding,p=2,dim=1)
        vectors[idx]=embedding.float().cpu().numpy()
        if begin%6400==0:print(json.dumps({'encoded_texts':min(begin+len(idx),len(texts)),'total':len(texts)}),flush=True)
    elapsed=time.perf_counter()-start
    if not np.isfinite(vectors).all():raise ValueError('Nonfinite encoder output')
    np.save(npfile,vectors,allow_pickle=False)
    del model,state
    if device.type=='cuda':torch.cuda.empty_cache()
    return vectors,{**spec,'cache_hit':False,'seconds':elapsed,'truncated_inputs':truncated,
        'encoder_parameters':parameter_count,'device':str(device)}


def batch_tensors(rows,vecs,device):
    lengths=[len(r['uv']) for r in rows];width=max(lengths)
    uids=np.zeros((len(rows),width),dtype=np.int64);valid=np.zeros_like(uids,dtype=bool);gold=np.zeros_like(uids,dtype=bool)
    for i,r in enumerate(rows):
        uids[i,:lengths[i]]=r['uv'];valid[i,:lengths[i]]=True;gold[i,r['gold_positions']]=True
    q=torch.as_tensor(vecs[[r['qv'] for r in rows]],device=device).unsqueeze(1).expand(-1,width,-1)
    u=torch.as_tensor(vecs[uids],device=device)
    return q,u,torch.as_tensor(valid,device=device),torch.as_tensor(gold,device=device),lengths


def predict(rows,vecs,cfg,device,model=None):
    scores=[]
    if model is not None:model.eval()
    with torch.inference_mode():
        for offset in range(0,len(rows),cfg.batch_size):
            group=rows[offset:offset+cfg.batch_size];q,u,valid,gold,lengths=batch_tensors(group,vecs,device)
            values=(q*u).sum(-1) if model is None else model(q,u)
            scores.extend(v[:length].float().cpu().numpy() for v,length in zip(values,lengths))
    return metric_rows(rows,scores)


def train_one(seed,rows,dev,vecs,cfg,device,out):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=Head(hidden=cfg.hidden).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    best=None;best_state=None;stale=0;history=[];start=time.perf_counter()
    rng=np.random.default_rng(seed)
    for epoch in range(1,cfg.epochs+1):
        model.train();order=rng.permutation(len(rows));total_loss=0.;count=0
        for offset in range(0,len(order),cfg.batch_size):
            group=[rows[i] for i in order[offset:offset+cfg.batch_size]]
            q,u,valid,gold,_=batch_tensors(group,vecs,device)
            optimizer.zero_grad(set_to_none=True);loss=ranking_loss(model(q,u),valid,gold)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total_loss+=float(loss.detach())*len(group);count+=len(group)
        metrics=aggregate(predict(dev,vecs,cfg,device,model));key=(metrics['Hit@1'],metrics['MRR'])
        improved=best is None or key>best
        if improved:
            best=key;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0;best_epoch=epoch
        else:stale+=1
        record={'seed':seed,'epoch':epoch,'train_loss':total_loss/count,'dev':metrics,'best_epoch':best_epoch}
        history.append(record);print(json.dumps(record),flush=True);write(out/f'seed{seed}_history.json',history)
        if stale>=cfg.patience:break
    assert best_state is not None
    model.load_state_dict(best_state);model.eval()
    from safetensors.torch import save_file
    weights=out/f'seed{seed}.safetensors';save_file(best_state,str(weights))
    return model,{'seed':seed,'best_epoch':best_epoch,'epochs_run':len(history),
        'train_seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),
        'checkpoint':weights.name,'sha256':digest(weights.read_bytes()),'best_dev':history[best_epoch-1]['dev']}


def write_predictions(path,rows):
    with Path(path).open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def benchmark_head(model,rows,vecs):
    """CPU-only per-query scoring, excludes embedding; identical threads/inputs."""
    model=model.cpu().eval();trials={'cosine':[],'mlp':[]}
    with torch.inference_mode():
        for row in rows[:110]:
            u=torch.from_numpy(vecs[row['uv']]);q=torch.from_numpy(vecs[row['qv']]).expand_as(u)
            start=time.perf_counter();(q*u).sum(-1);trials['cosine'].append((time.perf_counter()-start)*1000)
            start=time.perf_counter();model(q,u);trials['mlp'].append((time.perf_counter()-start)*1000)
    return {name:{'median_ms':float(np.median(v[10:])),'p95_ms':float(np.quantile(v[10:],.95)),
                  'queries':len(v[10:]),'scope':'CPU scoring of cached vectors, excluding encoder'}
            for name,v in trials.items() if len(v)>10}


def run_experiment(cfg):
    torch.set_num_threads(cfg.threads)
    if cfg.hidden!=64:raise ValueError('This registered experiment fixes hidden=64')
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Completed output exists; choose a new directory')
    device=torch.device('cuda' if cfg.device=='auto' and torch.cuda.is_available() else 'cpu' if cfg.device=='auto' else cfg.device)
    if device.type=='cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    data=download_data(cfg.data_dir);splits,data_spec=load_splits(data)
    full_counts={s:len(v) for s,v in splits.items()}
    if cfg.smoke_questions:
        splits={s:r[:cfg.smoke_questions] for s,r in splits.items()}
    spec={'config':asdict(cfg),'task':'annotated answer-utterance ranking, not full-support evidence or QA accuracy',
        'dataset':'FriendsQA v2 pinned author files','dataset_revision':DATA_REVISION,'splits':data_spec,
        'actual_split_sizes_author_files_take_precedence_over_README':full_counts,
        'encoder':ENCODER,'encoder_revision':ENCODER_REVISION,'architecture':'[q,u,abs(q-u),q*u] -> Linear1536x64 -> ReLU -> Linear64x1',
        'trainable_parameters':98433,'loss':'multi-positive listwise NLL, any annotated answer location',
        'checkpoint_selection':'dev Hit@1 primary, dev MRR tie-break; fixed early stopping',
        'primary_seed':13,'primary_endpoint':'test Hit@1 vs same-encoder cosine',
        'secondary_seeds':[s for s in cfg.seeds if s!=13],'secondary_metrics':['MRR','Hit@3','Hit@5'],
        'test_never_selects_checkpoint_hyperparameters_or_seed':True,'LiHua_used':False,
        'LLM_or_Judge_calls':0,'RRF':False,'Qwen3B_unchanged':True,'device':str(device),
        'smoke_only':bool(cfg.smoke_questions),'software':{'torch':torch.__version__,'numpy':np.__version__}}
    spec['trainer_source_sha256']=digest(Path(__file__).read_bytes())
    write(out/'protocol.json',spec)
    (out/'FRIENDSQA_LICENSE.txt').write_text((data/'LICENSE.txt').read_text())
    texts,packed=text_index(splits);vectors,enc_spec=encode_texts(texts,cfg,device);write(out/'encoder.json',enc_spec)
    cosine={s:predict(r,vectors,cfg,device) for s,r in packed.items() if s in ('dev','test')}
    for s,pred in cosine.items():write_predictions(out/f'cosine_{s}_predictions.jsonl',pred)
    baseline={s:aggregate(pred) for s,pred in cosine.items()};results=[]
    # Train and select all models on dev BEFORE inspecting the test results.
    trained=[]
    for seed in cfg.seeds:
        model,record=train_one(seed,packed['train'],packed['dev'],vectors,cfg,device,out)
        trained.append(record);del model
    for record in trained:
        from safetensors.torch import load_file
        model=Head(hidden=cfg.hidden).to(device);model.load_state_dict(load_file(str(out/record['checkpoint']),device=str(device)))
        predictions={s:predict(packed[s],vectors,cfg,device,model) for s in ('dev','test')}
        for s,pred in predictions.items():write_predictions(out/f"seed{record['seed']}_{s}_predictions.jsonl",pred)
        metrics={s:aggregate(pred) for s,pred in predictions.items()}
        paired=comparison(cosine['test'],predictions['test'],cfg.bootstrap,record['seed'])
        timing=benchmark_head(model,packed['test'],vectors)
        results.append({**record,'metrics':metrics,'paired_test_vs_cosine':paired,'scoring_latency':timing})
    summary={k:{'mean':float(np.mean([r['metrics']['test'][k] for r in results])),
                'std_between_training_seeds':float(np.std([r['metrics']['test'][k] for r in results],ddof=1)) if len(results)>1 else None}
             for k in ('Hit@1','Hit@3','Hit@5','MRR')}
    report={'status':'COMPLETE','task':spec['task'],'smoke_only':bool(cfg.smoke_questions),
        'baseline_cosine':baseline,'seeds':results,'test_seed_summary':summary,'primary_seed':13,
        'test_used_for_selection':False,'model_location':'checkpoints in this output directory',
        'QA_ACC_ERR_not_measured':True,'graph_contribution_not_measured':True,
        'no_LiHua_training_or_production_changes':True}
    write(out/'report.json',report)
    md=['# FriendsQA frozen MiniLM + MLP selector','',
        'Task: annotated answer-utterance localization. These scores are not QA ACC/ERR or full-support evidence recall.','',
        '| Model | Test Hit@1 | Test Hit@3 | Test MRR |','|---|---:|---:|---:|']
    b=baseline['test'];md.append(f"| Same MiniLM cosine | {b['Hit@1']:.4f} | {b['Hit@3']:.4f} | {b['MRR']:.4f} |")
    notes=[]
    for r in results:
        m=r['metrics']['test'];md.append(f"| MLP seed {r['seed']} | {m['Hit@1']:.4f} | {m['Hit@3']:.4f} | {m['MRR']:.4f} |")
        p=r['paired_test_vs_cosine'];notes.append(f"\nSeed {r['seed']}: {p['Hit@1_up']} up / {p['Hit@1_down']} down; exact McNemar p={p['exact_McNemar_p']:.6g}; scene-cluster bootstrap95% CI={p['scene_cluster_bootstrap_95_CI']}.\n")
    md+=notes
    md+=['','Primary seed13 is fixed in advance; all seeds are reported, not selected by test. '
        'Official scene-disjoint splits; train-only gradients, dev-only epoch selection. '
        'Unannotated turns are ranking distractors for answer location, not verified non-evidence. '
        'FriendsQA does not directly label required neighboring turns. No Qwen/QA/Judge calls. '
        'Transfer to LiHua and effects on final ACC/ERR remain unmeasured.']
    (out/'REPORT.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({'status':'COMPLETE','baseline':baseline['test'],'test_seed_summary':summary}),flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',default='data/friendsqa');p.add_argument('--output-dir',default='output/friendsqa_selector')
    p.add_argument('--cache-dir',default='cache/friendsqa_selector');p.add_argument('--device',default='auto')
    p.add_argument('--smoke-questions',type=int,default=0);p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--local-files-only',action='store_true');p.add_argument('--seed',type=int)
    a=p.parse_args(argv);cfg=Config(data_dir=a.data_dir,output_dir=a.output_dir,cache_dir=a.cache_dir,
        device=a.device,smoke_questions=a.smoke_questions,epochs=a.epochs,local_files_only=a.local_files_only,
        seeds=(a.seed,) if a.seed is not None else (13,37,73))
    run_experiment(cfg)


if __name__ == '__main__':main()
