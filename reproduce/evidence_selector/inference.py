"""Inference-only adapter for the fixed FriendsQA seed13 checkpoint.

Preserves the trained encoder revision, pooling, normalization and MLP features.
Does not inspect answers/evidence or update any model parameter.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .train_friendsqa import Head,ENCODER,ENCODER_REVISION


class LearnedScorer:
    def __init__(self,artifacts,cache_dir=None):
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig,AutoModel,AutoTokenizer
        from safetensors.torch import load,load_file
        torch.set_num_threads(4);artifacts=Path(artifacts)
        report=json.loads((artifacts/'report.json').read_text())
        if report['status']!='COMPLETE' or report['smoke_only']:raise ValueError('Only a complete fixed training run is accepted')
        record=next(r for r in report['seeds'] if r['seed']==13)
        checkpoint=artifacts/record['checkpoint']
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Checkpoint hash mismatch')
        spec=json.loads((artifacts/'encoder.json').read_text())
        if spec['model']!=ENCODER or spec['revision']!=ENCODER_REVISION or spec['max_length']!=256:raise ValueError('Unexpected encoder recipe')
        directory=Path(snapshot_download(ENCODER,revision=ENCODER_REVISION,local_files_only=True))
        weights=(directory/'model.safetensors').read_bytes()
        if hashlib.sha256(weights).hexdigest()!=spec['weights_sha256']:raise ValueError('Encoder hash mismatch')
        self.tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
        self.encoder=AutoModel.from_config(AutoConfig.from_pretrained(directory,local_files_only=True))
        state=load(weights);state.pop('embeddings.position_ids',None);self.encoder.load_state_dict(state,strict=True);self.encoder.eval()
        self.head=Head();self.head.load_state_dict(load_file(str(checkpoint)));self.head.eval()
        for p in [*self.encoder.parameters(),*self.head.parameters()]:p.requires_grad_(False)
        self.provenance={'encoder':ENCODER,'revision':ENCODER_REVISION,'encoder_sha256':spec['weights_sha256'],
            'checkpoint_sha256':record['sha256'],'fixed_seed':13,'train_dataset':'FriendsQA train only',
            'LiHua_training':False,'max_length':256,'pooling':'attention-mask mean + L2 normalization',
            'feature':'q,u,abs(q-u),q*u','trainable_head_parameters':98433,
            'input':'Same catalogue score_text (Time header + original speaker turn) as RECOMP comparator',
            'domain_shift':'FriendsQA training had speaker-prefixed turns without Time headers',
            'device':'CPU','generation_calls':0,'judge_calls':0}
        self.cache={};self.elapsed=0.;self.token_counts={}
        self.cache_dir=Path(cache_dir) if cache_dir else None
        if self.cache_dir:self.cache_dir.mkdir(parents=True,exist_ok=True)

    def encode(self,texts):
        missing=[]
        for text in dict.fromkeys(texts):
            if text in self.cache:continue
            key=hashlib.sha256((self.provenance['encoder_sha256']+'|masked_mean_L2_256|'+text).encode()).hexdigest()
            path=self.cache_dir/(key+'.npy') if self.cache_dir else None
            if path is not None and path.exists():
                vector=np.load(path,allow_pickle=False)
                if vector.shape!=(384,) or not np.isfinite(vector).all():raise ValueError('Invalid embedding cache')
                self.cache[text]=vector
            else:missing.append((text,path))
        lengths=self.tokenizer([x[0] for x in missing],truncation=False,add_special_tokens=True,verbose=False)['input_ids'] if missing else []
        self.token_counts.update({x[0]:len(ids) for x,ids in zip(missing,lengths)})
        missing=sorted(missing,key=lambda x:len(x[0]));start=time.perf_counter()
        for offset in range(0,len(missing),64):
            batch=missing[offset:offset+64]
            tokens=self.tokenizer([x[0] for x in batch],padding=True,truncation=True,max_length=256,return_tensors='pt')
            with torch.inference_mode():
                hidden=self.encoder(**tokens).last_hidden_state;mask=tokens['attention_mask'].unsqueeze(-1)
                vectors=F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp(min=1),p=2,dim=1).numpy()
            for (text,path),v in zip(batch,vectors):
                self.cache[text]=v.copy()
                if path is not None:np.save(path,v,allow_pickle=False)
        self.elapsed+=time.perf_counter()-start
        return np.stack([self.cache[t] for t in texts]) if texts else np.empty((0,384),np.float32)

    def score(self,query,cats,cosine=False):
        units=[s for cat in cats.values() for s in cat['segments'] if s['kind']!='header']
        vectors=self.encode([query,*[u['score_text'] for u in units]])
        if not units:return {}
        q=torch.from_numpy(vectors[0]).unsqueeze(0).expand(len(units),-1);u=torch.from_numpy(vectors[1:])
        with torch.inference_mode():scores=(q*u).sum(-1) if cosine else self.head(q,u)
        values=scores.numpy()
        if not np.isfinite(values).all():raise ValueError('Nonfinite scores')
        return {u['id']:float(s) for u,s in zip(units,values)}
