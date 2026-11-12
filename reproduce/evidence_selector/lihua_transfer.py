"""Frozen LiHua dev200 transfer, then paired Qwen3B QA20 without RRF.

Primary treatment fixed BEFORE results: seed13 MLP + source adjacency.
Gold enters diagnostics/judging only after all retrieval selections are frozen.
"""
import argparse
from collections import Counter
import csv
from functools import lru_cache
import json
import os
from pathlib import Path
import random
import runpy
import statistics
import time

from reproduce.graph_ablation.study import ROOT,INDEX,read,lines,write,append,sha,hashfiles,csvfile
from minirag.extractive_context import catalogue,prefix,select,verify_output
from minirag.utils import encode_string_by_tiktoken,list_of_list_to_csv
from .inference import LearnedScorer

OUT=ROOT/'review/learned_selector_lihua_20261111'
POOLS=ROOT/'review/c0_time_rrf_budget200_20261111/ranking_inputs.jsonl'
FROZEN=ROOT/'logs/p0_paths/20261109_dev200'
PRIOR20=ROOT/'review/no_rrf_qa20_20261111'
ARTIFACTS=ROOT/'review/friendsqa_selector_kaggle_20261111/friendsqa_selector_v1'
ARMS=('BASELINE','COSINE_NEIGHBORS','MLP_UTTERANCE','MLP_NEIGHBORS')
PRIMARY='MLP_NEIGHBORS'
CODE=['reproduce/evidence_selector/lihua_transfer.py','reproduce/evidence_selector/inference.py',
      'minirag/extractive_context.py','minirag/utils.py']


@lru_cache(maxsize=32768)
def cost(text):return len(encode_string_by_tiktoken(text))


def render(start,ranked,result):
    labels={c:i for i,c in enumerate(ranked)}
    content=list_of_list_to_csv([['id','content']]+[[labels[c],result['texts'][c]] for c in result['chunk_ids']])
    return start+'-----Sources-----\n```csv\n'+content+'\n```\n'


def guard(spec):
    assert hashfiles(INDEX)==spec['index_sha256']
    assert sha((ROOT/'baseline.yaml').read_bytes())==spec['baseline_yaml_sha256']
    assert all(sha((ROOT/n).read_bytes())==h for n,h in spec['inputs_sha256'].items())
    assert all(sha((ROOT/n).read_bytes())==h for n,h in spec['code_sha256'].items())


def prepare(out):
    if (out/'integrity.json').exists():
        spec=read(out/'manifest.json');guard(spec)
        assert sha((out/'contexts200.jsonl').read_bytes())==read(out/'integrity.json')['sha256']
        return spec,lines(out/'contexts200.jsonl')
    if out.exists():raise ValueError('Output exists without integrity; inspect incomplete preparation')
    before=hashfiles(INDEX);prior=read(PRIOR20/'manifest.json');assert before==prior['index_sha256']
    pools=lines(POOLS);assert len(pools)==200 and {r['id'] for r in pools}==set(range(200))
    source_contexts={r['question']:r['context'].split('-----Sources-----')[0] for r in lines(FROZEN/'contexts.jsonl') if r['variant']=='C0'}
    chunks=read(INDEX/'kv_store_text_chunks.json');docs=read(INDEX/'kv_store_full_docs.json')
    baseline20={r['id']:r for r in lines(PRIOR20/'contexts.jsonl') if r['arm']=='BASELINE_NO_RRF'}
    spec={'scope':'Frozen LiHua dev200 transfer; main paired Qwen3B QA20 vs baseline without RECOMP/RRF',
        'arms':list(ARMS),'primary':PRIMARY,'primary_fixed_before_LiHua_outcomes':True,
        'scorer_checkpoint':'seed13, selected in advance; no selection among seeds using LiHua',
        'index_sha256':before,'baseline_yaml_sha256':sha((ROOT/'baseline.yaml').read_bytes()),
        'inputs_sha256':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in
            [POOLS,FROZEN/'contexts.jsonl',PRIOR20/'contexts.jsonl',PRIOR20/'manifest.json',
             ARTIFACTS/'seed13.safetensors',ARTIFACTS/'encoder.json',ARTIFACTS/'report.json']},
        'code_sha256':{n:sha((ROOT/n).read_bytes()) for n in CODE},
        'pool':'C0 original graph/P0 + Tai source-time top30, no RRF; dense boost internal to C0 retained',
        'source_budget':4000,'qa20_indices':prior['indices'],'qa20_type_counts':{'Single':16,'Multi':2,'Null':2},
        'no_LiHua_training_or_hyperparameter_tuning':True,'model_call_generation_in_prepare':0,'judge_calls_in_prepare':0,
        'metric_warning':'Compressed gold source-ID or answer-string retention is a proxy, NOT full-support evidence or QA accuracy',
        'input_recipe':'Same catalogue Time header + original speaker turn for learned and cosine scorers',
        'domain_shift':'FriendsQA training had speaker-prefixed turns without Time metadata',
        'native_baseline_reference':'20 historical no-RRF native contexts must match byte-for-byte'}
    out.mkdir(parents=True);write(out/'manifest.json',spec)
    scorer=LearnedScorer(ARTIFACTS,ROOT/'.cache/minirag/learned_selector_vectors')
    used={c for p in pools for c in p['graph_time_ids'] if c in chunks};cats={};texts={}
    for c in used:
        texts[c]=chunks[c]['content'];did=chunks[c].get('full_doc_id')
        cats[c]=catalogue(c,texts[c],did,docs.get(did,{}).get('content'))
    alltexts=[p['question'] for p in pools]+[u['score_text'] for cat in cats.values() for u in cat['segments'] if u['kind']!='header']
    scorer.encode(alltexts)
    write(out/'model_provenance.json',{**scorer.provenance,'encoded_unique_texts':len(scorer.cache),
        'encoding_seconds':scorer.elapsed,'input_texts_over256':sum(n>256 for n in scorer.token_counts.values())})
    records=[];parity=0;span_checks=0
    for ordinal,pool in enumerate(sorted(pools,key=lambda p:p['id'])):
        i=pool['id'];q=pool['question'];raw=list(pool['graph_time_ids']);ranked=[c for c in raw if c in chunks]
        local={c:cats[c] for c in ranked};localtexts={c:texts[c] for c in ranked}
        scores=scorer.score(q,local);cosine=scorer.score(q,local,cosine=True)
        kept=prefix(ranked,localtexts,4000,cost)
        group={}
        for arm in ARMS:
            start=time.perf_counter()
            if arm=='BASELINE':
                result={'mode':'whole','chunk_ids':kept,'texts':{c:texts[c] for c in kept},
                    'spans':{c:[{'start':0,'end':len(texts[c]),'quote':texts[c]}] for c in kept},
                    'tokens':sum(cost(texts[c]) for c in kept),'target_bundles':[]}
            else:
                result=select(ranked,localtexts,local,cosine if arm=='COSINE_NEIGHBORS' else scores,
                              'utterance' if arm=='MLP_UTTERANCE' else 'neighbors',cost=cost)
                verify_output(result,localtexts,local)
            for c in result['chunk_ids']:
                for s in result['spans'][c]:
                    assert texts[c][s['start']:s['end']]==s['quote'];span_checks+=1
            assert result['tokens']<=4000 and set(result['chunk_ids'])<=set(ranked)
            ctx=render(source_contexts[q],raw,result)
            record={'id':i,'question':q,'arm':arm,'context':ctx,'context_sha256':sha(ctx),
                'chunk_ids':result['chunk_ids'],'pool_ids':raw,'source_content_tokens':result['tokens'],
                'selected_texts':result['texts'],'selected_spans':result['spans'],
                'selection_ms':1000*(time.perf_counter()-start)}
            if arm=='BASELINE' and i in baseline20:
                assert record['context_sha256']==baseline20[i]['context_sha256'];parity+=1
            append(out/'contexts200.jsonl',record);records.append(record);group[arm]=record
        assert len({r['context'].split('-----Sources-----')[0] for r in group.values()})==1
        if (ordinal+1)%20==0:print(json.dumps({'offline_questions':ordinal+1,'total':200,'QA_Judge_calls':0}),flush=True)
    assert parity==20;guard(spec)
    write(out/'integrity.json',{'sha256':sha((out/'contexts200.jsonl').read_bytes()),'contexts':len(records),
        'exact_native_baseline20_matches':parity,'verbatim_source_span_checks':span_checks,
        'all_pools_and_Entities_unchanged':True,'all_source_budgets_le4000':True,'index_baseline_unchanged':True})
    return spec,records


def diagnostics(out,spec,records):
    # Gold labels are loaded only AFTER all model scores and contexts are frozen.
    from reproduce.path.path_common import gold_map
    gold=gold_map()
    with (ROOT/'logs/devset.csv').open() as f:qa={r['Question']:r for r in csv.DictReader(f)}
    chunks=read(INDEX/'kv_store_text_chunks.json');labelled=[r for r in records if r['arm']=='BASELINE' and gold.get(r['question'])]
    denom=sum(len(set(gold[r['question']])) for r in labelled);metrics={};norm=lambda s:' '.join(s.casefold().split())
    for arm in ARMS:
        rows=[r for r in records if r['arm']==arm];hit=allhit=literal_n=literal=0
        for r in rows:
            g=set(gold.get(r['question'],[]));texts=r['selected_texts']
            if g:hit+=len(g&set(r['chunk_ids']));allhit+=int(g<=set(r['chunk_ids']))
            answer=qa[r['question']]['Gold Answer'].strip()
            if g and len(norm(answer))>=4 and norm(answer) not in ('yes','no','unknown','none'):
                eligible=[c for c in g if c in chunks and norm(answer) in norm(chunks[c]['content'])]
                if eligible:
                    literal_n+=1;literal+=int(any(c in texts and norm(answer) in norm(texts[c]) for c in eligible))
        metrics[arm]={'questions':len(rows),'labelled_questions':len(labelled),'gold_source_pairs':denom,
            'gold_source_IDs_with_any_text_proxy':hit,'all_gold_source_IDs_present_queries_proxy':allhit,
            'literal_answer_eligible_queries':literal_n,'literal_answer_retained_queries_proxy':literal,
            'median_source_tokens':statistics.median(r['source_content_tokens'] for r in rows),
            'median_selection_ms':statistics.median(r['selection_ms'] for r in rows),
            'full_support_recall':None,'QA_ACC_ERR':None}
    write(out/'offline_diagnostics.json',{'metrics':metrics,'compressed_proxy_not_support_recall':True,'gold_not_used_to_score_select':True,
          'fixed_primary':PRIMARY,'no_model_or_hyperparameter_selection_from_diagnostics':True})
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    spec,records=prepare(a.output);metrics=diagnostics(a.output,spec,records);print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
