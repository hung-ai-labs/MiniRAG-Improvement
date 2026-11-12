"""Read-only original-index sidecars, exact frozen-context replay and checkpoints."""
import ast
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import time

import networkx as nx
import numpy as np
import tiktoken

ROOT=Path(__file__).resolve().parents[2]
INDEX=ROOT/'LiHua-World-qwen-modal'
OUT=ROOT/'review/minirag_graph_8arm_ablation'
FROZEN=ROOT/'logs/p0_paths/20261109_dev200'
CANARY=[1,2,11,15,22,40]
COMPARISONS=[('B','G0'),('P','B'),('TS','P'),('TE','TS'),('D','P'),('TD','TE'),('TD','D'),('TDV','TD')]


def sha(data):return hashlib.sha256(data if isinstance(data,bytes) else data.encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()] if Path(p).exists() else []
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.writing');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,p)
def append(p,v):
 with Path(p).open('a') as f:f.write(json.dumps(v,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def csvfile(p,rows,fields):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def hashfiles(directory):return {str(p.relative_to(directory)):sha(p.read_bytes()) for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='.DS_Store'}


def freeze():
 from minirag.relation_metadata import ARMS,FEATURE_NAMES
 OUT.mkdir(parents=True,exist_ok=True)
 source_plan=read(FROZEN/'manifest.json')
 hashes={n:sha((INDEX/n).read_bytes()) for n in source_plan['index_sha256']}
 if hashes!=source_plan['index_sha256']:raise ValueError('Original index differs from frozen C3')
 qa=list(csv.DictReader((ROOT/'logs/devset.csv').open()))
 assert len(qa)==200 and len({r['Question'] for r in qa})==200
 rest=[i for i in range(200) if i not in CANARY]
 validation=sorted(rest,key=lambda i:sha(qa[i]['Question']))[:20]
 spec={'scope':'200 frozen dev questions, one judge pass, one generation sample; exploratory not pristine holdout',
       'canary_indices':CANARY,'validation_indices':validation,'extension_indices':[i for i in rest if i not in validation],
       'questions':[dict(id=i,**r) for i,r in enumerate(qa)],'generator':'gemini-flash-lite-latest','judge':'gemini-flash-lite-latest',
       'profile':'C3-P0 relation-lift','budget':4000,'top_k':60,'top30':30,'judge_repeats':1,
       'fixed_before_new_outcomes':True,'canary_gate':'ACC/ERR nonworse and either one endpoint strictly better OR strict evidence-pair gain with changed contexts; no unexplained >1 regression among six',
       'validation_gate':'same endpoint direction on 20 fixed new-phase validation questions, evidence no worse',
       'old_index_sha256':hashes,'old_source_plan_sha256':sha((FROZEN/'manifest.json').read_bytes()),
       'comparisons':COMPARISONS,'semantic_reviews':0,'directions_verified':0,
       'max_fix_rounds':2,'limits':{'deepseek_requests':8,'deepseek_input':24000,'deepseek_output':16000,'deepseek_USD':0.5},
       'no_key_rotation':True,'paid_generation_authorized':False}
 spec=json.loads(json.dumps(spec))  # tuple/list equality must survive JSON resume.
 target=OUT/'question_manifest.json'
 if target.exists():
  if read(target)!=spec:raise ValueError('Frozen protocol differs; do not overwrite')
 else:write(target,spec)
 write(OUT/'arm_config.json',{a:dict(zip(FEATURE_NAMES,v)) for a,v in ARMS.items()})
 return spec


def guard(spec):
 assert {n:sha((INDEX/n).read_bytes()) for n in spec['old_index_sha256']}==spec['old_index_sha256'],'Original index changed'


def sidecar():
 from minirag.relation_metadata import validate_span,validate_sidecar
 from minirag import temporal,source_event_time
 chunks=read(INDEX/'kv_store_text_chunks.json');docs=read(INDEX/'kv_store_full_docs.json')
 graph=nx.read_graphml(INDEX/'graph_chunk_entity_relation.graphml')
 by_time,years=temporal.overlay(docs,chunks)
 dated=source_event_time.build_overlay(docs,chunks,graph)
 source=ROOT/'review/graph_v2_stage1g_20261110/results_final/combined/overlay.json'
 v=read(source);base=v['base_stage1f']
 entities={r['entity_id']:r['canonical_name'] for r in base['entities']}
 spans={r['span_id']:r for r in base['spans']}
 # Cosmetic quotes/case only; no whitespace folding, alias union or entity merge.
 names={str(n).strip('"').upper():n for n in graph}
 relations=[];rejected=[]
 for rel in base['relations']:
  keys=[entities[rel['source_entity']],entities[rel['target_entity']]]
  endpoints=[names.get(k.upper()) for k in keys]
  for ev in rel['evidence']:
   for sid in ev.get('span_ids',[]):
    s=spans[sid]
    try:
     if None in endpoints or not graph.has_edge(*endpoints):raise ValueError('No exact physical endpoints; no alias guess')
     cid=s['source_chunk_id']
     if cid not in graph.edges[tuple(endpoints)].get('source_id','').split('<SEP>'):raise ValueError('No exact physical edge-source membership')
     obj=dict(relation_id=rel['relation_id'],endpoints=endpoints,chunk_id=cid,quote=s['quote'],start=s['start_offset'],end=s['end_offset'],chunk_sha256=s['source_text_sha256'])
     obj=validate_span(obj,chunks,docs);relations.append(obj)
    except ValueError as err:rejected.append({'relation_id':rel['relation_id'],'span_id':sid,'reason':str(err)})
 # Re-run the bounded formatting repair against the SAME cached raw extractions.
 # Original len>=5 parser can already accept the malformed tuple. Never claim
 # the strict Stage1G 3->4 result is an original MiniRAG graph gain.
 from reproduce.graph_index_pilot.original import recover,parse
 raw=read(ROOT/'review/graph_v2_stage1g_20261110/frozen_stage1f_input.json')['extractions']
 repairs=[]
 for r in raw:
  fixed,audit=recover(r['raw'])
  if audit:repairs.append({'chunk_id':r['chunk_id'],'repairs':audit,'original_relations':len(parse(r['chunk_id'],r['raw'])['relations']),'repaired_relations':len(parse(r['chunk_id'],fixed)['relations']),'applied_to_original_index':False,'status':'NO_RETRIEVAL_EFFECT_NO_FULL_RAW_HISTORY'})
 result={'schema':'MINIRAG_RELATION_METADATA_1','relations':relations,'by_source_time':by_time,'corpus_years':sorted(years),
         'dated_mentions':dated['records'],'source_binding_rejections':rejected,'parser_recovery_audit':repairs,
         'construction':'Read existing source stores/cached source-span extraction only; zero new corpus extraction, no QA gold inputs',
         'direction_verified':0,'trust_verified':0,'model_calls':0}
 validate_sidecar(result,graph,chunks,docs)
 return result,graph,chunks


def index_arms(spec):
 from minirag.relation_metadata import ARMS
 guard(spec)
 data,graph,chunks=sidecar()
 root=OUT/'indexes';root.mkdir(exist_ok=True)
 for arm in ARMS:
  dest=root/arm
  if dest.exists():
   if sha((dest/'graph_chunk_entity_relation.graphml').read_bytes())!=spec['old_index_sha256']['graph_chunk_entity_relation.graphml']:raise ValueError('Arm graph mutated')
   if read(dest/'relation_metadata.json')!=data:raise ValueError('Sidecar differs; no overwrite/resume contamination')
   continue
  dest.mkdir()
  for n in spec['old_index_sha256']:
   if n.endswith(('.json','.graphml')):shutil.copy2(INDEX/n,dest/n)
  write(dest/'relation_metadata.json',data)
  write(dest/'arm.json',{'arm':arm,'env':{'MINIRAG_GRAPH_ARM':arm},'graph_sha256':spec['old_index_sha256']['graph_chunk_entity_relation.graphml'],'topology_preserved':True,'original_weight_preserved':True,'new_extraction_calls':0})
 write(OUT/'indexing_audit.json',{'arm_count':8,'sidecar_relations':len(data['relations']),'dated_mentions':len(data['dated_mentions']),'parser_recovery':data['parser_recovery_audit'],'source_rejections':data['source_binding_rejections'],'verified_direction':0,'verified_trust':0,'full_indexing':False,'source_store_overlay_not_model_reindex':True,'api_calls':0})
 guard(spec)
 return data,graph,chunks


def restore(data):
 result={}
 for seed,r in data['entries'].items():
  score=float(r['Score']) if r['score_dtype']=='python_float' else np.dtype(r['score_dtype']).type(r['Score'])
  result[seed]={'Score':score,'Path':list(r['Path']),'PathChunkScores':dict(r['PathChunkScores'])}
 return {k:[result[s] for s in seeds] for k,seeds in data['groups'].items()}


def pack_context(baseline,rank,chunks,costs):
 used=0;rows=[];ids=[]
 for i,cid in enumerate(rank):
  if cid not in chunks:continue
  used+=costs[cid]
  if used>4000:break
  rows.append([i,chunks[cid]['content']]);ids.append(cid)
 marker='-----Sources-----\n```csv\n'
 if marker not in baseline:raise ValueError('Unknown baseline context format')
 prefix=baseline.split(marker)[0]+marker
 f=io.StringIO();csv.writer(f).writerows([['id','content'],*rows])
 return prefix+f.getvalue()+'\n```\n',ids


def offline(spec):
 from minirag import operate as op
 from minirag.relation_metadata import ARMS,rank
 from reproduce.path.c_profiles import apply_profile
 data,graph,chunks=index_arms(spec)
 apply_profile('C3')
 inputs={r['question']:r for r in lines(FROZEN/'inputs.jsonl')}
 old={(r['question'],r['variant']):r for r in lines(FROZEN/'contexts.jsonl')}
 ref={r['question']:r for r in lines(FROZEN/'replay.jsonl')}
 enc=tiktoken.get_encoding('o200k_base');costs={c:len(enc.encode(r['content'])) for c,r in chunks.items()}
 records=[]
 for qrow in spec['questions']:
  q=qrow['Question'];row=inputs[q];base=old[(q,'C3')]
  scores=Counter();rk=op.kwd2chunk(restore(row),row['vector_ids'],30,scores_out=scores)
  expected=ref[q]['selections']['C3']['graph_top30']
  assert rk==expected, 'C3 score/rank parity failed'
  rebuilt,kept=pack_context(base['context'],rk,chunks,costs)
  assert rebuilt==base['context'] and kept==base['chunk_ids'], 'Baseline packing parity failed'
  pairs=[(k,s,row['entries'][s]['Score']) for k,seeds in row['groups'].items() for s in seeds]
  for arm in ARMS:
   started=time.perf_counter();newrank,trace=rank(q,scores,row['vector_ids'],pairs,graph,data,arm)
   ctx,ids=pack_context(base['context'],newrank,chunks,costs)
   records.append({'id':qrow['id'],'question':q,'arm':arm,'ranked_ids':newrank,'chunk_ids':ids,'context':ctx,'context_sha256':sha(ctx),
      'baseline_context_sha256':base['context_sha256'],'metadata':trace,'latency_ms':1000*(time.perf_counter()-started),
      'retrieval_kind':'Exact frozen C3 aggregation replay + native-shared metadata adapter, not fresh path generation'})
 target=OUT/'retrieval_traces.jsonl'
 if target.exists():
  previous=lines(target)
  # Timings are observations, not deterministic method inputs.
  signature=lambda rows:[{k:v for k,v in r.items() if k not in ('latency_ms','metadata')}|{'metadata':{k:v for k,v in r['metadata'].items() if k!='ms'}} for r in rows]
  if signature(previous)!=signature(records):raise ValueError('Replay differs from checkpoint')
  records=previous
 else:
  for r in records:append(target,r)
 guard(spec)
 write(OUT/'offline_validation.json',{'baseline_rank_checks':200,'baseline_context_checks':200,'arms':8,'replays':len(records),'source_unchanged':True,'model_calls':0,'new_path_scoring':False})
 return records


def metrics(spec,records,ids):
 gold={r['question']:set(r.get('evidence',[])) for r in lines(ROOT/'logs/diag_path2chunk.jsonl') if r.get('evidence')}
 # diag evidence is session strings, not necessarily chunk IDs. Use the original
 # diagnostic's exact gold chunk field from the existing screen utility instead.
 from reproduce.path.path_common import gold_map
 gold=gold_map()
 selected={spec['questions'][i]['Question'] for i in ids}
 out={}
 for arm in ('G0','B','P','TS','TE','D','TD','TDV'):
  subset=[r for r in records if r['id'] in ids and r['arm']==arm]
  eligible=[r for r in subset if r['question'] in gold and gold[r['question']]]
  out[arm]={'questions':len(subset),'gold_pairs':sum(len(set(gold[r['question']])) for r in eligible),
    'kept_gold_pairs':sum(len(set(gold[r['question']])&set(r['chunk_ids'])) for r in eligible),
    'full_evidence':sum(set(gold[r['question']])<=set(r['chunk_ids']) for r in eligible),
    'changed_contexts':sum(r['context_sha256']!=r['baseline_context_sha256'] for r in subset),
    'direction_used':sum(r['metadata']['direction_used'] for r in subset),'trust_used':sum(r['metadata']['trust_used'] for r in subset)}
 return out


async def native_check(spec,records):
 """Verify frozen aggregation replay against native unchanged path generation."""
 import os
 from minirag import MiniRAG,QueryParam
 from reproduce.path.path_common import embedding,forbidden_llm,CACHE
 from reproduce.path.c_profiles import apply_profile
 from minirag.relation_metadata import ARMS
 target=OUT/'native_validation.json'
 if target.exists() and read(target).get('complete'):return read(target)
 expected={(r['id'],r['arm']):r for r in records}
 embed=embedding();checks=[]
 for arm in ARMS:
  work=OUT/'indexes'/arm
  apply_profile('C3')
  os.environ.update(MINIRAG_GRAPH_ARM=arm,MINIRAG_KW_CACHE=str(CACHE),MINIRAG_KW_CACHE_ONLY='1',MINIRAG_CONTEXT_LOG=str(OUT/'native_traces.jsonl'))
  rag=MiniRAG(working_dir=str(work),llm_model_func=forbidden_llm,llm_model_name='offline-frozen-parser',enable_llm_cache=False,embedding_func=embed,log_level='ERROR')
  for i in spec['canary_indices']:
   q=spec['questions'][i]['Question']
   ctx=await rag.aquery(q,QueryParam(mode='mini',top_k=60,only_need_context=True))
   if ctx!=expected[(i,arm)]['context']:raise ValueError(f'Native replay context mismatch: id={i} arm={arm}')
   checks.append({'id':i,'arm':arm,'context_match':True})
   print('NATIVE',arm,i,'exact',flush=True)
 result={'complete':True,'checks':checks,'native_queries':len(checks),'model_calls':0,'original_index_unchanged':True}
 write(target,result);os.environ.pop('MINIRAG_GRAPH_ARM',None);guard(spec)
 return result
