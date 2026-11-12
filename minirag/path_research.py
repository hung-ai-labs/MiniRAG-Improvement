"""Two OFF-by-default, source-specific path-flow experiments over original nx.Graph.

R1 redistributes a voted edge's unchanged budget only for source spans whose
query cosine exceeds the merged relation cosine. Unseen sources are retained.
R2 adds a soft explicit-date consistency vote; no UNKNOWN direction is enforced.
Neither mode changes path membership, graph topology or local candidate IDs.
"""
from collections import defaultdict
from datetime import date
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from . import temporal
from .relation_metadata import terms

MODES=('R1','R2')


def evidence_text(record, revision=1):
    if revision==1:
        return record['quote']
    # Same endpoint context as the merged-relation embedding, without asserting
    # actor/target orientation or adding any inferred predicate.
    participants=' | '.join(str(v).strip('"') for v in sorted(record['endpoints']))
    return 'Participants: '+participants+'\n'+record.get('speaker','')+': '+record['quote']


def temporal_path_weights(weights, supported_edges):
    """Soft time support is normalized between paths, not away on a one-source edge."""
    out={};changed=0
    for seed,paths in weights.items():
        out[seed]={}
        for path,weight in paths.items():
            edges=[tuple(sorted((u,v))) for u,v in zip(path,path[1:])]
            fraction=sum(e in supported_edges for e in edges)/len(edges) if edges else 0.
            out[seed][path]=weight*(1+fraction)
            changed+=fraction>0 and weight>0
    return out,changed


def balanced_weights(sources, evidence_lifts, time_support=None):
    """Mean-one source allocation; retain complete support, never mint edge mass."""
    if len(sources)!=len(set(sources)):
        raise ValueError('Sources must be distinct')
    time_support=time_support or {}
    if (set(evidence_lifts)|set(time_support))-set(sources):
        raise ValueError('Evidence outside edge source support')
    values={}
    for cid in sources:
        lift=float(evidence_lifts.get(cid,0));t=float(time_support.get(cid,0))
        if not math.isfinite(lift) or not 0<=lift<=1 or t not in (0.,1.):
            raise ValueError('Invalid source-specific evidence/date signal')
        values[cid]=1+lift+t
    mean=math.fsum(values.values())/len(values) if values else 1
    return {cid:v/mean for cid,v in values.items()}


def date_constraint(query,years):
    anchors=temporal.parse_query(query,years)
    days=[a.removeprefix('time:day:') for a in anchors if a.startswith('time:day:')]
    if len(days)!=1:
        return None
    # Restrict to a directly adjacent, unambiguous date condition. A chronology
    # question 'Did A before B?' is not itself a before-date filter.
    m=re.search(r'\b(on|before|after)\s+(\d{8}|\d{4}[-/]\d{2}[-/]\d{2})\b',query,re.I)
    if not m:
        return None
    if re.search(r'\b(before|after|between)\b',query[:m.start()]+query[m.end():],re.I):
        return None
    return m[1].lower(),days[0]


def date_matches(day,constraint):
    op,anchor=constraint
    return day==anchor if op=='on' else day<anchor if op=='before' else day>anchor


async def prepare(mode,query,graph,pairs,relation_scores,valid,embedding_func):
    if mode not in MODES:
        raise ValueError('Unknown path research mode')
    config=getattr(graph,'global_config',{})
    path=Path(config.get('working_dir',''))/'relation_metadata.json'
    if not path.is_file():
        raise ValueError('Research path mode requires a frozen source-validated sidecar')
    sidecar=json.loads(path.read_text())
    keys={tuple(sorted(pair)) for p in pairs.values() for pair in p}
    eligible=[r for r in sidecar['relations'] if tuple(sorted(r['endpoints'])) in keys and r['chunk_id'] in valid]
    lifts=defaultdict(dict);support=defaultdict(dict)
    revision=int(os.environ.get('MINIRAG_PATH_RESEARCH_REVISION','1'))
    if revision not in (1,2):raise ValueError('Unknown path research revision')
    quotes=list(dict.fromkeys(evidence_text(r,revision) for r in eligible))
    similarities={}
    if quotes:
        vectors=np.asarray(await embedding_func([query,*quotes]),dtype=np.float64)
        if vectors.shape!=(len(quotes)+1,384) or not np.isfinite(vectors).all():
            raise ValueError('Expected finite original 384d local embeddings')
        norms=np.linalg.norm(vectors,axis=1)
        if np.any(norms==0):raise ValueError('Zero evidence/query vector')
        similarities={q:max(0.,min(1.,float(np.dot(vectors[0],v)/(norms[0]*n)))) for q,v,n in zip(quotes,vectors[1:],norms[1:])}
    constraint=date_constraint(query,set(sidecar['corpus_years'])) if mode=='R2' else None
    keywords=terms(query)
    for r in eligible:
        key=tuple(sorted(r['endpoints']));cid=r['chunk_id']
        if key not in relation_scores:raise ValueError('Missing physical path relation cosine')
        # Missing evidence does not count as negative evidence. Only a positive
        # residual above the already-existing C3 relation relevance receives mass.
        lift=max(0.,similarities[evidence_text(r,revision)]-max(0.,float(relation_scores[key])))
        lifts[key][cid]=max(lifts[key].get(cid,0),lift)
    if constraint and keywords:
        # Time-source metadata has its own source/physical-edge binding. It must
        # not accidentally depend on whether a separate semantic quote alignment
        # exists on that edge. This is a SOFT dated-mention signal, not entailment.
        for mention in sidecar['dated_mentions']:
            cid=mention['chunk']
            if cid not in valid or not keywords.intersection(terms(mention['quote'])):
                continue
            if not date_matches(mention['mentioned_day'],constraint):
                continue
            for edge in mention['bound_edges']:
                key=tuple(sorted(edge))
                if key in keys:
                    support[key][cid]=1.
    output={}
    for key in set(lifts)|set(support):
        row=await graph.get_edge(*key)
        sources=list(dict.fromkeys(c for c in row.get('source_id','').split('<SEP>') if c in valid))
        weights=balanced_weights(sources,lifts[key],support[key])
        if any(abs(v-1)>1e-12 for v in weights.values()):output[key]=weights
    return output,{'mode':mode,'revision':revision,'bound_spans':len(eligible),'quote_embeddings':len(quotes),
        'redistributed_edges':len(output),'positive_evidence_lifts':sum(v>0 for d in lifts.values() for v in d.values()),
        'temporal_supports':sum(v>0 for d in support.values() for v in d.values()),
        'temporal_edge_keys':[list(k) for k in sorted(support) if any(support[k].values())],
        'verified_direction_constraints':0,'direction_fallback':'UNKNOWN retained',
        'path_membership_changed':False,'edge_budget_preserved':True}
