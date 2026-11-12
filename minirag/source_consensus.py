"""Opt-in joint edge/source allocation; no semantic verification or new retrieval.

The shuffled control corrupts ONLY the consensus signal, not real source support.
An index-wide bipartite double-edge swap preserves edge and source degrees.
"""
from collections import Counter
import hashlib
import json
import random

MODES=('selection_only','joint','shuffled')
LAMBDA=0.5
SHUFFLE_SEED=20261111


def validate_mode(mode):
    if mode not in MODES:raise ValueError('Unknown source consensus mode')


def source_sets(nx_graph,valid):
    return {tuple(sorted((u,v))):set(d.get('source_id','').split('<SEP>'))&set(valid)
            for u,v,d in nx_graph.edges(data=True)}


def shuffled_sources(sources,seed=SHUFFLE_SEED):
    """Deterministic swaps, preserving every row/column incidence degree exactly."""
    rows={e:set(v) for e,v in sorted(sources.items())}
    incidences=sorted((e,c) for e,cs in rows.items() for c in cs)
    rng=random.Random(seed);swaps=0
    if len(incidences)>1:
        for _ in range(20*len(incidences)):
            i,j=rng.sample(range(len(incidences)),2)
            (e,a),(f,b)=incidences[i],incidences[j]
            if e==f or a==b or b in rows[e] or a in rows[f]:continue
            rows[e].remove(a);rows[e].add(b);rows[f].remove(b);rows[f].add(a)
            incidences[i]=(e,b);incidences[j]=(f,a);swaps+=1
    before=Counter(c for cs in sources.values() for c in cs)
    after=Counter(c for cs in rows.values() for c in cs)
    assert before==after and all(len(rows[e])==len(cs) for e,cs in sources.items())
    signature=hashlib.sha256(json.dumps([(e,sorted(cs)) for e,cs in rows.items()]).encode()).hexdigest()
    return rows,{'seed':seed,'successful_swaps':swaps,'edge_source_degrees_preserved':True,
                 'signal_sha256':signature}


def joint_distribution(path,voted,sources,signal,lam=LAMBDA):
    """Mix true edge-membership distribution with a joint-source component.

    One hop, missing consensus, and missing voted support use exact old behavior.
    Candidate sources cannot be introduced by a shuffled signal.
    """
    if not 0<=lam<1:raise ValueError('Mixture requires 0 <= lambda < 1')
    hops=list(dict.fromkeys(tuple(sorted((u,v))) for u,v in zip(path,path[1:])))
    voted=list(dict.fromkeys(tuple(sorted(e)) for e in voted))
    if len(hops)<2 or not voted or lam==0:return None
    counts=Counter(c for e in voted for c in sorted(sources[e]))
    if not counts:return None
    common=set.intersection(*(signal[e] for e in hops))&counts.keys()
    if not common:return None
    n=sum(counts.values())
    return {c:(1-lam)*v/n+(lam/len(common) if c in common else 0.) for c,v in counts.items()}


async def prepare_allocation(entries,graph,pairs,valid,mode):
    validate_mode(mode)
    if mode=='selection_only':
        return None,{'mode':mode,'lambda':0.,'eligible_unique_paths':0,'changed_unique_paths':0}
    # A private read-only cache; never mutate the storage's NetworkX graph.
    token=tuple(sorted(valid))
    cache=getattr(graph,'_source_consensus_cache',None)
    if cache is None or cache['valid']!=token:
        raw=getattr(graph,'_graph',None)
        if raw is None:raise ValueError('Consensus requires the original NetworkX source index')
        sources=source_sets(raw,valid);shuffled,meta=shuffled_sources(sources)
        cache={'valid':token,'sources':sources,'shuffled':shuffled,'shuffle_meta':meta}
        graph._source_consensus_cache=cache
    sources=cache['sources'];signal=cache['shuffled'] if mode=='shuffled' else sources
    paths={p for entry in entries.values() for p in entry['Path']}
    output={};eligible=0
    for path in sorted(paths):
        if len(path)>=3 and pairs.get(path):eligible+=1
        dist=joint_distribution(path,pairs.get(path,()),sources,signal)
        if dist is not None:output[path]=dist
    return output,{'mode':mode,'lambda':LAMBDA,'eligible_unique_paths':eligible,
        'changed_unique_paths':len(output),'node_channel_unchanged':True,
        'path_membership_unchanged':True,'edge_budget_preserved':True,
        'true_candidate_support_unchanged':True,
        'shuffle':cache['shuffle_meta'] if mode=='shuffled' else None}
