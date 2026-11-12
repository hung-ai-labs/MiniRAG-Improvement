"""Shared complete path provenance projection, independent of the scorer."""
import asyncio
from collections import defaultdict
import math
import time
from .prompt import GRAPH_FIELD_SEP
from .utils import calculate_similarity, split_string_by_multi_markers

MODES = frozenset(("full", "local", "unbounded"))

def _split(value):
    return split_string_by_multi_markers(value or "", [GRAPH_FIELD_SEP])

def _sources(node, *, tail=False, query=None, valid=None):
    ids = _split((node or {}).get("source_id", ""))
    descriptions = _split((node or {}).get("description", ""))
    if tail and query is not None and len(ids) == len(descriptions) and len(ids) > 5:
        selected = calculate_similarity(descriptions, query, k=int(max(5, len(ids)/2)))
        ids = [ids[i] for i in selected]
    return tuple(dict.fromkeys(c for c in ids if valid is None or c in valid))

async def project_paths(entries, graph, pairs, query, max_chunks, mode, *, valid=None, trace=None, path_weights=None, source_affinity=None, edge_source_weights=None, path_edge_allocation=None):
    """Project original paths; optional weights change scores, never path membership."""
    if mode not in MODES:
        raise ValueError(f"Unknown P2 provenance mode: {mode!r}")
    if path_edge_allocation is not None and (edge_source_weights is not None or source_affinity is not None):
        raise ValueError('Joint path allocation cannot be combined with other source reweighting')
    started=time.perf_counter()
    names=sorted(set(entries) | {n for e in entries.values() for p in e["Path"] for n in p})
    nodes=dict(zip(names,await asyncio.gather(*(graph.get_node(n) for n in names))))
    seed_sources={n:_sources(d,valid=valid) for n,d in nodes.items()}
    tail_sources={n:_sources(d,tail=True,query=query,valid=valid) for n,d in nodes.items()}
    keys=sorted({tuple(sorted(pair)) for e in entries.values() for p in e["Path"] for pair in pairs.get(p,[])})
    edge_data=await asyncio.gather(*(graph.get_edge(*key) for key in keys))
    if any(d is None for d in edge_data):
        raise ValueError("Voted P2 edge has no physical graph edge")
    edge_sources={key:_sources(d,valid=valid) for key,d in zip(keys,edge_data)}
    if edge_source_weights is not None:
        if set(edge_source_weights)-set(edge_sources):
            raise ValueError('Research allocation refers to a non-voted edge')
        for key, weights in edge_source_weights.items():
            if set(weights)!=set(edge_sources[key]) or any(not math.isfinite(v) or v<0 for v in weights.values()):
                raise ValueError('Research allocation must retain every original source')
            if not math.isclose(math.fsum(weights.values()),len(weights),rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('Research allocation must preserve each edge source budget')
    diagnostics={"mode":mode,"node_loads":len(names),"edge_loads":len(keys),"paths_projected":0,
                 "node_mass":0.0,"edge_mass":0.0,"candidate_chunk_ids":[],"isolated_fallbacks":0}
    candidates=set()
    for seed,entry in entries.items():
        paths=entry["Path"]
        if path_weights is None:
            raw={p:float(s[0])+float(s[1])+1.0 for p,s in paths.items()}
        else:
            if set(path_weights[seed]) != set(paths):
                raise ValueError('Reweighting must retain every original path')
            raw={p:float(path_weights[seed][p]) for p in paths}
        if any(not math.isfinite(v) or v < 0 or (path_weights is None and v == 0) for v in raw.values()):
            raise ValueError('Path scores must be finite and nonnegative')
        if paths and math.fsum(raw.values()) <= 0:
            raise ValueError('A nonempty seed requires positive total path mass')
        total=math.fsum(raw.values()) if mode != "unbounded" else 1.0
        node_coeff,edge_coeff=defaultdict(list),defaultdict(list)
        joint_coeff=defaultdict(list)
        for path,weight in raw.items():
            mass=weight/total
            node_keys=[(path[0],False)]+[(n,True) for n in path[1:]]
            edge_keys=list(dict.fromkeys(tuple(sorted(pair)) for pair in pairs.get(path,[])))
            n_node=sum(len(tail_sources[n] if tail else seed_sources[n]) for n,tail in node_keys)
            n_edge=sum(len(edge_sources[key]) for key in edge_keys)
            channels=int(n_node>0)+int(n_edge>0)
            if not channels:
                continue
            if n_node:
                coefficient=mass/channels/n_node
                for key in node_keys:
                    node_coeff[key].append(coefficient)
                diagnostics["node_mass"]+=mass/channels
            if n_edge:
                allocation=(path_edge_allocation or {}).get(path)
                if allocation is None:
                    coefficient=mass/channels/n_edge
                    for key in edge_keys:
                        edge_coeff[key].append(coefficient)
                else:
                    support={c for key in edge_keys for c in edge_sources[key]}
                    if (set(allocation)!=support or any(not math.isfinite(v) or v<=0 for v in allocation.values())
                            or not math.isclose(math.fsum(allocation.values()),1.,rel_tol=1e-10,abs_tol=1e-10)):
                        raise ValueError('Joint path allocation must preserve positive support and unit edge budget')
                    for cid,value in allocation.items():
                        joint_coeff[cid].append(mass/channels*value)
                diagnostics["edge_mass"]+=mass/channels
            diagnostics["paths_projected"]+=1
        contributions=defaultdict(list)
        for cid,values in joint_coeff.items():
            contributions[cid].extend(values)
        for (name,tail),values in node_coeff.items():
            coefficient=math.fsum(values)
            for cid in tail_sources[name] if tail else seed_sources[name]:
                factor = source_affinity('node', name, cid) if source_affinity else 1.
                if not math.isfinite(factor) or not 0 <= factor <= 1: raise ValueError('Invalid source affinity')
                contributions[cid].append(coefficient * factor)
        for key,values in edge_coeff.items():
            coefficient=math.fsum(values)
            for cid in edge_sources[key]:
                factor = source_affinity('edge', key, cid) if source_affinity else 1.
                if not math.isfinite(factor) or not 0 <= factor <= 1: raise ValueError('Invalid source affinity')
                if edge_source_weights is not None and key in edge_source_weights:
                    factor *= edge_source_weights[key][cid]
                contributions[cid].append(coefficient * factor)
        if not paths:
            ids=seed_sources[seed]
            for cid in ids:
                factor = source_affinity('node', seed, cid) if source_affinity else 1.
                if not math.isfinite(factor) or not 0 <= factor <= 1: raise ValueError('Invalid source affinity')
                contributions[cid].append(factor/len(ids))
            diagnostics["isolated_fallbacks"]+=1
            diagnostics["node_mass"]+=bool(ids)
        scores={cid:math.fsum(values) for cid,values in contributions.items()}
        if mode != "unbounded" and math.fsum(scores.values()) > 1+1e-9:
            raise ArithmeticError("P2 seed path budget exceeded one")
        candidates.update(scores)
        ranked=sorted(scores,key=lambda c:(-round(scores[c],12),c))[:max_chunks]
        entry["Path"]=ranked
        entry["PathChunkScores"]={c:float(scores[c]) for c in ranked}
    diagnostics["candidate_chunk_ids"]=sorted(candidates)
    diagnostics["projection_ms"]=1000*(time.perf_counter()-started)
    if trace is not None:
        trace.update(diagnostics)
    return entries
