"""Opt-in metadata adapters over the original undirected MiniRAG graph.

No new graph, extraction model, predicate truth, path generator or path formula.
Exact spans are candidate provenance, not semantic verification. UNKNOWN always
retains the original route. Used both by the offline experiment and native mini.
"""
from collections import Counter
import hashlib
import hmac
import json
import os
import re
import time
from pathlib import Path

from . import temporal

ARMS = {
    'G0': (False, False, False, False, False, False),
    'B': (True, False, False, False, False, False),
    'P': (True, True, False, False, False, False),
    'TS': (True, True, True, False, False, False),
    'TE': (True, True, True, True, False, False),
    'D': (True, True, False, False, True, False),
    'TD': (True, True, True, True, True, False),
    'TDV': (True, True, True, True, True, True),
}
FEATURE_NAMES = ('parser_recovery', 'provenance', 'source_time', 'event_time', 'direction', 'trust')
STOP = set('a an the is was were be been being do did does what which who whom whose when where why how and or of to for from with by as at in on it its this that these those he she his her him they their them i me my you your we our us has have had can could would should will shall any some about through conversation mentions mention say said tell told ask asked asks question before after between first last earliest latest time date day days year today tomorrow yesterday tonight during within new'.split())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def review_binding(record):
    fields=('relation_id','endpoints','chunk_id','quote','start','end','chunk_sha256','doc_start','doc_end','document_sha256')
    return digest(json.dumps({k:record[k] for k in fields},sort_keys=True,ensure_ascii=False,separators=(',',':')))


def effective_reviews(sidecar, trusted_keys=None):
    """Only externally signed, source-bound manual reviews can activate D/trust.

    Never generate a review/key or promote an extractor's confidence. Candidate
    records remain UNKNOWN. Missing, stale or invalid review material abstains.
    """
    if not sidecar.get('reviews'):
        return []
    if trusted_keys is None:
        try:trusted_keys=json.loads(os.environ.get('MINIRAG_GRAPH_REVIEW_KEYS','{}'))
        except ValueError:return []
    if not isinstance(trusted_keys,dict):return []
    records={review_binding(r):r for r in sidecar['relations']}
    result=[]
    for review in sidecar['reviews']:
        if not isinstance(review,dict) or review.get('origin')!='EXTERNAL_MANUAL_REVIEW':continue
        row=records.get(review.get('assertion_sha256'))
        key=trusted_keys.get(review.get('reviewer'))
        if isinstance(key,str):key=key.encode()
        if row is None or not isinstance(key,bytes) or len(key)<16:continue
        unsigned={k:v for k,v in review.items() if k!='signature'}
        signature=hmac.new(key,json.dumps(unsigned,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode(),hashlib.sha256).hexdigest()
        supplied=review.get('signature','')
        if not isinstance(supplied,str) or not re.fullmatch(r'[0-9a-f]{64}',supplied) or not hmac.compare_digest(signature,supplied):continue
        if review.get('verdict') not in ('SUPPORTED','REJECTED') or not review.get('rationale'):continue
        if review.get('direction')=='DIRECTED' and sorted([review.get('actor',''),review.get('target','')])!=sorted(row['endpoints']):continue
        if review.get('direction') not in ('DIRECTED','UNDIRECTED','UNKNOWN'):continue
        result.append((row,review))
    return result


def reviewed_order(query, ids, sidecar, *, direction=False, trust=False, trusted_keys=None):
    """Stable conditional preference; unknown evidence is retained unchanged."""
    tags=Counter();bad=set();supported=set()
    reviews=effective_reviews(sidecar,trusted_keys)
    text=' '.join(query.casefold().split())
    for row,review in reviews:
        phrase=str(review.get('predicate_surface','')).strip().casefold()
        if not phrase or not re.search(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)',text):continue
        cid=row['chunk_id']
        if cid not in ids:continue
        if review['verdict']=='SUPPORTED':supported.add(cid)
        elif trust:bad.add(cid)
        if direction and review['verdict']=='SUPPORTED' and review.get('direction')=='DIRECTED':
            actor=' '.join(review['actor'].strip('"').casefold().split())
            target=' '.join(review['target'].strip('"').casefold().split())
            forward=r'\bfrom\s+'+re.escape(actor)+r'\s+to\s+'+re.escape(target)+r'\b'
            reverse=r'\bfrom\s+'+re.escape(target)+r'\s+to\s+'+re.escape(actor)+r'\b'
            if re.search(forward,text):tags[cid]+=1
            elif re.search(reverse,text):tags[cid]-=1
    result=list(ids)
    if tags:result=sorted(result,key=lambda c:-tags[c])
    # Rejecting one relation does not invalidate a chunk with other supported
    # query-matching facts. No UNKNOWN source is discarded or marked supported.
    rejected_only=bad-supported
    if trust and rejected_only:result=sorted(result,key=lambda c:c in rejected_only)
    return result,dict(direction_used=sum(v!=0 for v in tags.values()),trust_used=len(rejected_only),verified_reviews=len(reviews))


def terms(text):
    return {w[:-1] if len(w) > 4 and w.endswith('s') else w
            for w in re.findall(r'[a-z]+', text.casefold()) if w not in STOP and len(w) >= 3}


def validate_span(span, chunks, docs):
    cid = span['chunk_id']
    chunk = chunks[cid]
    quote = span['quote']
    start, end = span['start'], span['end']
    if not quote or not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or start < 0 or end <= start:
        raise ValueError('Invalid evidence offsets')
    if chunk['content'][start:end] != quote:
        raise ValueError('Evidence does not match chunk')
    document = docs[chunk['full_doc_id']]['content']
    # Continuation chunks may be substrings; do not assume a zero document offset.
    if document.count(chunk['content']) != 1:
        raise ValueError('Chunk location in full document is ambiguous')
    base = document.index(chunk['content'])
    if document[base+start:base+end] != quote:
        raise ValueError('Evidence does not match full document')
    if digest(chunk['content']) != span['chunk_sha256']:
        raise ValueError('Evidence source hash stale')
    return {**span, 'doc_start': base+start, 'doc_end': base+end,
            'document_sha256': digest(document), 'source_status': 'EXACT_SOURCE_VALID',
            'semantic_status': 'UNKNOWN', 'direction_status': 'UNKNOWN', 'trust_status': 'UNKNOWN'}


def validate_sidecar(sidecar, graph, chunks, docs):
    if sidecar.get('schema') != 'MINIRAG_RELATION_METADATA_1' or graph.is_directed() or graph.is_multigraph():
        raise ValueError('Sidecar requires original simple nx.Graph')
    for record in sidecar['relations']:
        u, v = record['endpoints']
        if not graph.has_edge(u, v):
            raise ValueError('Metadata cannot create a physical edge')
        if record['chunk_id'] not in str(graph.edges[u, v].get('source_id', '')).split('<SEP>'):
            raise ValueError('Relation evidence lacks original edge source membership')
        checked = validate_span(record, chunks, docs)
        for k in ('doc_start', 'doc_end', 'document_sha256'):
            if record.get(k) != checked[k]:
                raise ValueError('Document binding stale')
        if record.get('semantic_status') != 'UNKNOWN' or record.get('direction_status') != 'UNKNOWN' or record.get('trust_status') != 'UNKNOWN':
            raise ValueError('This candidate-only index has no external semantic reviews')
    return True


def rank(query, base_scores, vector_ids, pairs, graph, sidecar, arm, *, limit=30, trusted_keys=None, feature_flags=None):
    """Conditional chunk reranking; the original path scoring is unchanged.

    P uses a bounded [1,2] source-span lexical support multiplier only for an
    existing seed-adjacent physical relation; this is a soft heuristic, not truth.
    TS reuses Tai's frozen temporal adapter constants. TE uses explicit mentioned
    dates, not document dates. D/trust abstain without externally reviewed data.
    """
    if arm not in ARMS:
        raise ValueError('Unknown graph arm')
    started = time.perf_counter()
    if feature_flags is not None:
        if set(feature_flags)!=set(FEATURE_NAMES) or any(type(v) is not bool for v in feature_flags.values()):
            raise ValueError('Feature ablation must specify all six boolean flags')
        selected=tuple(feature_flags[k] for k in FEATURE_NAMES)
    else:selected=ARMS[arm]
    recovery, provenance, source_time, event_time, direction, trust = selected
    original = [c for c, _ in Counter(base_scores).most_common(limit)]
    scores = Counter(base_scores)
    trace = dict(arm=arm, changes=[], direction_used=0, trust_used=0,
        provenance_used=0, event_time_used=0, source_time_used=0,
        predicate_verification='UNKNOWN', path_scoring_changed=False)
    if not any((provenance,source_time,event_time,direction,trust)):
        trace.update(status='NO_RETRIEVAL_EFFECT', ms=1000*(time.perf_counter()-started))
        return original, trace
    keywords = terms(query)
    active_seeds = {seed for _, seed, score in pairs if float(score) > 0}
    support = {}
    if provenance and keywords:
        for r in sidecar['relations']:
            if r['chunk_id'] not in scores or not active_seeds.intersection(r['endpoints']):
                continue
            overlap = len(keywords & terms(r['quote'])) / len(keywords)
            if overlap > 0:
                support[r['chunk_id']] = max(support.get(r['chunk_id'], 0), overlap)
        for cid, value in support.items():
            scores[cid] *= 1 + value
        trace['provenance_used'] = len(support)
    by_time = sidecar['by_source_time']
    anchors = temporal.parse_query(query, set(sidecar['corpus_years'])) if source_time else ()
    result = [c for c, _ in scores.most_common(limit)]
    if anchors:
        h = temporal.entity_h(graph, pairs)
        result, selected = temporal.rank_temporal(scores, by_time, anchors, h, vector_ids, limit=limit)
        trace.update(source_time_used=sum(map(len, selected.values())), anchors=list(anchors))
        if event_time:
            # No event pipeline: reuse source-bound dated mentions and add only
            # a uniquely resolved off-source date with query lexical support.
            added = Counter(scores)
            for ids in selected.values():
                if ids:
                    boost = 2 * (10 if ids[0] in set(vector_ids) else 1)
                    for cid in ids:
                        added[cid] += boost*h.get(cid, 1.)
            candidates = {}
            for r in sidecar['dated_mentions']:
                if r['mentioned_day'] == r['source_day'] or not keywords:
                    continue
                if not any(a == 'time:day:'+r['mentioned_day'] or a == 'time:month:'+r['mentioned_day'][:7] for a in anchors):
                    continue
                overlap = len(keywords & terms(r['quote'])) / len(keywords)
                if overlap > 0 and r['bound_edges']:
                    candidates[r['chunk']] = max(candidates.get(r['chunk'], 0), overlap)
            # Same bounded top-3 and source-adapter boost; no parameter search.
            for cid, value in sorted(candidates.items(), key=lambda x:(-x[1], x[0]))[:3]:
                added[cid] += 2*h.get(cid, 1.)*value
            trace['event_time_used'] = min(3, len(candidates))
            result = [c for c, _ in added.most_common(limit)]
    if direction or trust:
        result,review_trace=reviewed_order(query,result,sidecar,direction=direction,trust=trust,trusted_keys=trusted_keys)
        trace['direction_used']=review_trace['direction_used']
        trace['trust_used']=review_trace['trust_used']
    # Current experiment has no external reviews; these fallbacks remain exact.
    if direction and not trace['direction_used']:
        trace['direction_fallback'] = 'NO_VERIFIED_DIRECTION'
    if trust and not trace['trust_used']:
        trace['trust_fallback'] = 'NO_EXTERNAL_SEMANTIC_REVIEWS'
    trace.update(status='CHANGED' if result != original else 'NO_RETRIEVAL_EFFECT',
                 ms=1000*(time.perf_counter()-started))
    return result, trace


def load_for_graph(graph_storage):
    path = Path(graph_storage.global_config['working_dir']) / 'relation_metadata.json'
    sidecar=json.loads(path.read_text(encoding='utf-8'))
    if sidecar.get('reviews'):
        # External reviews cannot make a stale source trustworthy.
        chunks=json.loads(path.with_name('kv_store_text_chunks.json').read_text())
        docs=json.loads(path.with_name('kv_store_full_docs.json').read_text())
        validate_sidecar(sidecar,graph_storage._graph,chunks,docs)
    return sidecar
