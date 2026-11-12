"""Source-bound dated mentions over a frozen graph; not asserted event facts."""
from collections import Counter, defaultdict
from datetime import date, timedelta
from functools import lru_cache
import hashlib
import math
import re
import time

from . import temporal
from .prompt import GRAPH_FIELD_SEP

VERSION = '20261110-source-event-time'
MODES = ('full', 'no_graph', 'no_source_binding', 'no_event_time')
STOP = set(('a an the is was were be been being do did does what which who whom whose '
            'when where why how and or of to for from with by as at in on it its this that '
            'these those he she his her him they their them i me my you your we our us '
            'has have had can could would should will shall any some about through '
            'conversation mentions mention say said tell told ask asked asks question '
            'before after between first last earliest latest time date day days year '
            'today tomorrow yesterday tonight during within new').split())
STOP.update(temporal.MONTHS)
STOP.update(name[:3] for name in temporal.MONTHS)
UTTERANCE = re.compile(r'^(?P<speaker>[^:\n]{1,90}):[ \t]*(?P<body>[^\n]+)', re.M)
RELATIVE = re.compile(r'\b(?:the\s+)?day\s+after\s+tomorrow\b|'
                      r'\b(?:the\s+)?day\s+before\s+yesterday\b|'
                      r'\btomorrow\b|\byesterday\b|\btoday\b|\btonight\b', re.I)


def compact(s):
    return ''.join(c for c in str(s).casefold() if c.isalnum())


def tokens(s):
    out = set()
    for token in re.findall(r'[a-z]+', s.casefold()):
        if token in STOP or len(token) < 3:
            continue
        if token.endswith('ing') and len(token) > 5:
            token = token[:-3]
        elif token.endswith('ed') and len(token) > 4:
            token = token[:-2]
        elif token.endswith('s') and len(token) > 4:
            token = token[:-1]
        out.add(token)
    return out


@lru_cache(maxsize=None)
def name_pattern(name):
    parts = re.findall(r'\w+', name.strip('"').casefold())
    if not parts or len(''.join(parts)) < 3:
        return None
    return re.compile(r'(?<!\w)' + r'[\s_-]*'.join(map(re.escape, parts)) + r'(?!\w)')


def literal(name, text):
    pattern = name_pattern(name)
    return pattern is not None and pattern.search(text.casefold()) is not None


def mentioned_days(text, source_day):
    result = {date.fromisoformat(a.removeprefix('time:day:')): 'absolute'
              for a in temporal.parse_query(text, {source_day.year}) if a.startswith('time:day:')}
    for match in RELATIVE.finditer(text):
        phrase = match.group().casefold()
        delta = 2 if 'after' in phrase else -2 if 'before' in phrase else (
            1 if 'tomorrow' in phrase else -1 if 'yesterday' in phrase else 0)
        result[source_day + timedelta(days=delta)] = phrase
    return result


def build_overlay(docs, chunks, graph):
    """Quote spans are exact; edges are provenance/participants, not entailment."""
    started = time.perf_counter()
    docs_time = {}
    for did, row in docs.items():
        match = re.match(r'^Time:\s*(\d{4})(\d{2})(\d{2})([^\n]*)', row.get('content', ''))
        if match:
            try:
                docs_time[did] = (date(*map(int, match.groups()[:3])), match.group().removeprefix('Time:').strip())
            except ValueError:
                pass
    by_name = defaultdict(set)
    for name in graph:
        by_name[compact(name)].add(name)
    source_edges = defaultdict(set)
    for u, v, data in graph.edges(data=True):
        for cid in str(data.get('source_id', '')).split(GRAPH_FIELD_SEP):
            if cid in chunks:
                source_edges[cid].add(tuple(sorted((u, v))))
    records = []
    ambiguous = 0
    for cid, chunk in sorted(chunks.items()):
        stamp = docs_time.get(chunk.get('full_doc_id'))
        if stamp is None:
            continue
        source_day, source_stamp = stamp
        text = chunk['content']
        utterances = [m for m in UTTERANCE.finditer(text) if m['speaker'].strip().casefold() != 'time']
        participants = set().union(*(by_name.get(compact(m['speaker']), set()) for m in utterances)) if utterances else set()
        for m in utterances:
            days = mentioned_days(m['body'], source_day)
            if not days:
                continue
            # Multiple distinct dates in one utterance need finer event binding.
            if len(days) != 1:
                ambiguous += 1
                continue
            speakers = by_name.get(compact(m['speaker']), set())
            adjacent = {tuple(sorted((s, n))) for s in speakers for n in graph.neighbors(s)}
            present = participants | {n for edge in adjacent for n in edge
                                      if n not in participants and literal(n, m['body'])}
            loose = sorted(edge for edge in adjacent if set(edge) <= present)
            bound = sorted(set(loose) & source_edges[cid])
            for event_day, cue in days.items():
                key = f'{cid}:{m.start("body")}:{m.end("body")}:{event_day.isoformat()}'
                records.append({'occurrence_id': hashlib.sha256(key.encode()).hexdigest(),
                                'chunk': cid, 'speaker': m['speaker'],
                                'source_day': source_day.isoformat(), 'source_timestamp': source_stamp,
                                'mentioned_day': event_day.isoformat(), 'date_cue': cue,
                                'quote': m['body'], 'start': m.start('body'), 'end': m.end('body'),
                                'bound_edges': [list(e) for e in bound],
                                'loose_edges': [list(e) for e in loose],
                                'status': 'dated_mention; occurrence/predicate entailment unverified'})
    return {'records': records, 'ambiguous_utterances_skipped': ambiguous,
            'construction_ms': 1000 * (time.perf_counter() - started)}


def best_anchors(groups):
    result = {}
    for keyword, matches in groups.items():
        if matches:
            seed, score = max(matches, key=lambda x: float(x[1]))
            result[keyword] = (seed, max(0., float(score)))
    return result


def doc_scores(base, by_time, anchors, h, vector_ids):
    """Exact Tai score addition, exposing scores without modifying Tai's module."""
    scores = Counter(base)
    _, lists = temporal.rank_temporal(base, by_time, anchors, h, vector_ids,
                                      limit=len(base) + sum(len(by_time.get(a, ())) for a in anchors))
    for ids in lists.values():
        if ids:
            boost = 2 * (10 if ids[0] in vector_ids else 1)
            for cid in ids:
                scores[cid] += boost * h.get(cid, 1.)
    return scores


def rank(base, by_time, anchors, h, vector_ids, overlay, groups, graph, query, *, mode='full', limit=30):
    if mode not in MODES:
        raise ValueError('Unknown source event-time mode')
    started = time.perf_counter()
    scores = doc_scores(base, by_time, anchors, h, vector_ids)
    initial = [c for c, _ in scores.most_common(limit)]
    trace = {'version': VERSION, 'mode': mode, 'query_anchors': list(anchors),
             'event_candidates': [], 'selected_occurrences': [], 'anchor_budgets': {},
             'tai_top30': initial, 'original_C0_top30': [c for c, _ in base.most_common(limit)]}
    best = best_anchors(groups)
    person_words = set()
    for keyword, (seed, score) in best.items():
        if 'person' in str(graph.nodes[seed].get('entity_type', '')).casefold():
            person_words |= tokens(seed) | tokens(keyword) | {compact(seed), compact(keyword)}
    query_words = tokens(query) - person_words
    if mode == 'no_event_time' or not anchors or not query_words:
        trace['selection_ms'] = 1000 * (time.perf_counter() - started)
        return initial, trace
    for anchor in anchors:
        candidates, all_support = {}, {}
        for record in overlay['records']:
            if record['mentioned_day'] == record['source_day']:
                continue
            day = record['mentioned_day']
            if (anchor.startswith('time:day:') and day != anchor.removeprefix('time:day:')
                    or anchor.startswith('time:month:') and day[:7] != anchor.removeprefix('time:month:')):
                continue
            lexical = len(query_words & tokens(record['quote'])) / len(query_words)
            if lexical <= 0:
                continue
            linked = {n for e in record['loose_edges' if mode == 'no_source_binding' else 'bound_edges'] for n in e}
            g = 0. if mode == 'no_graph' else math.fsum(score for seed, score in best.values() if seed in linked)
            raw = lexical * (1. + g)
            cid = record['chunk']
            item = {'occurrence_id': record['occurrence_id'], 'chunk': cid,
                    'lexical': lexical, 'graph_support': g, 'raw': raw}
            if cid not in candidates or raw > candidates[cid]['raw']:
                candidates[cid] = item
            all_support[cid] = max(all_support.get(cid, 0.), lexical)
        if not candidates:
            continue
        # Identical exposure and budget across all event-graph ablations.
        for cid in sorted(candidates):
            scores.setdefault(cid, 0.)
        independent = min(candidates, key=lambda c: (-all_support[c], -base.get(c, 0.), c))
        budget = 2 * (10 if independent in vector_ids else 1)
        selected = sorted(candidates, key=lambda c: (-candidates[c]['raw'], -base.get(c, 0.), c))[:3]
        total = math.fsum(candidates[c]['raw'] for c in selected)
        for cid in selected:
            scores[cid] += budget * candidates[cid]['raw'] / total
        trace['anchor_budgets'][anchor] = budget
        trace['event_candidates'].extend(sorted(candidates))
        trace['selected_occurrences'].extend(candidates[c] for c in selected)
    trace['event_candidates'] = sorted(set(trace['event_candidates']))
    trace['selection_ms'] = 1000 * (time.perf_counter() - started)
    return [cid for cid, _ in scores.most_common(limit)], trace


_CACHE = {}


def loaded_overlay(chunks_db, graph):
    import json
    from pathlib import Path
    key = (id(chunks_db), id(graph))
    if key not in _CACHE:
        docs = json.loads(Path(chunks_db._file_name).with_name('kv_store_full_docs.json').read_text())
        _CACHE[key] = build_overlay(docs, chunks_db._data, graph)
    return _CACHE[key]
