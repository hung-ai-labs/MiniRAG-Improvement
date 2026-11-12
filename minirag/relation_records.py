"""Opt-in, immutable source-bound records captured BEFORE graph edge merging.

An extraction record is not necessarily one semantic occurrence. The original
model may summarize several actions. No predicate/direction is verified here.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def capture_relations(directory, chunk_id, chunk, relations):
    """Idempotent per-source extraction snapshot; no mutation or graph upsert."""
    content = chunk['content']
    rows = {}
    for ordinal, relation in enumerate(relations):
        if relation['source_id'] != chunk_id:
            raise ValueError('Relation source differs from extraction chunk')
        original = {k: relation[k] for k in ('src_id', 'tgt_id', 'description', 'keywords', 'weight', 'source_id')}
        rid = 'rr:' + digest(canonical([chunk_id, digest(content), original]))
        if rid in rows:
            rows[rid]['extraction_ordinals'].append(ordinal)
            continue
        rows[rid] = dict(record_id=rid, record_kind='RELATION_EXTRACTION_RECORD',
            source_chunk_id=chunk_id, source_document_id=chunk.get('full_doc_id'),
            source_text_sha256=digest(content), original_relation=original,
            extraction_ordinals=[ordinal], source_status='BOUND_TO_EXTRACTION_INPUT',
            semantic_status='UNKNOWN', direction_status='MODEL_OUTPUT_UNVERIFIED',
            quote=None, start=None, end=None)
    payload = {'schema': 'MINIRAG_SOURCE_RELATIONS_1', 'source_chunk_id': chunk_id,
        'source_document_id': chunk.get('full_doc_id'), 'source_text_sha256': digest(content),
        'record_count': len(rows), 'records': list(rows.values())}
    # Different extraction outputs are separate immutable snapshots, not silent
    # replacement of a prior run. Replaying the same output creates no duplicate.
    body = canonical(payload) + '\n'
    target = Path(directory) / (digest(chunk_id) + '.' + digest(body) + '.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_text() != body:
                raise ValueError('Immutable relation record collision')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload
