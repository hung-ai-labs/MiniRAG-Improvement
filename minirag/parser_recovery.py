"""Conservative formatting recovery for new indexing, default OFF at caller.

Do not reinsert historical tuples into a merged graph: doing so double-counts
weights. A repair changes syntax only and never verifies relation semantics.
"""
import re


def recover(raw):
    if not isinstance(raw, str):
        raise TypeError("Extraction must be a string")
    records = raw.split("##")
    audit = []
    for i, record in enumerate(records):
        if not record.lstrip().startswith('("relationship"') or record.count("<|>") != 4 or record.count('"|>') != 1:
            continue
        fixed = record.replace('"|>', '"<|>', 1)
        match = re.search(r"\((.*)\)", fixed)
        fields = match[1].split("<|>") if match else []
        if len(fields) == 6 and all(x.strip() for x in fields) and re.fullmatch(r"\d+(?:\.\d+)?", fields[-1].strip()):
            records[i] = fixed
            audit.append(dict(record_index=i, reason="ONE_MISSING_LT", original=record, repaired=fixed,
                              semantic_status="UNKNOWN"))
    return "##".join(records), audit
