"""Isolated indexing ablations. No production registration, QA or model on import."""

VERSION = "graph-index-eight-arm-20261111-v1"
ARMS = {
    "G0": dict(recovery=False, provenance=False, source_time=False, event_time=False, directed=False, trust=False),
    "B": dict(recovery=True, provenance=False, source_time=False, event_time=False, directed=False, trust=False),
    "P": dict(recovery=True, provenance=True, source_time=False, event_time=False, directed=False, trust=False),
    "TS": dict(recovery=True, provenance=True, source_time=True, event_time=False, directed=False, trust=False),
    "TE": dict(recovery=True, provenance=True, source_time=True, event_time=True, directed=False, trust=False),
    "D": dict(recovery=True, provenance=True, source_time=False, event_time=False, directed=True, trust=False),
    "TD": dict(recovery=True, provenance=True, source_time=True, event_time=True, directed=True, trust=False),
    "TDV": dict(recovery=True, provenance=True, source_time=True, event_time=True, directed=True, trust=True),
}
