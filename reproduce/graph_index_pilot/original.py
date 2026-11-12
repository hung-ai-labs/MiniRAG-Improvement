"""Execute only checked-out MiniRAG parser/merge functions in memory.

AST loading avoids importing minirag's SDKs/install hooks. No Graph V2 dependency.
G0 deliberately preserves the original len>=5 relationship parser, including its
permissive malformed-tuple behavior. Parser recovery is not guaranteed to add an edge.
"""
import ast
import asyncio
from collections import Counter, defaultdict
import hashlib
import html
import logging
from pathlib import Path
import re
import runpy

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
SEP = "<SEP>"


def _functions(path, names, ns):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    if len(selected) != len(names):
        raise ValueError("Original parser/merge function inventory changed")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)


def namespace():
    ns = dict(re=re, html=html, asyncio=asyncio, Counter=Counter, defaultdict=defaultdict,
              GRAPH_FIELD_SEP=SEP, logger=logging.getLogger(__name__))
    _functions(ROOT / "minirag/utils.py", {"clean_str", "is_float_regex", "split_string_by_multi_markers"}, ns)
    _functions(ROOT / "minirag/operate.py", {"_handle_single_entity_extraction", "_handle_single_relationship_extraction",
               "_merge_nodes_then_upsert", "_merge_edges_then_upsert"}, ns)
    return ns


def prompt(text):
    p = runpy.run_path(str(ROOT / "minirag/prompt.py"))["PROMPTS"]
    return p["entity_extraction"].format(input_text=text, tuple_delimiter=p["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=p["DEFAULT_RECORD_DELIMITER"], completion_delimiter=p["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(p["DEFAULT_ENTITY_TYPES"]))


def relation_id(cid, index):
    return "relation-" + hashlib.sha256(f"{cid}:{index}".encode()).hexdigest()[:24]


def recover(raw):
    """Only a single missing '<' in a six-field numeric relationship tuple."""
    records, audit = raw.split("##"), []
    for i, rec in enumerate(records):
        if not rec.lstrip().startswith('("relationship"') or rec.count("<|>") != 4 or rec.count('"|>') != 1:
            continue
        fixed = rec.replace('"|>', '"<|>', 1)
        m = re.search(r"\((.*)\)", fixed)
        attrs = m[1].split("<|>") if m else []
        if len(attrs) == 6 and all(x.strip() for x in attrs) and re.fullmatch(r"\d+(?:\.\d+)?", attrs[-1].strip()):
            records[i] = fixed
            audit.append(dict(record_index=i, original=rec, repaired=fixed, reason="ONE_MISSING_LT"))
    return "##".join(records), audit


async def _parse(cid, raw, ns):
    entities, relations, rejected = [], [], []
    records = ns["split_string_by_multi_markers"](raw, ["##", "<|COMPLETE|>"])
    for i, record in enumerate(records):
        m = re.search(r"\((.*)\)", record)
        if not m:
            if record.strip():
                rejected.append(dict(record_index=i, reason="INCOMPLETE_TUPLE"))
            continue
        attrs = ns["split_string_by_multi_markers"](m[1], ["<|>"])
        entity = await ns["_handle_single_entity_extraction"](attrs, cid)
        relation = None if entity else await ns["_handle_single_relationship_extraction"](attrs, cid)
        if entity:
            entities.append(entity)
        elif relation:
            relations.append({**relation, "relation_id": relation_id(cid, i), "record_index": i, "raw_arity": len(attrs)})
        elif attrs and attrs[0] != '"content_keywords"':
            rejected.append(dict(record_index=i, reason="ORIGINAL_PARSER_REJECTED"))
    return dict(entities=entities, relations=relations, rejected=rejected)


def parse(cid, raw):
    return asyncio.run(_parse(cid, raw, namespace()))


class MemoryGraph:
    def __init__(self):
        self.graph = nx.Graph()

    async def get_node(self, n):
        return self.graph.nodes.get(n)

    async def has_node(self, n):
        return n in self.graph

    async def upsert_node(self, n, node_data):
        self.graph.add_node(n, **node_data)

    async def has_edge(self, a, b):
        return self.graph.has_edge(a, b)

    async def get_edge(self, a, b):
        return self.graph.edges.get((a, b))

    async def upsert_edge(self, a, b, edge_data):
        self.graph.add_edge(a, b, **edge_data)


async def _merge(parsed):
    ns, store = namespace(), MemoryGraph()
    nodes, edges = defaultdict(list), defaultdict(list)
    for batch in parsed:
        for row in batch["entities"]:
            nodes[row["entity_name"]].append(row)
        for row in batch["relations"]:
            # Exactly the endpoint merge of extract_entities, not tuple direction.
            edges[tuple(sorted((row["src_id"], row["tgt_id"])))].append(row)
    for name, rows in nodes.items():
        await ns["_merge_nodes_then_upsert"](name, rows, store, {})
    for (u, v), rows in edges.items():
        await ns["_merge_edges_then_upsert"](u, v, rows, store, {})
    # Source IDs are sets in the original code. Canonicalize serialization only.
    for _, attr in store.graph.nodes(data=True):
        attr["source_id"] = SEP.join(sorted(set(attr["source_id"].split(SEP))))
    for _, _, attr in store.graph.edges(data=True):
        attr["source_id"] = SEP.join(sorted(set(attr["source_id"].split(SEP))))
    return store.graph


def merge(parsed):
    return asyncio.run(_merge(parsed))
