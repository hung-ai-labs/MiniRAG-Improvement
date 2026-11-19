"""Install the slide's ST algorithm over the historical C0 function in memory.

The only dataset adaptation is publication-date metadata -> document chunks.
No text, embedding, index, graph scoring, Entities table or A1 budget is edited.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from datetime import date
import importlib.util
import inspect
from pathlib import Path


def publication_overlay(docs, chunks):
    dates = {}
    for row in docs:
        # Indexed news metadata, never dates extracted from an article's body.
        stamp = row.get('published_at', '')
        dates[row['minirag_doc_id']] = date.fromisoformat(stamp[:10])
    by_time = defaultdict(list)
    years = set()
    for cid, row in sorted(chunks.items()):
        d = dates.get(row['full_doc_id'])
        if d is None:
            raise ValueError(f'No publication metadata for document {row["full_doc_id"]}')
        years.add(d.year)
        by_time['time:day:' + d.isoformat()].append(cid)
        by_time[f'time:month:{d.year:04d}-{d.month:02d}'].append(cid)
    return dict(by_time), years


def load_temporal(path):
    spec = importlib.util.spec_from_file_location('minirag._mhr_st_temporal', Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counter_function(original):
    """Expose the existing kwd2chunk's Counter without rewriting its scoring."""
    tree = ast.parse(inspect.getsource(original))
    fn = tree.body[0]
    if not isinstance(fn.body[-1], ast.Return):
        raise ValueError('Unexpected kwd2chunk return')
    fn.name = '_st_native_scores'
    fn.body[-1] = ast.Return(ast.Name('final_chunk', ast.Load()))
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<st-native-counter>', 'exec'), namespace)
    return namespace[fn.name]


class SourceTime:
    def __init__(self, operate, temporal, docs, chunks):
        self.op = operate
        self.temporal = temporal
        self.by_time, self.years = publication_overlay(docs, chunks)
        self.original = operate._build_mini_query_context
        self.scores = counter_function(operate.kwd2chunk)
        self.trace = None
        tree = ast.parse(inspect.getsource(self.original))
        fn = tree.body[0]
        fn.name = '_st_context'
        inserted = replaced = guards = 0
        body = []
        for statement in fn.body:
            if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name)
                    and statement.value.func.id == 'scorednode2chunk'):
                body += ast.parse('''
_st_pairs = [(keyword, name, scored_edged_reasoning_path[name]['Score'])
             for keyword, names in ent_from_query_dict.items() for name in names
             if name in scored_edged_reasoning_path]
''').body
                inserted += 1
            if (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name)
                    and statement.value.func.id == 'kwd2chunk'):
                statement.value = ast.parse('''_st_rank(ent_from_query_dict, chunks_ids,
                    int(query_param.top_k / 2), knowledge_graph_inst._graph, _st_pairs)''', mode='eval').body
                replaced += 1
            # Same ST fallback as the slide implementation: a time-selected source
            # may be returned when the graph has no matching nodes/edges.
            if (isinstance(statement, ast.If) and isinstance(statement.test, ast.UnaryOp)
                    and isinstance(statement.test.op, ast.Not)
                    and ast.unparse(statement.test.operand) in ('len(results_node)', 'len(results_edge)')
                    and len(statement.body) == 1 and isinstance(statement.body[0], ast.Return)):
                statement.test = ast.BoolOp(ast.And(), [statement.test,
                    ast.UnaryOp(ast.Not(), ast.Name('final_chunk_id', ast.Load()))])
                guards += 1
            body.append(statement)
        if (inserted, replaced, guards) != (1, 1, 2):
            raise ValueError(f'C0 function shape changed: {inserted}/{replaced}/{guards}')
        fn.body = body
        namespace = dict(self.original.__globals__) | {'_st_rank': self.rank}
        exec(compile(ast.fix_missing_locations(tree), '<mhr-source-time-only>', 'exec'), namespace)
        self.modified = namespace[fn.name]
        self.signature = inspect.signature(self.original)

    def rank(self, groups, vector_ids, limit, graph, pairs):
        native = self.op.kwd2chunk(groups, vector_ids, limit)
        scores = self.scores(groups, vector_ids, limit)
        if native != [c for c, _ in scores.most_common(limit)]:
            raise ValueError('Native Counter does not reproduce C0 ranking')
        h = self.temporal.entity_h(graph, pairs)
        ranked, promoted = self.temporal.rank_temporal(
            scores, self.by_time, self.trace['anchors'], h, vector_ids,
            expand=True, h_one=False, limit=limit)
        self.trace.update(native_ids=native, ranked_ids=ranked, promoted=promoted,
                          ranking_changed=ranked != native)
        return ranked

    async def call(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs).arguments
        anchors = self.temporal.parse_query(bound['originalquery'], self.years)
        self.trace = {'anchors': list(anchors), 'ranking_changed': False, 'promoted': {}}
        if not anchors:
            return await self.original(*args, **kwargs)
        return await self.modified(*args, **kwargs)

    def install(self):
        self.op._build_mini_query_context = self.call
