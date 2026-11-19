"""Apply the registered keyword-reply rule to the frozen Phi context paths."""

import ast
from functools import lru_cache, wraps
import hashlib
import json
import os
from pathlib import Path

import json_repair

from . import profile

EVENTS = profile.OUT / 'kwguard_events.jsonl'


def failure_reason(reply, prompt=''):
    try:
        data = json_repair.loads(reply)
    except json.JSONDecodeError:
        try:
            repaired = reply.replace(prompt[:-1], '').replace('user', '').replace('model', '').strip()
            repaired = '{' + repaired.split('{')[1].split('}')[0] + '}'
            data = json_repair.loads(repaired)
        except (IndexError, json.JSONDecodeError):
            return 'keyword_json_parse_failed'
    if not isinstance(data, dict):
        return 'keyword_reply_not_dict'
    types = data.get('answer_type_keywords', [])
    entities = data.get('entities_from_query', [])
    if isinstance(types, (list, dict, str)) and any(not isinstance(item, str) for item in types):
        return 'answer_type_keywords_non_string'
    if isinstance(entities, (list, str)) and any(not isinstance(item, str) for item in entities[:5]):
        return 'entities_from_query_first_five_non_string'
    return None


def bad(reply, prompt=''):
    return failure_reason(reply, prompt) is not None


@lru_cache(maxsize=8)
def _query_ids(path):
    with path.open(encoding='utf-8') as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        return {row['prompt_sha256']: row['query_id'] for row in rows}


def _split(path, query_id):
    if path.name.startswith('kw_dev_') or path.name == 'synthetic_kw.jsonl':
        return 'dev'
    if path.name.startswith('kw_s'):
        return 'test'
    from .runtime import questions
    return next(split for split in ('dev', 'test')
                if any(row['query_id'] == query_id for row in questions(split)))


def _log(query_id, seed, split, reason):
    event = {'query_id': query_id, 'seed': int(seed), 'split': split, 'reason': reason}
    line = (json.dumps(event, ensure_ascii=False) + '\n').encode('utf-8')
    with EVENTS.open('ab', buffering=0) as handle:
        # Serialize appends across Windows CRT file descriptors as well as POSIX workers.
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        handle.write(line)


class _GuardedReply(Exception):
    pass


class _FailureResponse(str):
    def __new__(cls, text, reason):
        response = super().__new__(cls, text)
        response.reason = reason
        return response


def install(operate):
    lookup = operate._keyword_llm
    query = operate.minirag_query

    @wraps(lookup)
    async def frozen_lookup(use_model_func, kw_prompt):
        reply = await lookup(use_model_func, kw_prompt)
        cache = os.environ.get('MINIRAG_KW_CACHE', '').strip()
        if cache and os.environ.get('MINIRAG_KW_CACHE_ONLY') == '1':
            reason = failure_reason(reply, kw_prompt)
            if reason:
                path = Path(cache).resolve()
                key = hashlib.sha256(kw_prompt.encode('utf-8')).hexdigest()
                query_id = _query_ids(path)[key]
                _log(query_id, os.environ['MINIRAG_SLM_SEED'], _split(path, query_id), reason)
                raise _GuardedReply(reason)
        return reply

    @wraps(query)
    async def minirag_query(*args, **kwargs):
        try:
            return await query(*args, **kwargs)
        except _GuardedReply as exc:
            return _FailureResponse(operate.PROMPTS['fail_response'], exc.args[0])

    operate._keyword_llm = frozen_lookup
    operate.minirag_query = minirag_query


def fields(value, *, include_context=False):
    if not isinstance(value, _FailureResponse):
        return {}
    result = {'keyword_guard': value.reason}
    if include_context:
        result['context'] = str(value)
    return result


def inherited_fields(record, split, seed):
    reason = record.get('keyword_guard')
    if reason is None:
        return {}
    _log(record['query_id'], seed, split, reason)
    return {'keyword_guard': reason}


def count(rows):
    return sum(row['status'] == 'early_exit' and 'keyword_guard' in row for row in rows)


# Keep the frozen record writers and their checkpoint formats. Only add guard
# fields and counts at the points where the writers have the final status.
_SUMMARIES = {
    'reproduce.mhr_st.pipeline': ('summary', '(all_rows[row["query_id"]] for row in wanted)'),
    'reproduce.mhr_f.pool': ('summary', 'rows'),
    'reproduce.mhr_f.run': ('summary', 'ctx.values()'),
    'reproduce.mhr_ce.run': ('summary', 'results'),
    'reproduce.mhr_set.run': ('result', 'results'),
}


def _expression(source):
    return ast.parse(source, mode='eval').body


class RecordSource(ast.NodeTransformer):
    def __init__(self, module):
        self.module = module
        self.function = None

    def _function(self, node):
        parent = self.function
        self.function = node.name
        node = self.generic_visit(node)
        self.function = parent
        return node

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Assign(self, node):
        self.generic_visit(node)
        target = ast.unparse(node.targets[0])
        if self.module == 'reproduce.mhr_st.pipeline' and self.function == 'run' and target == 'previous':
            return [node, *ast.parse('''
guard = _kwguard.inherited_fields(previous, args.split, args.seed)
if guard:
    append(out, previous | {'source': 'keyword_guard', 'anchors': [], 'same_as_c0': True, 'chunk_ids': []})
    continue
''').body]
        if self.module == 'reproduce.mhr_f.pool' and self.function == 'run' and target == 'anchors':
            node.value = _expression('[] if _kwguard.fields(ctx) else adapter.trace["anchors"]')
        if self.module in _SUMMARIES and self.function in ('run', 'contexts'):
            name, rows = _SUMMARIES[self.module]
            if target == name and isinstance(node.value, ast.Dict):
                node.value.keys.append(ast.Constant('keyword_guard_count'))
                node.value.values.append(_expression(f'_kwguard.count({rows})'))
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if (self.module == 'reproduce.mhr_f.run' and self.function == 'run'
                and ast.unparse(node.test) == 'validated < 2'):
            node.test = _expression('validated < 2 and "keyword_guard" not in ref')
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        if (self.module, self.function) == ('reproduce.mhr_f.run', 'run'):
            for index, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == 'same_as_st':
                    node.values[index] = _expression(
                        's == ref["status"] and (ctx if s == "ok" or _kwguard.fields(ctx) else None) == ref["context"]')
        if (self.module, self.function) == ('reproduce.mhr_set.run', 'contexts'):
            for index, key in enumerate(node.keys):
                if (isinstance(key, ast.Constant) and key.value == 'same_as_ce'
                        and isinstance(node.values[index], ast.Compare)):
                    node.values[index] = _expression('context == ref["context"]')
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if not node.args:
            return node
        target = ast.unparse(node.func)
        extension = None
        if (self.module, self.function, target) == ('phased_qa', 'context_phase', 'append_jsonl'):
            extension = '_kwguard.fields(context, include_context=True)'
        elif (self.module, self.function, target) == ('reproduce.mhr_f.pool', 'run', 'append'):
            extension = '_kwguard.fields(ctx)'
        elif (self.module, self.function, target) == ('reproduce.mhr_f.run', 'run', 'append'):
            output = ast.unparse(node.args[0])
            extension = {'pools_out': '_kwguard.fields(ctx)',
                         'ctx_out': '_kwguard.fields(ctx, include_context=True)'}.get(output)
        elif self.module in ('reproduce.mhr_ce.run', 'reproduce.mhr_set.run') and (
                self.function, target) == ('contexts', 'results.append'):
            extension = '_kwguard.inherited_fields(ref, "test", a.seed)'
        if extension:
            node.args[-1] = ast.BinOp(left=node.args[-1], op=ast.BitOr(), right=_expression(extension))
        return node

    def visit_Expr(self, node):
        self.generic_visit(node)
        if (self.module, self.function) == ('reproduce.mhr_f.run', 'query') and (
                ast.unparse(node.value) == 'cap.clear()'):
            return [node, *ast.parse('adapter.trace = {"anchors": []}').body]
        return node
