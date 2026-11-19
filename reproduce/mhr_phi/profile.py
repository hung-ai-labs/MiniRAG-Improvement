"""Select Phi before importing or running any of the frozen ladder modules."""

import ast
from contextlib import contextmanager
from contextvars import ContextVar
from importlib.machinery import PathFinder, SourceFileLoader
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'review/mhr_phi_20261118'
ST = OUT / 'st'
F = OUT / 'f'
CE = OUT / 'ce'
SET = OUT / 'set'
C0 = OUT / 'c0'
ARCHIVE = C0 / 'archive'
DEV_C0 = C0 / 'dev/C0_s101'
INDEX = ROOT / 'MultiHop-RAG-phi-ropefix609'
PILOT = OUT / 'index_source/indexes/multihop_rag-phi-pilot30'
SERVING = OUT / 'serving_phi_ropefix_modal.json'
MODEL = 'microsoft/Phi-3.5-mini-instruct'
TITLE = 'Phi-3.5-mini-instruct (LongRoPE fix)'
PACKAGES = tuple(f'reproduce.mhr_{row}' for row in ('st', 'f', 'ce', 'set'))
QWEN_PATHS = {
    'review/mhr_st_20261112': 'ST',
    'review/mhr_f_20261114': 'F',
    'review/mhr_ce_20261116': 'CE',
    'review/mhr_set_20261117': 'SET',
    'MultiHop-RAG-qwen-full609': 'INDEX',
}
QWEN_ROOTS = tuple(ROOT / path for path in QWEN_PATHS)
SHARED_SOURCE = QWEN_ROOTS[0] / 'source'
SHARED_INPUTS = QWEN_ROOTS[0] / 'inputs'
EMBEDDING_RECEIPT = QWEN_ROOTS[0] / 'embedding_cache.json'

_dotenv_read = ContextVar('phi_dotenv_read', default=None)
_installed = False
_network_denied = False
_NETWORK_EVENTS = {'socket.connect', 'socket.connect_ex', 'socket.bind', 'socket.getaddrinfo',
                   'socket.getnameinfo', 'socket.gethostbyname', 'socket.gethostbyaddr',
                   'socket.sendto', 'socket.sendmsg', 'http.client.connect'}


def is_ladder(name):
    return any(name == package or name.startswith(package + '.') for package in PACKAGES)


def _path(value):
    if isinstance(value, (str, bytes, os.PathLike)):
        return Path(os.fsdecode(value))
    return None


def _protect(path):
    if path is None:
        return
    literal = Path(os.path.abspath(path))
    resolved = path.resolve()
    if any(candidate.is_relative_to(root) for candidate in (literal, resolved) for root in QWEN_ROOTS):
        raise PermissionError(f'Qwen artifact is read-only under the Phi profile: {path}')


def _audit(event, args):
    if _network_denied and event in _NETWORK_EVENTS:
        # CPython on Windows builds asyncio's internal wakeup pipe with a
        # loopback socket pair. Only that standard-library call may connect.
        frame = sys._getframe(1)
        internal_pipe = (event in ('socket.bind', 'socket.connect')
                         and frame.f_globals.get('__name__') == 'socket'
                         and frame.f_code.co_name == '_fallback_socketpair'
                         and args[1][0] in ('127.0.0.1', '::1'))
        if not internal_pipe:
            raise PermissionError(f'Network is disabled for offline Phi work: {event}')
    if event == 'open':
        path = _path(args[0])
        if path is None:
            return
        mode, flags = args[1:]
        mutation_flags = (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
                          | getattr(os, 'O_TEMPORARY', 0))
        writing = bool((mode and any(flag in mode for flag in 'wax+')) or flags & mutation_flags)
        allowed = _dotenv_read.get()
        names = (path.name.lower(), path.resolve().name.lower()) if not writing else (path.name.lower(),)
        if any(name == '.env' or name.startswith('.env.') for name in names) or (
            allowed is not None and path.resolve() == allowed
        ):
            if writing or allowed is None or path.resolve() != allowed:
                raise PermissionError('Only the Phi launcher may read its explicitly selected dotenv file')
        if writing:
            _protect(path)
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        for value in args[:2]:
            _protect(_path(value))
    elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.chown',
                   'os.utime', 'os.truncate', 'os.chflags', 'shutil.rmtree'):
        _protect(_path(args[0]))


@contextmanager
def _launcher_dotenv(path):
    allowed = Path(path).resolve(strict=True)
    token = _dotenv_read.set(allowed)
    try:
        yield allowed
    finally:
        _dotenv_read.reset(token)


def no_dotenv(*args, **kwargs):
    return False


# These are the function-local exceptions found in the AST scan. Ordinary
# imported aliases and derived constants evaluate normally after the rewrite.
_FUNCTIONS = {
    'mhr_st.common': {'setup': 'return _phi_runtime.setup(source)'},
    'mhr_st.run': {'prepare': 'return _phi_runtime.prepare_st()',
                   'preflight': 'return _phi_runtime.preflight_st()'},
    'mhr_st.finish': {'amendment': 'return None'},
    'mhr_ce.offline': {'setup': ('from reproduce.mhr_f import study\n'
                                 'study.setup = setup\n'
                                 'return _phi_runtime.setup(source)')},
}
_PATH_EXPRESSIONS = {
    "OUT / 'archive_c0'": 'ARCHIVE',
    "ROOT / 'runs/multihop_rag/dev400/C0_s101'": 'DEV_C0',
    "ROOT.parent / 'MiniRAG/runs/multihop_rag/dev400/C0_s101/context.jsonl'": 'DEV_C0 / "context.jsonl"',
    "OUT / 'source'": 'SHARED_SOURCE',
    "ST_OUT / 'source'": 'SHARED_SOURCE',
    "ST_OUT / 'embedding_cache.json'": 'EMBEDDING_RECEIPT',
    "OUT / 'embedding_cache.json'": 'EMBEDDING_RECEIPT',
}


class _PhiSource(ast.NodeTransformer):
    def __init__(self, module):
        self.module = module.removeprefix('reproduce.')

    def visit_FunctionDef(self, node):
        replacement = _FUNCTIONS.get(self.module, {}).get(node.name)
        if replacement:
            node.body = ast.parse(replacement).body
            return node
        if self.module == 'mhr_st.pipeline' and node.name == 'execute':
            # The original execute judges all ST answers before finish can apply
            # the registered identical-answer reuse. Let finish do the judging.
            block = node.body[0]
            first = next(i for i, statement in enumerate(block.body)
                         if isinstance(statement, ast.For)
                         and "judge('test', seed)" in ast.unparse(statement))
            block.body[first:-1] = ast.parse(
                "write(OUT / 'status.json', {'state': 'contexts_and_answers_done'})").body
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        if not isinstance(node.op, ast.Div):
            return self.generic_visit(node)
        replacement = _PATH_EXPRESSIONS.get(ast.unparse(node))
        if replacement:
            return ast.copy_location(ast.parse('_phi.' + replacement, mode='eval').body, node)
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            path = node.right.value
            if path == 'source/reproduce/research/serving_qwen_modal.json':
                return ast.copy_location(ast.parse('_phi.SERVING', mode='eval').body, node)
            if path.startswith('source/'):
                return ast.copy_location(ast.parse(
                    f'_phi.SHARED_SOURCE / {path.removeprefix("source/")!r}', mode='eval').body, node)
        if (isinstance(node.right, ast.JoinedStr)
                and isinstance(node.right.values[0], ast.Constant)
                and node.right.values[0].value.startswith('archive_c0/')):
            node.left = ast.parse('_phi.ARCHIVE', mode='eval').body
            node.right.values[0].value = node.right.values[0].value.removeprefix('archive_c0/')
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return node
        # The scores module constructs its review path from two segments.
        if self.module == 'mhr_ce.scores' and node.value == 'mhr_ce_20261116':
            node.value = 'mhr_phi_20261118/ce'
        elif node.value in QWEN_PATHS:
            node.value = str(globals()[QWEN_PATHS[node.value]].relative_to(ROOT))
        elif 'Qwen2.5-3B | Judge:' in node.value:
            node.value = node.value.replace('Qwen2.5-3B', TITLE)
        elif node.value == 'SET vs F is secondary because CE was not confirmed over F.':
            node.value = 'SET vs F is a registered secondary comparison.'
        elif self.module == 'mhr_st.run' and 'MultiHop-RAG/Qwen benchmark' in node.value:
            node.value = node.value.replace('MultiHop-RAG/Qwen benchmark', 'MultiHop-RAG/Phi benchmark')
        elif self.module == 'mhr_f.pool' and 'review/mhr_st_20261112/source' in node.value:
            node.value = node.value.replace('review/mhr_st_20261112/source', 'shared frozen source')
            node.value = node.value.replace('MiniRAG dev400 C0_s101', 'Phi dev400 C0_s101')
        elif self.module == 'mhr_ce.scores' and 'review/mhr_ce_20261116/' in node.value:
            node.value = node.value.replace('review/mhr_ce_20261116/', 'review/mhr_phi_20261118/ce/')
        elif self.module == 'mhr_ce.modal_ce' and 'review/mhr_ce_20261116/PROTOCOL.md' in node.value:
            node.value = node.value.replace('review/mhr_ce_20261116/PROTOCOL.md', 'review/mhr_phi_20261118/PROTOCOL.md')
        return node


class _PhiLoader(SourceFileLoader):
    def get_code(self, fullname):
        from . import kwguard

        filename = self.get_filename(fullname)
        tree = ast.parse(self.get_data(filename), filename)
        if is_ladder(fullname):
            tree = _PhiSource(fullname).visit(tree)
        tree = kwguard.RecordSource(fullname).visit(tree)
        # Preserve the docstring and future imports, for both ordinary imports
        # and runpy's independent __main__ namespace.
        first = int(bool(ast.get_docstring(tree, clean=False)))
        while (first < len(tree.body) and isinstance(tree.body[first], ast.ImportFrom)
               and tree.body[first].module == '__future__'):
            first += 1
        tree.body[first:first] = ast.parse(
            'from reproduce.mhr_phi import profile as _phi, runtime as _phi_runtime, kwguard as _kwguard').body
        return compile(ast.fix_missing_locations(tree), filename, 'exec')

    def exec_module(self, module):
        from . import jsonl

        super().exec_module(module)
        jsonl.install(module)  # DEVIATIONS.md 2026-11-19: split JSONL on "\n" only


class _JsonlLoader(SourceFileLoader):
    def exec_module(self, module):
        from . import jsonl

        super().exec_module(module)
        jsonl.install(module)


class _KeywordLoader(SourceFileLoader):
    def exec_module(self, module):
        from . import kwguard

        super().exec_module(module)
        kwguard.install(module)


class _PhiFinder:
    _frozen = {
        'minirag.operate': (SHARED_SOURCE / 'minirag/operate.py', _KeywordLoader),
        'phased_qa': (SHARED_SOURCE / 'reproduce/research/phased_qa.py', _PhiLoader),
        'judge_run': (SHARED_SOURCE / 'reproduce/research/judge_run.py', _JsonlLoader),
    }

    def find_spec(self, fullname, path=None, target=None):
        frozen = self._frozen.get(fullname)
        if not is_ladder(fullname) and frozen is None:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and isinstance(spec.loader, SourceFileLoader):
            if frozen is None:
                spec.loader = _PhiLoader(fullname, spec.origin)
            elif Path(spec.origin).resolve() == frozen[0]:
                spec.loader = frozen[1](fullname, spec.origin)
        return spec


def apply(*, offline=False):
    """Install the profile before ladder import, with optional socket denial."""
    global _installed, _network_denied
    _network_denied |= offline
    if _installed:
        return
    already_loaded = sorted(name for name in sys.modules if is_ladder(name) or name in _PhiFinder._frozen)
    if already_loaded:
        raise RuntimeError(f'Apply the Phi profile before importing the ladder: {already_loaded}')
    sys.dont_write_bytecode = True
    sys.addaudithook(_audit)
    sys.meta_path.insert(0, _PhiFinder())
    import dotenv
    dotenv.load_dotenv = no_dotenv
    _installed = True
