"""Load just the frozen CSV/token utilities without MiniRAG's eager ML imports.

Compile the actual function ASTs from the archived utils.py, rather than
copying implementations. The offline audit remains active during tokenization.
"""
from __future__ import annotations

import ast
import csv
import io
import platform
import sys
import types

from reproduce.mhr_ce import offline
from reproduce.mhr_st.common import OUT as ST_OUT


def install():
    # NumPy asks platform.machine() during import. Cache the local OS query
    # before installing the filesystem audit (its Windows implementation opens
    # the virtual NUL device; it does not create or modify a filesystem file).
    platform.uname()
    offline.setup()
    import tiktoken
    source = ST_OUT / 'source/minirag/utils.py'
    names = {'encode_string_by_tiktoken', 'list_of_list_to_csv'}
    tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in body} != names:
        raise RuntimeError('Frozen utility functions missing')
    package = types.ModuleType('minirag')
    package.__path__ = [str(source.parent)]
    module = types.ModuleType('minirag.utils')
    module.__file__ = str(source)
    module.__dict__.update(ENCODER=None, tiktoken=tiktoken, csv=csv, io=io)
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + body,
                      type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(source), 'exec'), module.__dict__)
    package.utils = module
    sys.modules['minirag'] = package
    sys.modules['minirag.utils'] = module
