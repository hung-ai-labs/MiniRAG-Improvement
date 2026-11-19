"""Run the registered CE secondary-check code on Phi artifacts."""

import argparse
import ast

from . import profile as p, runtime

_SCRIPTS = {'ce-metrics': 'check.py', 'ce-drift': 'drift.py'}
_PATHS = {p.QWEN_ROOTS[0].name: p.ST, p.QWEN_ROOTS[1].name: p.F, p.QWEN_ROOTS[2].name: p.CE}


class _PhiPaths(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            root, _, relative = node.value.partition('/')
            if root in _PATHS:
                node.value = _PATHS[root].as_posix() + '/' + relative
        return node

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'load_dotenv'):
            return None
        return self.generic_visit(node)


def source_for(step):
    path = p.QWEN_ROOTS[2] / 'secondary_checks' / _SCRIPTS[step]
    return _PhiPaths().visit(ast.parse(path.read_bytes(), filename=str(path))), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=tuple(_SCRIPTS))
    args = parser.parse_args()
    if not p._installed:
        parser.error('invoke the secondary checks through reproduce.mhr_phi.launch')
    runtime.setup()
    tree, path = source_for(args.step)
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'),
         {'__name__': '__main__', '__file__': str(path)})


if __name__ == '__main__':
    raise SystemExit(main())
