"""Offline profile checks. Run with python -B -m reproduce.mhr_phi.test_profile."""

import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from . import launch, profile as p, runtime


def ladder_modules():
    """Scan all package files, including modules used only in child processes."""
    for package in p.PACKAGES:
        for path in sorted((p.ROOT / package.replace('.', '/')).rglob('*.py')):
            relative = path.relative_to(p.ROOT).with_suffix('')
            parts = relative.parts[:-1] if relative.name == '__init__' else relative.parts
            yield '.'.join(parts), path


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.apply(offline=True)
        launch.install_subprocess_rewrite()
        cls.modules = {name: importlib.import_module(name) for name, _ in ladder_modules()}

    def test_every_global_and_imported_alias(self):
        allowed = {p.SHARED_SOURCE, p.EMBEDDING_RECEIPT, p.SHARED_INPUTS / 'dev.csv',
                   p.SHARED_INPUTS / 'test.csv'}
        forbidden = (*p.QWEN_PATHS, 'qwen-full609', 'serving_qwen',
                     'minirag/runs/multihop_rag/dev400', 'Qwen2.5-3B')
        for name, module in self.modules.items():
            for key, value in vars(module).items():
                if isinstance(value, (str, Path)):
                    normalized = str(value).replace('\\', '/').lower()
                    if any(marker.lower() in normalized for marker in forbidden):
                        self.assertIn(Path(value), allowed, (name, key, value))

        definitions = {
            'reproduce.mhr_st.common': {'OUT': p.ST, 'INDEX': p.INDEX, 'ARCHIVE': p.ARCHIVE},
            'reproduce.mhr_f.pool': {'OUT': p.F, 'DEV_C0_SRC': p.DEV_C0 / 'context.jsonl'},
            'reproduce.mhr_f.run': {'OUT': p.F, 'MODEL': p.F / 'model_F.json'},
            'reproduce.mhr_ce.offline': {'OUT': p.CE},
            'reproduce.mhr_ce.scores': {'OUT': p.CE},
            'reproduce.mhr_ce.run': {'OUT': p.CE, 'MODEL': p.CE / 'model_CE.json'},
            'reproduce.mhr_set.offline': {'OUT': p.SET},
            'reproduce.mhr_set.study': {'MODEL': p.CE / 'model_CE.json',
                                        'SCORES': p.CE / 'dev_offline/scores_CEF.jsonl',
                                        'DEV_POOLS': p.F / 'dev/s101/pools.jsonl'},
            'reproduce.mhr_set.run': {'OUT': p.SET, 'PARAMS': p.SET / 'set_params.json'},
        }
        for name, expected in definitions.items():
            for key, value in expected.items():
                self.assertEqual(getattr(self.modules[name], key), value, (name, key))

        copies = []
        for name, path in ladder_modules():
            module = self.modules[name]
            tree = ast.parse(path.read_bytes(), filename=str(path))
            for node in tree.body:
                if not isinstance(node, ast.ImportFrom):
                    continue
                origin = node.module or ''
                if node.level:
                    origin = importlib.util.resolve_name('.' * node.level + origin, module.__package__)
                for binding in node.names:
                    expected = definitions.get(origin, {}).get(binding.name)
                    if expected is not None:
                        key = binding.asname or binding.name
                        self.assertEqual(getattr(module, key), expected, (name, key, origin))
                        copies.append(f'{name}.{key}')
        self.assertTrue(copies, 'AST import scan found no artifact aliases')
        print(f'Profile import scan: {len(self.modules)} modules, {len(copies)} artifact aliases', flush=True)

    def test_runtime_paths_and_question_hashes(self):
        pipeline = self.modules['reproduce.mhr_st.pipeline']
        self.assertEqual(pipeline.reference('dev', 101), p.DEV_C0)
        self.assertEqual(pipeline.reference('test', 202), p.ARCHIVE / 'qa/s202/C0_s202')
        self.assertEqual(self.modules['reproduce.mhr_st.finish'].QUESTIONS, p.ST / 'inputs/test.csv')
        self.assertEqual(runtime.setup(), p.SHARED_SOURCE.resolve())
        self.assertEqual(runtime.phased_runner().INDEX, p.INDEX)
        self.assertEqual(runtime.phased_runner().snapshot.__defaults__, (p.INDEX,))
        self.assertTrue(all(runtime.source_checks().values()))
        from research.mhr_phase3 import shared_context
        self.assertEqual(Path(shared_context.__file__).resolve(),
                         p.SHARED_SOURCE / 'reproduce/research/mhr_phase3/shared_context.py')
        pool = self.modules['reproduce.mhr_f.pool']
        with patch.object(pool, 'records', return_value=[]) as records:
            self.assertEqual(pool.c0_reference('test', 303), {})
            records.assert_called_once_with(p.ARCHIVE / 'qa/s303/C0_s303/context.jsonl')
        for split, expected in runtime.QUESTION_HASHES.items():
            self.assertEqual(runtime.sha(p.SHARED_INPUTS / f'{split}.csv'), expected)
            self.assertEqual(runtime.sha(p.ST / 'inputs' / f'{split}.csv'), expected)

    def test_subprocess_run_and_popen_propagate_profile(self):
        prefix = [sys.executable, '-B', '-u', '-m']
        for name in ('reproduce.mhr_st.run', 'reproduce.mhr_st.judge', 'reproduce.mhr_f.pool',
                     'reproduce.mhr_ce.run', 'reproduce.mhr_ce.judge_entry',
                     'reproduce.mhr_set.run', 'reproduce.mhr_set.judge_entry', launch.C0_MODULE):
            original = prefix + [name, '--help']
            self.assertEqual(launch.rewrite_command(original), prefix + [launch.MODULE, name, '--help'])
        other = prefix + ['json.tool']
        self.assertIs(launch.rewrite_command(other), other)
        script = [sys.executable, 'script.py', '-m', 'reproduce.mhr_st.run']
        self.assertIs(launch.rewrite_command(script), script)
        already = prefix + [launch.MODULE, 'reproduce.mhr_st.run']
        self.assertIs(launch.rewrite_command(already), already)

        command = prefix + ['reproduce.mhr_st.run', 'status']
        options = {'cwd': p.ROOT, 'env': os.environ | {'MINIRAG_DOTENV': str(p.ROOT / '.env.phi-profile-test')},
                   'text': True}
        result = subprocess.run(command, check=True, capture_output=True, **options)
        self.assertEqual(Path(json.loads(result.stdout)['run']), p.ST)
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options) as child:
            stdout, stderr = child.communicate()
            self.assertEqual(child.returncode, 0, stderr)
        self.assertEqual(Path(json.loads(stdout)['run']), p.ST)

    def test_guard_blocks_qwen_writes_dotenv_and_sockets(self):
        for root in p.QWEN_ROOTS:
            target = root / '_phi_guard_must_never_exist'
            with self.assertRaisesRegex(PermissionError, 'Qwen artifact is read-only'):
                target.open('xb')
            with self.assertRaisesRegex(PermissionError, 'Qwen artifact is read-only'):
                sys.audit('os.rename', str(p.OUT / 'placeholder'), str(target), -1, -1)
            with self.assertRaisesRegex(PermissionError, 'Qwen artifact is read-only'):
                sys.audit('os.remove', str(target), -1)
        for name in ('.env', '.ENV.local'):
            with self.assertRaisesRegex(PermissionError, 'Only the Phi launcher'):
                (p.ROOT / name).open('rb')
        with self.assertRaisesRegex(PermissionError, 'Network is disabled'):
            sys.audit('socket.getaddrinfo', 'offline-test.invalid', 443, 0, 0, 0)
        sys.audit('open', str(p.OUT / 'permitted_append.jsonl'), 'a', os.O_WRONLY | os.O_APPEND)

    def test_mock_serving_and_response_gates(self):
        from types import SimpleNamespace

        spec = runtime.serving_spec()
        verified = [{'id': 'minirag-slm'}, {'id': p.MODEL, 'model_revision': spec['model_revision']}]

        def reply(data):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

        env = {'MINIRAG_SLM_URL': 'https://offline-test.invalid/v1', 'MINIRAG_SLM_KEY': 'test',
               'MINIRAG_KW_CACHE': ''}
        with patch.dict(os.environ, env):
            with patch('httpx.get', side_effect=[reply({'version': '0.11.0'}), reply({'data': verified})]) as get:
                self.assertEqual(runtime.check_serving()['observed_model_revision'], spec['model_revision'])
                self.assertEqual([call.args[0].rsplit('.invalid', 1)[1] for call in get.call_args_list],
                                 ['/version', '/v1/models'])
            for version, models, error in (
                ('0.11.0', [{'id': 'minirag-slm'}], 'model names'),
                ('0.12.0', verified, 'vLLM version'),
                ('0.11.0', [verified[0], verified[1] | {'model_revision': 'wrong'}], 'model revision'),
            ):
                with patch('httpx.get', side_effect=[reply({'version': version}), reply({'data': models})]):
                    with self.assertRaisesRegex(RuntimeError, error):
                        runtime.check_serving()
            for name, dest in (('reproduce.mhr_st.pipeline', p.ST), ('reproduce.mhr_f.run', p.F)):
                module = self.modules[name]
                with (patch.object(module, 'read', wraps=module.read) as read,
                      patch.object(module, 'append') as append,
                      patch('httpx.get', side_effect=[reply({'version': '0.11.0'}), reply({'data': verified})])):
                    module.configure_slm(SimpleNamespace(use_slm=lambda: None), 101)
                    read.assert_called_once_with(p.SERVING)
                    self.assertEqual(append.call_args.args[0], dest / 'serving_checks.jsonl')
                    self.assertEqual(append.call_args.args[1]['model'], p.MODEL)
        runtime.require_phi_response({'ok': True, 'response_model': p.MODEL})
        runtime.require_phi_response({'ok': False})
        for model in ('Qwen/Qwen2.5-3B-Instruct', 'minirag-slm', None):
            with self.assertRaisesRegex(RuntimeError, 'did not identify'):
                runtime.require_phi_response({'ok': True, 'response_model': model})


if __name__ == '__main__':
    unittest.main(verbosity=2)
