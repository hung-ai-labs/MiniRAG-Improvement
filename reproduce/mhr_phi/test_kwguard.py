"""Offline regression checks against the real Phi and Qwen keyword caches."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from . import c0, kwguard, profile as p, runtime

BAD_ID = 'mhr-0315'
CONTROLS = ('mhr-0000', 'mhr-0001', 'mhr-0002')


class KeywordGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('PYTHONHASHSEED') != c0.HASH_SEED:
            raise RuntimeError('Start the tests with PYTHONHASHSEED=13')
        p.apply(offline=True)
        cls.phased = runtime.phased_runner()
        cls.cohort = c0.Cohort('test', 202)
        cls.good = [row for row in cls.cohort.rows() if row['query_id'] in CONTROLS]
        cls.bad_row = next(row for row in cls.cohort.rows() if row['query_id'] == BAD_ID)
        cls.log_key = {row['query_id']: row['Question'] for row in [*cls.good, cls.bad_row]}
        cls.fail_reason = 'entities_from_query_first_five_non_string'
        c0.OFFLINE.mkdir(exist_ok=True)
        cls.root = Path(tempfile.mkdtemp(prefix='kwguard_', dir=c0.OFFLINE))
        print(f'Keyword guard test artifacts: {cls.root}', flush=True)

    def test_parse_rule(self):
        for reply, expected in (
            ('[]', True), ('{}', False), ('{"answer_type_keywords": ["person", 2023]}', True),
            ('{"entities_from_query": ["a", "b", "c", "d", 2023, "f"]}', True),
            ('{"entities_from_query": ["a", "b", "c", "d", "e", 2023]}', False),
        ):
            with self.subTest(reply=reply):
                self.assertEqual(kwguard.bad(reply), expected)
        prompt = 'Example {irrelevant}.'
        reply = 'user Example {irrelevant}model {"answer_type_keywords": [2023]} '
        with patch.object(kwguard.json_repair, 'loads', side_effect=[
            json.JSONDecodeError('test the fallback', reply, 0), {'answer_type_keywords': [2023]},
        ]) as loads:
            self.assertTrue(kwguard.bad(reply, prompt))
            self.assertEqual(loads.call_args.args, ('{"answer_type_keywords": [2023]}',))

    def test_real_caches(self):
        checks = {
            p.ST / 'inputs/kw_dev_s101.jsonl': set(),
            p.ST / 's101/kw_s101.jsonl': set(),
            self.cohort.keywords: {BAD_ID},
            p.SHARED_INPUTS / 'kw_dev_s101.jsonl': set(),
        } | {path: set() for path in sorted(p.QWEN_ROOTS[0].glob('s*/kw_s*.jsonl'))}
        self.assertEqual(len(checks), 7)
        for path, expected in checks.items():
            with self.subTest(path=path):
                rows = self.phased.read_jsonl(path)
                self.assertTrue(rows)
                bad_ids = {qid for qid, row in rows.items() if kwguard.bad(row['result'])}
                self.assertEqual(bad_ids, expected)
                print(f'{path.parent.name}/{path.name}: {len(rows)} replies, guarded IDs {sorted(bad_ids)}')

    def test_good_replies_emit_no_events(self):
        from minirag import operate
        from minirag.prompt import PROMPTS

        graph = ET.parse(p.INDEX / 'graph_chunk_entity_relation.graphml')
        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
        type_key = next(node.get('id') for node in graph.findall('g:key', ns)
                        if node.get('attr.name') == 'entity_type' and node.get('for') == 'node')
        types = sorted({node.text.lower() for node in graph.findall('g:graph/g:node/g:data', ns)
                        if node.get('key') == type_key and node.text})
        events = self.root / 's101_events.jsonl'

        async def check(split):
            cohort = c0.Cohort(split, 101)
            self.phased.apply_env('C0', 101, kw_cache=cohort.keywords, cache_only=True)
            expected = self.phased.read_jsonl(cohort.keywords)
            for row in cohort.rows():
                prompt = PROMPTS['minirag_query2kwd'].format(query=row['Question'], TYPE_POOL=types)
                frozen = expected[row['query_id']]
                self.assertEqual(hashlib.sha256(prompt.encode('utf-8')).hexdigest(), frozen['prompt_sha256'])
                self.assertEqual(await operate._keyword_llm(self.phased._forbidden_llm, prompt), frozen['result'])

        with patch.object(kwguard, 'EVENTS', events):
            for split in ('dev', 'test'):
                asyncio.run(check(split))
        self.assertFalse(events.exists())

    def test_c0_and_pool_on_full_index(self):
        from minirag.prompt import PROMPTS
        from research.mhr_phase3.shared_context import SharedContext, stable_log
        from reproduce.mhr_f import pool
        from reproduce.mhr_st.adapter import SourceTime

        runtime.verify_index(p.INDEX)
        wanted = [*self.good, self.bad_row]
        events = self.root / 'events.jsonl'

        def c0_run(name, rows):
            class Sample(c0.Cohort):
                @property
                def dest(self):
                    return dest

                def rows(self):
                    return rows

            dest = self.root / name
            sample = Sample('test', 202)
            with patch.object(c0, 'WORKERS', 1):
                c0.context_worker(sample, self.phased, 0)
            part = dest / 'worker0'
            return (self.phased.read_jsonl(part / 'contexts.jsonl'),
                    self.phased.read_jsonl(part / 'context.jsonl', key='query'))

        def check_prefix(adapter, *args, **kwargs):
            query = adapter.signature.bind(*args, **kwargs).arguments['originalquery']
            self.assertNotEqual(query, self.bad_row['Question'])

        original = SharedContext.call

        async def shared_call(adapter, *args, **kwargs):
            check_prefix(adapter, *args, **kwargs)
            return await original(adapter, *args, **kwargs)

        with patch.object(kwguard, 'EVENTS', events), patch.object(SharedContext, 'call', shared_call):
            guarded, logs = c0_run('c0_guarded', wanted)
            self.assertEqual(guarded[BAD_ID]['status'], 'early_exit')
            self.assertEqual(guarded[BAD_ID]['context'], PROMPTS['fail_response'])
            self.assertEqual(guarded[BAD_ID]['keyword_guard'], self.fail_reason)
            self.assertNotIn(self.log_key[BAD_ID], logs)
            self.assertEqual(list(self.phased.read_jsonl(events)), [BAD_ID])
            with patch.object(kwguard, 'failure_reason', return_value=None):
                unguarded, old_logs = c0_run('c0_unguarded_controls', self.good)
            shards = [self.cohort.dest / f'worker{shard}' for shard in range(c0.WORKERS)]
            prior = {qid: row for shard in shards for qid, row in
                     self.phased.read_jsonl(shard / 'contexts.jsonl').items()}
            for qid in CONTROLS:
                self.assertEqual(guarded[qid]['status'], 'ok')
                context = guarded[qid]['context'].encode('utf-8')
                self.assertEqual(context, unguarded[qid]['context'].encode('utf-8'))
                if qid in prior:
                    self.assertEqual(context, prior[qid]['context'].encode('utf-8'))
                self.assertEqual(stable_log(logs[self.log_key[qid]]), stable_log(old_logs[self.log_key[qid]]))

        def pool_run(name, rows):
            dest = self.root / name
            args = SimpleNamespace(split='test', seed=202, workers=1, shard=0, limit=None)
            sources = {query: rec['chunk_ids'] for query, rec in logs.items()}
            with (patch.object(pool, 'OUT', dest), patch.object(pool, 'questions', return_value=rows),
                  patch.object(pool, 'c0_reference', return_value=sources)):
                self.assertEqual(pool.worker(args), 0)
            return self.phased.read_jsonl(dest / 'test/s202/pools_part0.jsonl')

        source_time = SourceTime.call

        async def source_time_call(adapter, *args, **kwargs):
            check_prefix(adapter, *args, **kwargs)
            return await source_time(adapter, *args, **kwargs)

        with patch.object(kwguard, 'EVENTS', events), patch.object(SourceTime, 'call', source_time_call):
            pools = pool_run('pool_guarded', wanted)
            with patch.object(kwguard, 'failure_reason', return_value=None):
                old_pools = pool_run('pool_unguarded_controls', self.good)
        bad_pool = pools[BAD_ID]
        self.assertEqual(bad_pool['status'], 'early_exit')
        self.assertEqual(bad_pool['keyword_guard'], self.fail_reason)
        for field in ('kept', 'final_full', 'graph_full', 'graph_scores', 'dense_ids', 'groups', 'boosted'):
            self.assertFalse(bad_pool[field], field)
        for qid in CONTROLS:
            self.assertEqual(pools[qid]['status'], 'ok')
            self.assertEqual({k: v for k, v in pools[qid].items() if k != 'retrieval_s'},
                             {k: v for k, v in old_pools[qid].items() if k != 'retrieval_s'})
        expected = {'query_id': BAD_ID, 'seed': 202, 'split': 'test', 'reason': self.fail_reason}
        self.assertEqual([json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()],
                         [expected, expected])


if __name__ == '__main__':
    if os.environ.get('PYTHONHASHSEED') != c0.HASH_SEED:
        command = [sys.executable, '-B', '-m', 'reproduce.mhr_phi.test_kwguard', *sys.argv[1:]]
        raise SystemExit(subprocess.call(command, cwd=p.ROOT,
                                        env=os.environ | {'PYTHONHASHSEED': c0.HASH_SEED, 'PYTHONUTF8': '1'}))
    unittest.main(verbosity=2)
