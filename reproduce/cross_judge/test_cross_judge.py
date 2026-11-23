"""Offline tests. The injected HTTP transport never opens a socket."""

import io
import json
import threading
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory

from . import (
    HISTORICAL, LIHUA_KEY, LIHUA_PARSER, MHR_KEY, Answer, Dataset, digest, journal_path,
    read_journal, source_symbols, unique_answers,
)
from .analyze import agreement, analyze_dataset, comparison, interjudge, quality
from .run import DEFAULT_BUDGET, BudgetStop, keys_for, plan_dataset, run_dataset

A, E, N, U = "accurate", "error", "neither", "unparsed"
FAKE_KEY = "test-key-123"


def synthetic(judge="deepseek", count=1):
    answers = {str(i): Answer(str(i), digest(f"prediction {i}"), f"question {i}", "gold", f"prediction {i}")
               for i in range(count)}
    if judge == "deepseek":
        dataset, model, parse = MHR_KEY, "gemini-3.5-flash-lite", HISTORICAL.parse_label
    else:
        dataset, model = LIHUA_KEY, "deepseek-flash"
        parse = source_symbols(LIHUA_PARSER, "parse", "LABELS").parse
    return Dataset(dataset, "Synthetic", model, judge, "{question}\n{gold}\n{pred}", parse,
                   {"1": {"CE": answers, "SET": dict(answers)}, "2": {"CE": dict(answers), "SET": dict(answers)}},
                   {config: {seed: [(A, A, A)] * count for seed in ("1", "2")} for config in ("CE", "SET")})


class FakeHTTP:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.lock = threading.Lock()

    def __call__(self, request, timeout):
        with self.lock:
            self.requests.append((json.loads(request.data), request.get_header("Authorization"), timeout))
            reply = self.replies.pop(0) if self.replies else A
        if isinstance(reply, int):
            raise urllib.error.HTTPError(request.full_url, reply, FAKE_KEY, {}, io.BytesIO())
        return io.BytesIO(json.dumps({
            "model": "test-model",
            "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 1, "prompt_cache_hit_tokens": 20},
        }).encode())


class CrossJudgeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)

    def execute(self, dataset, http, mode="run", limits=DEFAULT_BUDGET, sleep=None):
        delays = [] if sleep is None else sleep
        run_dataset(dataset, mode, (FAKE_KEY,), self.out, limits, http=http, sleep=delays.append)
        return read_journal(journal_path(dataset, self.out))

    def test_dedupe_resume_and_request_contract(self):
        dataset = synthetic(count=2)
        self.assertEqual(len(unique_answers(dataset)[0]), 2)
        self.assertEqual(plan_dataset(dataset)["planned_calls"], 6)
        first = FakeHTTP()
        self.assertEqual(len(self.execute(dataset, first, "smoke")), 2)
        second = FakeHTTP()
        rows = self.execute(dataset, second)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(second.requests), 4)
        self.assertEqual({r["judge_pass"] for r in rows}, {1, 2, 3})
        self.assertEqual(rows[0]["configs"], ["CE", "SET"])
        self.assertEqual(rows[0]["seed"], [1, 2])
        self.assertEqual(len(rows[0]["uses"]), 4)
        for body, auth, _ in first.requests:
            self.assertEqual(auth, f"Bearer {FAKE_KEY}")
            prompt = body["messages"][0]["content"]
            self.assertIn(prompt, {f"question {i}\ngold\nprediction {i}" for i in range(2)})
            self.assertEqual(body, {
                "model": "deepseek-flash", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16, "thinking": {"type": "disabled"},
            })
        resumed = FakeHTTP()
        self.assertEqual(self.execute(dataset, resumed), rows)
        self.assertEqual(resumed.requests, [])

    def test_smoke_selects_ten_and_gemini_uses_provider_defaults(self):
        dataset = synthetic("gemini", 12)
        http = FakeHTTP("unrecognised reply")
        rows = self.execute(dataset, http, "smoke")
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({row["id"] for row in rows}), 10)
        self.assertEqual({row["judge_pass"] for row in rows}, {1})
        body = http.requests[0][0]
        self.assertEqual(set(body), {"model", "messages"})
        self.assertEqual(body["model"], "gemini-3.5-flash-lite")
        self.assertEqual(next(row["verdict"] for row in rows if row["raw"] == "unrecognised reply"), N)

    def test_failures_retry_without_verdict_and_unparsed_is_preserved(self):
        dataset = synthetic()
        malformed = "malformed reply " + FAKE_KEY
        http, delays = FakeHTTP(429, 503, malformed, malformed, malformed), []
        rows = self.execute(dataset, http, "smoke", sleep=delays)
        # two HTTP retries, then three asks (judge_run.py --invalid-tries 3) before recording unparsed
        self.assertEqual(len(http.requests), 5)
        self.assertEqual(len(delays), 2)
        self.assertEqual(rows[0]["verdict"], U)
        self.assertIn("[redacted]", rows[0]["raw"])
        receipts = [row for row in read_journal(self.out / f"{MHR_KEY}_deepseek_calls.jsonl")
                    if row["event"] == "receipt"]
        self.assertEqual([r["ok"] for r in receipts], [False, False, False, False, True])
        self.assertEqual([r["status"] for r in receipts[2:4]], ["unparsed_reask"] * 2)
        self.assertEqual(receipts[-1]["usage"]["prompt_tokens"], 100)
        self.assertGreater(receipts[-1]["cost_usd"], 0)
        for file in self.out.glob("*.jsonl"):
            self.assertNotIn(FAKE_KEY, file.read_text(encoding="utf-8"))
        other = synthetic("gemini")
        with self.assertRaisesRegex(RuntimeError, "401"):
            self.execute(other, FakeHTTP(401), "smoke")
        self.assertFalse(journal_path(other, self.out).exists())
        with self.assertRaisesRegex(RuntimeError, "attempt 6"):
            self.execute(other, FakeHTTP(*([503] * 6)), "smoke")
        self.assertFalse(journal_path(other, self.out).exists())

    def test_unparsed_reask_recovers_a_label_on_multihop_only(self):
        rows = self.execute(synthetic(), FakeHTTP("no label here", E), "smoke")
        self.assertEqual(rows[0]["verdict"], E)
        lihua = synthetic("gemini")
        http = FakeHTTP("no label here", E)
        rows = self.execute(lihua, http, "smoke")
        self.assertEqual(len(http.requests), 1)  # LiHua parser falls back to neither; no re-ask
        self.assertEqual(rows[0]["verdict"], N)

    def test_receipt_recovers_a_missing_verdict_without_new_call(self):
        dataset = synthetic()
        rows = self.execute(dataset, FakeHTTP(), "smoke")
        # Simulate the restart window after a receipt was saved but before its verdict append.
        journal_path(dataset, self.out).unlink()
        http = FakeHTTP()
        self.assertEqual(self.execute(dataset, http, "smoke"), rows)
        self.assertEqual(http.requests, [])

    def test_budget_stops_before_dispatch_and_caps_retries_across_resume(self):
        limits = {"deepseek": {"max_usd": 0.000001}, "gemini": {"max_requests": 1}}
        deepseek = FakeHTTP()
        with self.assertRaisesRegex(BudgetStop, "USD guard"):
            self.execute(synthetic(), deepseek, "smoke", limits)
        self.assertEqual(deepseek.requests, [])
        gemini = FakeHTTP(429)
        with self.assertRaisesRegex(BudgetStop, "request cap"):
            self.execute(synthetic("gemini"), gemini, "smoke", limits)
        self.assertEqual(len(gemini.requests), 1)
        resumed = FakeHTTP()
        with self.assertRaisesRegex(BudgetStop, "request cap"):
            self.execute(synthetic("gemini"), resumed, "smoke", limits)
        self.assertEqual(resumed.requests, [])

    def test_keys_follow_environment_then_override_then_root(self):
        default, override = self.out / ".env", self.out / "elsewhere.env"
        default.write_text("DEEPSEEK_API_KEY=root-d\nGEMINI_API_KEY=root-g\n", encoding="utf-8")
        override.write_text("export DEEPSEEK_API_KEY='override-d'\n", encoding="utf-8")
        env = {"MINIRAG_DOTENV": str(override), "GOOGLE_API_KEY": "env-g"}
        self.assertEqual(keys_for(("deepseek", "gemini"), env, self.out),
                         {"deepseek": ("override-d",), "gemini": ("env-g",)})
        self.assertEqual(keys_for(("deepseek", "gemini"), {}, self.out),
                         {"deepseek": ("root-d",), "gemini": ("root-g",)})
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY.*GEMINI_API_KEY"):
            keys_for(("deepseek", "gemini"), {}, self.out / "missing-directory")

    def test_majority_pass_rates_and_paired_fallback(self):
        votes = [(A, A, E), (E, N, A), (U, U, A)]
        mhr = quality(synthetic(), votes)
        self.assertEqual(mhr["counts"], {A: 1, E: 0, N: 0, "no_majority": 1, U: 1})
        self.assertAlmostEqual(mhr["acc_pct"], 100 / 3)
        lihua = synthetic("gemini")
        self.assertAlmostEqual(quality(lihua, votes)["acc_pct"], 400 / 9)
        score = comparison(lihua, [(A, E, N)], [(E, A, N)])
        self.assertEqual(score["delta_sign"], 0)
        self.assertEqual(score["mcnemar"]["down"], 1)
        self.assertEqual(score["mcnemar"]["p_exact"], 1)
        self.assertAlmostEqual(comparison(synthetic(), [(E, E, E)] * 4, [(A, A, A)] * 4)
                               ["mcnemar"]["p_exact"], 0.125)

    def test_agreement_kappas_and_no_majority_bucket(self):
        a, e, n, split, unparsed = (A, A, A), (E, E, E), (N, N, N), (A, E, N), (U, U, N)
        result = interjudge([(a, a), (a, e), (e, e), (n, n), (split, a), (e, unparsed), (split, split)])
        self.assertEqual(result["confusion_3x3"], [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(result["confusion_with_no_majority"][-1], [1, 0, 0, 1])
        self.assertEqual(result["agreement_valid_pct"], 75)
        self.assertAlmostEqual(result["agreement_all_pct"], 300 / 7)
        self.assertAlmostEqual(result["cohen_kappa"], 7 / 11)
        self.assertIsNone(interjudge([(a, a)])["cohen_kappa"])
        self_result = agreement([a, (E, E, N), unparsed])
        self.assertEqual(self_result["excluded_unparsed_rows"], 1)
        self.assertEqual(self_result["unanimous_pct"], 50)
        self.assertAlmostEqual(self_result["fleiss_kappa"], 5 / 11)

    def test_analysis_requires_three_passes_and_pools_answer_instances(self):
        dataset = synthetic()
        self.execute(dataset, FakeHTTP(), "smoke")
        result = analyze_dataset(dataset, self.out, {})
        self.assertEqual(result["journaled_calls"], 1)
        self.assertEqual(result["pooled"]["inter_judge"]["n_total"], 0)
        self.assertFalse(result["seeds"]["1"]["complete"])
        self.execute(dataset, FakeHTTP())
        result = analyze_dataset(dataset, self.out, {})
        self.assertEqual(result["journaled_calls"], 3)
        self.assertEqual(result["pooled"]["inter_judge"]["n_total"], 4)
        self.assertTrue(result["seeds"]["1"]["complete"])
        self.assertTrue(result["seeds"]["1"]["delta_sign_agrees"])


if __name__ == "__main__":
    unittest.main()
