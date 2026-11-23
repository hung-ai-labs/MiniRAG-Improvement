"""Plan, smoke-test or run the swapped judges. Only smoke and run use the network."""

import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from http.client import HTTPException
from pathlib import Path

from reproduce.mhr_st.common import write

from . import (
    DEEPSEEK, GEMINI_UTILS, MHR_KEY, MODELS, OUT, PASSES, ROOT, call_key, digest, journal_path,
    load_datasets, read_journal, read_json, source_symbols, unique_answers, verdict_index,
)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
URLS = {"deepseek": DEEPSEEK.URL, "gemini": GEMINI_URL}
DEFAULT_BUDGET = {"deepseek": {"max_usd": 15.0}, "gemini": {"max_requests": 10000}}
GEMINI_KEY = re.compile(r"GEMINI_API_KEY(?:_\d+)?|GEMINI_API_KEYS|GOOGLE_API_KEY")
KEY_NAMES = {
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY (or GEMINI_API_KEYS, GEMINI_API_KEY_<number>, GOOGLE_API_KEY)",
}
DEEPSEEK_MAX_TOKENS = 16
GEMINI_OUTPUT_ESTIMATE = 2
WORKERS = 4
MAX_ATTEMPTS = 6
INVALID_TRIES = 3
ORDER_SEED = 1309
DECODE_GEMINI = source_symbols(GEMINI_UTILS, "safe_unicode_decode", re=re).safe_unicode_decode


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def prompt_for(dataset, answer):
    return dataset.template.format(question=answer.question, gold=answer.gold, pred=answer.prediction)


def request_body(judge, prompt):
    body = {"model": MODELS[judge], "messages": [{"role": "user", "content": prompt}]}
    if judge == "deepseek":
        body.update(max_tokens=DEEPSEEK_MAX_TOKENS, thinking={"type": "disabled"})
    return body


def plan_dataset(dataset):
    answers, uses = unique_answers(dataset)
    n_calls = len(answers) * len(PASSES)
    input_tokens = len(PASSES) * sum(
        (len(prompt_for(dataset, answer).encode("utf-8")) + 3) // 4 + 8 for answer in answers.values()
    )
    output_per_call = DEEPSEEK_MAX_TOKENS if dataset.judge == "deepseek" else GEMINI_OUTPUT_ESTIMATE
    output_tokens = n_calls * output_per_call
    result = {
        "dataset": dataset.name, "judge_model": MODELS[dataset.judge],
        "unique_answers": len(answers), "answer_instances": sum(map(len, uses.values())),
        "planned_calls": n_calls, "smoke_calls": min(len(answers), 10),
        "input_tokens_est": input_tokens, "output_tokens_est": output_tokens,
        "output_tokens_per_call_est": output_per_call, "prompt_sha256": digest(dataset.template),
        "price_usd_per_million": None, "estimated_usd": None,
    }
    if dataset.judge == "deepseek":
        result.update(price_usd_per_million=DEEPSEEK.PRICE, estimated_usd=DEEPSEEK.cost({
            "prompt_tokens": input_tokens, "completion_tokens": output_tokens,
        }))
    return result


def plan(datasets, sources, out):
    by_judge = {d.judge: plan_dataset(d) for d in datasets}
    report = {
        "passes": list(PASSES),
        "configs": ["CE", "SET"],
        "input_estimate": "Sum ceil(UTF-8 prompt bytes / 4) + 8 chat framing tokens per call.",
        "output_estimate": "16 (the request cap) per DeepSeek call, 2 visible label tokens per Gemini call. "
                           "Gemini provider reasoning, retries and input cache discounts are excluded.",
        "gemini_price": "unknown (no price for this model found in the repository)",
        "prompts_identical": len({d.template for d in datasets}) == 1 if len(datasets) == 2 else None,
        "judges": by_judge,
        "source_sha256": dict(sorted(sources.items())),
    }
    write(out / "plan.json", report)
    print(report["input_estimate"], report["output_estimate"])
    for value in by_judge.values():
        cost = value["estimated_usd"]
        usd = f"${cost:.4f}" if cost is not None else "unknown"
        print(f"{value['dataset']} / {value['judge_model']}: {value['unique_answers']:,} unique answers, "
              f"{value['planned_calls']:,} calls, {value['input_tokens_est']:,} input tokens, "
              f"{value['output_tokens_est']:,} output tokens, USD {usd}")


def dotenv_keys(path):
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line.startswith("export ") or line.startswith("export\t"):
                line = line.split(maxsplit=1)[1]
            name, equals, value = line.partition("=")
            name = name.strip()
            if not equals or not (name == "DEEPSEEK_API_KEY" or GEMINI_KEY.fullmatch(name)):
                continue
            value = value.strip()
            if value.startswith(("'", '"')):
                value = value[1:].partition(value[0])[0]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0]
            if value:
                values[name] = value
    return values


def keys_for(judges, env=os.environ, root=ROOT):
    keys = {}
    override = env.get("MINIRAG_DOTENV")
    paths = [Path(override)] if override else []
    paths.append(root / ".env")
    for source in (None, *paths):
        if len(keys) == len(judges):
            break
        values = env if source is None else dotenv_keys(source)
        deepseek = values.get("DEEPSEEK_API_KEY", "").strip()
        if deepseek and "deepseek" in judges:
            keys.setdefault("deepseek", (deepseek,))
        if "gemini" in judges and "gemini" not in keys:
            names = [name for name in values if GEMINI_KEY.fullmatch(name)]
            gemini = tuple(dict.fromkeys(
                part.strip() for name in sorted(names) for part in values[name].split(",") if part.strip()
            ))
            if gemini:
                keys["gemini"] = gemini
    missing = [KEY_NAMES[judge] for judge in judges if judge not in keys]
    if missing:
        raise ValueError("Missing API key(s): " + "; ".join(missing))
    return keys


def append(path, row):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def run_lock(out):
    """Serialize live runs so concurrent invocations share one budget and journal."""
    with (out / ".run.lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Another cross-judge run holds the output lock") from None
        yield


class BudgetStop(RuntimeError):
    pass


def budget_limits(out):
    path = out / "approved_budget.json"
    limits = read_json(path)
    usd = limits["deepseek"]["max_usd"]
    requests = limits["gemini"]["max_requests"]
    if (type(usd) not in (float, int) or not 0 < usd <= DEFAULT_BUDGET["deepseek"]["max_usd"]
            or type(requests) is not int or requests < 1):
        raise ValueError("approved_budget.json needs 0 < deepseek.max_usd <= 15 and a positive gemini.max_requests")
    return limits


class Ledger:
    """Journal attempts before dispatch and reconcile them with durable receipts."""

    def __init__(self, path, judge, limits):
        self.path = path
        self.judge = judge
        self.limits = limits[judge]
        self.ready = threading.Condition()
        self.started, self.receipts = {}, {}
        self.successes = {}
        self.active = {}
        self.spent = 0.0
        for row in read_journal(path):
            request_id = row["request_id"]
            if row["event"] == "started" and request_id == len(self.started) + 1:
                self.started[request_id] = row
            elif row["event"] == "receipt" and request_id in self.started and request_id not in self.receipts:
                self.receipts[request_id] = row
                self.spent += row["budget_charge_usd"]
                if row["ok"]:
                    verdict = row["result"]
                    key = call_key(verdict)
                    if key in self.successes or key != call_key(self.started[request_id]):
                        raise ValueError(f"Duplicate or mismatched successful receipt in {path.name}")
                    self.successes[key] = verdict
            else:
                raise ValueError(f"Unexpected attempt event in {path.name}")
        # A crash or timeout can leave an already billable request unconfirmed.
        self.spent += sum(row["reserved_usd"] for rid, row in self.started.items() if rid not in self.receipts)

    def start(self, answer, judge_pass, payload, prompt, attempt):
        reserve = 0.0
        if self.judge == "deepseek":
            reserve = DEEPSEEK.cost({"prompt_tokens": len(prompt.encode("utf-8")) + 32,
                                     "completion_tokens": DEEPSEEK_MAX_TOKENS})
        with self.ready:
            if self.judge == "deepseek":
                limit = self.limits["max_usd"]
                while self.spent + reserve <= limit < self.spent + reserve + sum(self.active.values()):
                    self.ready.wait()
                if self.spent + reserve > limit:
                    raise BudgetStop(f"DeepSeek USD guard reached (${self.spent:.4f} charged or reserved)")
            elif len(self.started) >= self.limits["max_requests"]:
                raise BudgetStop(f"Gemini request cap reached ({len(self.started):,} HTTP attempts)")
            rid = len(self.started) + 1
            row = {
                "event": "started", "request_id": rid, "attempt": attempt,
                "id": answer.id, "prediction_sha256": answer.prediction_sha256, "judge_pass": judge_pass,
                "request_sha256": sha256(payload).hexdigest(), "reserved_usd": reserve, "timestamp": timestamp(),
            }
            append(self.path, row)
            self.started[rid] = row
            self.active[rid] = reserve
            return rid

    def finish(self, rid, receipt, verdict_path):
        with self.ready:
            reserve = self.active[rid]
            measured = receipt["cost_usd"]
            charge = measured if measured is not None else reserve
            row = {"event": "receipt", "request_id": rid, "timestamp": timestamp(),
                   "budget_charge_usd": charge, **receipt}
            try:
                append(self.path, row)
                if row["ok"]:
                    append(verdict_path, row["result"])
            finally:
                self.spent += charge
                del self.active[rid]
                self.ready.notify_all()


def verdict_row(dataset, answer, uses, judge_pass, verdict, raw):
    seeds = sorted({int(seed) for seed, _ in uses})
    return {
        "dataset": dataset.key, "configs": sorted({config for _, config in uses}),
        "seed": seeds[0] if len(seeds) == 1 else seeds,
        "uses": [{"seed": int(seed), "config": config} for seed, config in uses],
        "id": answer.id, "prediction_sha256": answer.prediction_sha256,
        "judge_model": MODELS[dataset.judge], "judge_pass": judge_pass,
        "verdict": verdict, "raw": raw, "timestamp": timestamp(),
    }


def redact(text, keys):
    if not isinstance(text, str):
        return None
    for key in keys:
        text = text.replace(key, "[redacted]")
    return text


def token_usage(response):
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    names = ("prompt_tokens", "completion_tokens", "total_tokens",
             "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
    return {name: usage[name] for name in names if type(usage.get(name)) is int and usage[name] >= 0}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def worker_request(req, timeout):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    return opener.open(req, timeout=timeout)


def judge_call(dataset, answer, uses, judge_pass, keys, ledger, path, http, sleep):
    prompt = prompt_for(dataset, answer)
    payload = json.dumps(request_body(dataset.judge, prompt)).encode("utf-8")
    judge = dataset.judge
    # The historical MultiHop-RAG judge re-asks an unparseable reply up to INVALID_TRIES times
    # before recording a failed parse (judge_run.py --invalid-tries 3); the LiHua parser never
    # returns None, so this only applies there. Re-asks do not consume HTTP retry attempts.
    invalid = 0
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        rid = ledger.start(answer, judge_pass, payload, prompt, attempt)
        request = urllib.request.Request(
            URLS[judge], data=payload,
            headers={"Authorization": "Bearer " + keys[(rid - 1) % len(keys)],
                     "Content-Type": "application/json"},
        )
        start = time.monotonic()
        receipt = {"ok": False, "usage": {}, "cost_usd": None}
        retry = True
        try:
            with http(request, timeout=120) as response:
                body = json.loads(response.read())
            if body.get("error"):
                raise ValueError("Provider error in successful HTTP response")
            usage = token_usage(body)
            receipt["usage"] = usage
            if judge == "deepseek" and {"prompt_tokens", "completion_tokens"} <= usage.keys():
                usage_for_cost = dict(usage)
                if usage.get("prompt_cache_hit_tokens", 0) > usage["prompt_tokens"]:
                    usage_for_cost["prompt_cache_hit_tokens"] = 0
                receipt["cost_usd"] = DEEPSEEK.cost(usage_for_cost)
            choices = body.get("choices")
            first = {} if judge == "gemini" and not choices else choices[0]
            raw = first.get("message", {}).get("content") or ""
            if judge == "gemini" and r"\u" in raw:
                raw = DECODE_GEMINI(raw.encode("utf-8"))
            verdict = dataset.parser(raw) or "unparsed"
            if verdict == "unparsed" and dataset.key == MHR_KEY and invalid + 1 < INVALID_TRIES:
                invalid += 1
                attempt -= 1
                receipt.update(status="unparsed_reask", raw=redact(raw, keys))
                receipt["latency_s"] = time.monotonic() - start
                ledger.finish(rid, receipt, path)
                continue
            receipt.update(ok=True, status=200, provider_model=redact(body.get("model"), keys),
                           finish_reason=redact(first.get("finish_reason"), keys),
                           result=verdict_row(dataset, answer, uses, judge_pass, verdict, redact(raw, keys)))
        except urllib.error.HTTPError as exc:
            receipt["status"] = exc.code
            retry = exc.code == 429 or 500 <= exc.code <= 599
            if exc.code < 500:
                receipt["cost_usd"] = 0.0 if judge == "deepseek" else None
            exc.close()
        except (OSError, HTTPException, ValueError, TypeError, AttributeError, LookupError) as exc:
            # Exception strings and error bodies can contain authorization details.
            receipt["status"] = type(exc).__name__
        receipt["latency_s"] = time.monotonic() - start
        ledger.finish(rid, receipt, path)
        if receipt["ok"]:
            return
        if not retry or attempt == MAX_ATTEMPTS:
            raise RuntimeError(f"{MODELS[judge]} request failed ({receipt['status']}, attempt {attempt}); "
                               "rerun to resume")
        sleep(min(60, 2 ** attempt) + random.random())


def run_dataset(dataset, mode, keys, out, limits, http=worker_request, sleep=time.sleep):
    answers, uses = unique_answers(dataset)
    path = journal_path(dataset, out)
    done = verdict_index(dataset, answers, path)
    ledger = Ledger(path.with_name(path.stem + "_calls.jsonl"), dataset.judge, limits)
    recovered = verdict_index(dataset, answers, ledger.path, ledger.successes.values())
    for key, row in recovered.items():
        if key not in done:
            append(path, row)
            done[key] = row
        elif done[key] != row:
            raise ValueError(f"Receipt and verdict differ in {path.name}")
    ordered = sorted(answers)
    rng = random.Random(ORDER_SEED)
    if mode == "smoke":
        rng.shuffle(ordered)
        ordered = ordered[:10]
    todo = [(key, judge_pass) for key in ordered for judge_pass in ((1,) if mode == "smoke" else PASSES)
            if (*key, judge_pass) not in done]
    rng.shuffle(todo)
    print(f"{dataset.name} / {MODELS[dataset.judge]}: {len(todo):,} calls pending ({mode})", flush=True)
    if not todo:
        return

    # Keep at most WORKERS in flight and stop submitting when any one fails.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = iter(todo)

        def submit(item):
            key, judge_pass = item
            return pool.submit(judge_call, dataset, answers[key], uses[key], judge_pass, keys, ledger, path, http, sleep)

        in_flight = {submit(next(pending)) for _ in range(min(WORKERS, len(todo)))}
        completed = 0
        while in_flight:
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
            completed += len(finished)
            if completed == len(todo) or completed // 500 > (completed - len(finished)) // 500:
                print(f"{dataset.name}: {completed:,}/{len(todo):,} verdicts written", flush=True)
            for _ in finished:
                item = next(pending, None)
                if item is not None:
                    in_flight.add(submit(item))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "smoke", "run"))
    parser.add_argument("--judge", choices=("deepseek", "gemini", "both"), required=True)
    args = parser.parse_args(argv)
    judges = tuple(MODELS) if args.judge == "both" else (args.judge,)
    try:
        keys = keys_for(judges) if args.mode != "plan" else {}
        sources = {}
        datasets = load_datasets(sources)
        selected = [datasets[judge] for judge in judges]
        if args.mode == "plan":
            plan(selected, sources, OUT)
            if not (OUT / "approved_budget.json").exists():
                write(OUT / "approved_budget.json", DEFAULT_BUDGET)
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            with run_lock(OUT):
                if not (OUT / "approved_budget.json").exists():
                    write(OUT / "approved_budget.json", DEFAULT_BUDGET)
                limits = budget_limits(OUT)
                for dataset in selected:
                    run_dataset(dataset, args.mode, keys[dataset.judge], OUT, limits)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cross-judge {args.mode}: {exc}\n")


if __name__ == "__main__":
    main()
