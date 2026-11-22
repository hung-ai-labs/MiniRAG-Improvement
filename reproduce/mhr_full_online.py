"""Generation and judging of the development cohort for the full evaluation."""
from __future__ import annotations

import os

from reproduce import mhr_full as full
from reproduce.mhr_st.common import read, records, sha


def setup():
    from reproduce.mhr_set import deferred, offline

    offline.guard(network=True)
    # Redirect the existing operators' serving log and judge paths to this run.
    deferred.r.OUT = full.OUT
    return deferred


def inputs():
    whole, dev, _ = full.questions()
    path = full.OUT / "dev_contexts.jsonl"
    manifest = read(full.OUT / "context_manifest.json")
    if manifest["generation_seed"] != full.SEED or sha(path) != manifest["contexts_sha256"]:
        raise ValueError("Prepared contexts do not match the fixed evaluation")
    context_rows = full.keyed(records(path))
    if set(context_rows) != set(dev) or any(set(r["contexts"]) != set(full.ARMS) for r in context_rows.values()):
        raise ValueError("Incomplete contexts for an evaluation configuration")
    return {qid: whole[qid] for qid in whole if qid in dev}, context_rows


def distinct_rows(dev, contexts):
    inputs, mapping, values = [], {}, {}
    for qid, row in dev.items():
        unique = {}
        mapping[qid] = {}
        for arm in full.ARMS:
            context = contexts[qid]["contexts"][arm]
            if context not in unique:
                key = f"{qid}-ctx{len(unique)}"
                unique[context] = key
                inputs.append(row | {"query_id": key})
                values[key] = {"status": "ok", "context": context}
            mapping[qid][arm] = unique[context]
    return inputs, values, mapping


def answers():
    dev, contexts = inputs()
    rows, values, _ = distinct_rows(dev, contexts)
    destination = full.OUT / "generation"
    path = destination / "answers.jsonl"
    cached = full.keyed(records(path))
    if set(cached) - set(values):
        raise ValueError("Generation journal contains unrelated questions")
    pending = [row for row in rows if row["query_id"] not in cached]
    if pending:
        missing = [name for name in ("MINIRAG_SLM_URL", "MINIRAG_SLM_KEY") if not os.environ.get(name)]
        if missing:
            raise SystemExit("Qwen serving environment is missing: " + ", ".join(missing))
        from reproduce import modal_account
        modal_account.check({"MINIRAG_SLM_URL": os.environ["MINIRAG_SLM_URL"]})
        operator = setup()
        import phased_qa as phased
        from minirag.base import QueryParam

        destination.mkdir(parents=True, exist_ok=True)
        phased.apply_env("C0", full.SEED)
        operator.configure_slm(phased, full.SEED)
        rag = phased.build(destination / "work", llm_max_async=32)
        phased.traced(destination, lambda: phased.answer_phase(rag.llm_model_func, rows, values, "mini", path,
                                                                QueryParam().response_type))
    if set(full.keyed(records(path))) != set(values):
        raise ValueError("Generation incomplete")
    print(f"Generated {len(values)} distinct contexts for {len(dev)} questions across five configurations.")


def judge(rounds):
    if rounds < 1:
        raise ValueError("At least one judge attempt round is required")
    dev, contexts = inputs()
    _, values, mapping = distinct_rows(dev, contexts)
    answers = full.keyed(records(full.OUT / "generation/answers.jsonl"))
    if set(answers) != set(values):
        raise ValueError("Run the complete generation phase before judging")
    question_rows, answer_rows, by_arm = [], [], {}
    for qid, row in dev.items():
        unique = {}
        by_arm[qid] = {}
        for arm, key in mapping[qid].items():
            prediction = answers[key]["answer"]
            if prediction not in unique:
                key = f"{qid}-answer{len(unique)}"
                unique[prediction] = key
                question_rows.append(row | {"query_id": key})
                answer_rows.append({"Question": row["Question"], "Gold Answer": row["Gold Answer"],
                                    "minirag": prediction})
            by_arm[qid][arm] = unique[prediction]
    operator = setup()
    destination = full.OUT / "judging"
    operator.ensure_csv(destination / "questions.csv", list(question_rows[0]), question_rows)
    operator.ensure_csv(destination / "answers.csv", list(answer_rows[0]), answer_rows)
    run_dir = destination / "run"
    operator.run_judge(destination / "questions.csv", destination / "answers.csv", run_dir, rounds)
    records_by_pass = {}
    for row in records(run_dir / "judge.jsonl"):
        key = (row["query_id"], row["pass"])
        if key in records_by_pass or row["status"] not in {"ok", "unparsed"}:
            raise ValueError(f"Invalid judge record for {key}")
        records_by_pass[key] = row["label"]
    result = []
    for qid in dev:
        verdicts = {arm: full.majority([records_by_pass[key, p] for p in (1, 2, 3)])
                    for arm, key in by_arm[qid].items()}
        result.append({"query_id": qid, "verdicts": verdicts})
    path = full.OUT / "dev_verdicts.jsonl"
    if not path.exists():
        operator.offline.jsonl(path, result)
    elif records(path) != result:
        raise ValueError("Stored dev verdicts differ")
    full.summarize()
