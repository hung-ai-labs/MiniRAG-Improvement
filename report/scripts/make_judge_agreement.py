"""Measure agreement from the saved verdicts behind tab:both-datasets."""

import csv
import hashlib
import json
from collections import Counter
from math import isclose
from statistics import mean

from make_selection_tables import ARMS, LIHUA, MULTIHOP, OUTPUT, REPORT, REVIEW, number

ROOT = REPORT.parent
PASSES = (1, 2, 3)
LABELS = ("accurate", "error", "neither")
PASS_FIELDS = tuple(f"pass{p}" for p in PASSES)
MHR_TEST = REVIEW / "mhr_st_20261112/inputs/test.csv"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def unique(rows, key, path):
    indexed = {key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate row key in {path}")
    return indexed


def source_hash(path, sources):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sources[path.relative_to(ROOT).as_posix()] = digest
    return digest


def agreement(votes):
    # Same categories, exclusions and formula as reproduce/mhr_set/reporting.py:kappa.
    valid = [row for row in votes if "unparsed" not in row]
    n = len(valid)
    if any(label not in LABELS for row in valid for label in row):
        raise ValueError("Fleiss kappa requires three valid labels per included answer")
    unanimous = sum(len(set(row)) == 1 for row in valid)
    observed = expected = kappa = None
    if n:
        observed = sum((sum(row.count(c) ** 2 for c in LABELS) - 3) / 6 for row in valid) / n
        proportions = [sum(row.count(c) for row in valid) / (3 * n) for c in LABELS]
        expected = sum(p * p for p in proportions)
        if expected < 1:
            kappa = (observed - expected) / (1 - expected)
    return {
        "n_total": len(votes),
        "n": n,
        "excluded_unparsed_rows": len(votes) - n,
        "unanimous": unanimous,
        "unanimous_pct": 100 * unanimous / n if n else None,
        "observed_agreement": observed,
        "chance_agreement": expected,
        "fleiss_kappa": kappa,
    }


def lihua_votes(sources):
    report = read_json(LIHUA)
    source_hash(LIHUA, sources)
    judge = report["judge"]
    contexts_path = LIHUA.with_name("contexts.jsonl")
    answers_path = LIHUA.with_name("answers.jsonl")
    judgments_path = LIHUA.with_name("judgments_deepseek.jsonl")
    if source_hash(contexts_path, sources) != report["build"]["contexts_sha256"]:
        raise ValueError("LiHua-World contexts differ from the table inputs")
    for path in (answers_path, judgments_path):
        source_hash(path, sources)
    ids = sorted({row["id"] for row in read_jsonl(contexts_path)})
    if len(ids) != report["questions"] or tuple(report["judge_passes"]) != PASSES:
        raise ValueError("Unexpected LiHua-World question or judge-pass coverage")
    passes = read_jsonl(judgments_path)
    if any(row["judge_model"] != judge or row["judge_pass"] not in PASSES for row in passes):
        raise ValueError("Unexpected LiHua-World judge model or pass")
    judgments = unique(passes, lambda r: (r["id"], r["prediction_sha256"], r["judge_pass"]), judgments_path)
    answers = unique(read_jsonl(answers_path), lambda r: (r["round"], r["arm"], r["id"]), answers_path)
    by_arm = {}
    for arm, alias in ARMS:
        by_round = {}
        for run in report["rounds"]:
            votes = []
            for qid in ids:
                digest = answers[run, alias, qid]["prediction_sha256"]
                votes.append(tuple(judgments[qid, digest, p]["verdict"] for p in PASSES))
            by_round[str(run)] = votes
        # The original table averages the label rates, with every pass and answer in its denominator.
        for label in LABELS:
            rates = [round(100 * sum(row.count(label) for row in votes) / (len(PASSES) * len(ids)), 2)
                     for votes in by_round.values()]
            if rates != report["table"]["all"][alias][label]["per_round"]:
                raise ValueError(f"LiHua-World label rates differ from the table inputs for {arm}")
        by_arm[arm] = by_round
    return judge, by_arm


def multihop_votes(sources):
    report_path = MULTIHOP / "RESULTS.json"
    report = read_json(report_path)
    source_hash(report_path, sources)
    source_hash(MHR_TEST, sources)
    questions = unique(read_csv(MHR_TEST), lambda r: r["query_id"], MHR_TEST)
    if len(questions) != report["n"]:
        raise ValueError("MultiHop-RAG test questions differ from the table inputs")
    # Use the full per-question exports hashed by RESULTS.json. The delta journals
    # cover only newly judged answers, and the full-run journal covers development answers.
    manifest = {name.replace("\\", "/"): digest for name, digest in report["input_sha256"].items()}
    by_arm = {}
    for arm, _ in ARMS:
        by_seed = {}
        for seed, seed_report in report["seeds"].items():
            suffix = f"/test/{arm}_s{seed}/judge_gemini35/per_question.csv"
            name, = (name for name in manifest if name.endswith(suffix))
            path = ROOT / name
            if source_hash(path, sources) != manifest[name]:
                raise ValueError(f"MultiHop-RAG verdicts differ from the table inputs in {name}")
            rows = unique(read_csv(path), lambda r: r["query_id"], path)
            if rows.keys() != questions.keys() or any(r["Type"] != questions[q]["Type"] for q, r in rows.items()):
                raise ValueError(f"Unexpected MultiHop-RAG question coverage in {name}")
            votes = [tuple(row[field] for field in PASS_FIELDS) for row in rows.values()]
            majority = Counter(row["majority"] for row in rows.values())
            if any(not isclose(100 * majority[label] / len(rows), rate)
                   for label, rate in seed_report["quality"][arm].items()):
                raise ValueError(f"MultiHop-RAG label rates differ from the table inputs in {name}")
            if arm == "SET":
                saved = report["secondary_checks"]["fleiss_kappa_SET"][seed]["all"]
                computed = agreement(votes)
                if any(not isclose(computed[key], value) for key, value in saved.items()):
                    raise ValueError(f"SET agreement differs from RESULTS.json for seed {seed}")
            by_seed[seed] = votes
        by_arm[arm] = by_seed
    return report["judge"], by_arm


def summarize(judge, seed_unit, by_arm, sources):
    return {
        "judge_model": judge,
        "seed_unit": seed_unit,
        "source_sha256": dict(sorted(sources.items())),
        "configs": {
            arm: {
                "by_seed": {seed: agreement(votes) for seed, votes in by_seed.items()},
                "pooled": agreement([row for votes in by_seed.values() for row in votes]),
            }
            for arm, by_seed in by_arm.items()
        },
    }


def render(datasets):
    lihua, multihop = datasets.values()
    lines = [
        r"\begin{tabular}{lrrrrrr}\toprule",
        r" & \multicolumn{3}{c}{\textbf{LiHua-World}} & \multicolumn{3}{c}{\textbf{MultiHop-RAG}}\\",
        rf" & \multicolumn{{3}}{{c}}{{\texttt{{{lihua['judge_model']}}}}}"
        rf" & \multicolumn{{3}}{{c}}{{\texttt{{{multihop['judge_model']}}}}}\\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Cấu hình & $n$ & Trùng nhãn (\%) & $\kappa$ & $n$ & Trùng nhãn (\%) & $\kappa$\\\midrule",
    ]
    for arm, _ in ARMS:
        cells = [arm if arm == "C0" else "+" + arm]
        for dataset in datasets.values():
            pooled = dataset["configs"][arm]["pooled"]
            cells.append(str(pooled["n"]))
            for key, digits in (("unanimous_pct", 2), ("fleiss_kappa", 3)):
                value = pooled[key]
                cells.append("--" if value is None else number(value, digits))
        lines.append(" & ".join(cells) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    datasets = {}
    for name, seed_unit, read_votes in (
        ("LiHua-World", "generation round (request seed 13 + question id + 100000 * (round - 1), if generated)",
         lihua_votes),
        ("MultiHop-RAG", "generation seed", multihop_votes),
    ):
        sources = {}
        judge, by_arm = read_votes(sources)
        datasets[name] = summarize(judge, seed_unit, by_arm, sources)
    result = {
        "judge_passes": list(PASSES),
        "counting_unit": "One row per question per configuration and generation run. "
                         "Shared verdicts count in every configuration and run that uses them.",
        "denominator": "n_total includes all table answers; n excludes rows with any unparsed pass.",
        "pooling": "Combine answer rows across generation runs before computing each metric.",
        "datasets": datasets,
    }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "judge_agreement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "judge_agreement.tex").write_text(render(datasets), encoding="utf-8")


if __name__ == "__main__":
    main()
