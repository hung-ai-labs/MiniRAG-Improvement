"""Compare the original verdicts with the swapped judge's saved verdicts offline."""

import argparse
import re
from collections import Counter
from math import isclose
from statistics import mean, stdev

from reproduce.mhr_st.common import write

from . import (
    ARMS, CONFIGS, HISTORICAL, LABELS, LIHUA, LIHUA_KEY, LIHUA_MAJORITY, MAIN_TABLE, MHR_BINARY,
    MHR_REPORT, MODELS, OUT, OUTPUT, PASSES, ROOT, agreement, journal_path, load_datasets, number,
    read_json, read_journal, source_hash, unique_answers, verdict_index,
)

NO_MAJORITY = "no_majority"
MAJORITIES = (*LABELS, NO_MAJORITY)
QUALITY_LABELS = (*MAJORITIES, "unparsed")


def quality(dataset, votes):
    """Return both label counts and unrounded percentages under the dataset rule."""
    if dataset.key == LIHUA_KEY:
        # Take the mean of the three pass percentages as in pipeline.report().
        counts = [Counter(column) for column in zip(*votes)]
        total = Counter(label for row in votes for label in row)
        rates = {label: mean(100 * count[label] / len(votes) for count in counts) if votes else None
                 for label in QUALITY_LABELS}
        denominator = len(votes) * len(PASSES)
    else:
        total = Counter(map(HISTORICAL.majority, votes))
        rates = {label: 100 * total[label] / len(votes) if votes else None for label in QUALITY_LABELS}
        denominator = len(votes)
    return {"n_questions": len(votes), "denominator": denominator,
            "counts": {label: total[label] for label in QUALITY_LABELS},
            "acc_pct": rates["accurate"], "err_pct": rates["error"],
            "neither_pct": rates["neither"], "no_majority_pct": rates[NO_MAJORITY],
            "unparsed_pct": rates["unparsed"]}


def check_equal(actual, expected, context):
    if actual != expected:
        raise AssertionError(f"{context}: recomputed {actual!r}, expected {expected!r}")


def original_sanity(datasets, sources):
    """Check all five configurations per seed and against the rendered ACC/ERR cells."""
    lihua = read_json(LIHUA)["table"]["all"]
    multihop = read_json(MHR_REPORT)
    source_hash(MAIN_TABLE, sources)
    table = {}
    for line in MAIN_TABLE.read_text(encoding="utf-8").splitlines():
        if line.startswith(("C0 &", "+")):
            name, *cells = line.removesuffix(r"\\").split(" & ")
            table[name.lstrip("+")] = [re.sub(r"\\textbf\{([^{}]+)\}", r"\1", cell) for cell in cells]
    check_equal(set(table), {config for config, _ in ARMS}, "tab:both-datasets configurations")
    sanity = {}
    rendered = {config: [] for config, _ in ARMS}
    for dataset in sorted(datasets, key=lambda d: d.key):
        is_lihua = dataset.key == LIHUA_KEY
        by_config = {}
        for config, alias in ARMS:
            stats = {seed: quality(dataset, votes) for seed, votes in dataset.original_votes[config].items()}
            means = {}
            for label, metric in (("accurate", "acc_pct"), ("error", "err_pct")):
                rates = [row[metric] for row in stats.values()]
                average, sd = mean(rates), stdev(rates)
                if is_lihua:
                    expected = lihua[alias][label]
                    check_equal([round(rate, 2) for rate in rates], expected["per_round"],
                                f"{dataset.name}/{config}/{label} round percentages")
                    check_equal((round(average, 2), round(sd, 2)), (expected["mean"], expected["sd_rounds"]),
                                f"{dataset.name}/{config}/{label} table mean and SD")
                    average, sd = round(average, 2), round(sd, 2)
                else:
                    expected = multihop["generation_mean_sd"][config][label]
                    for seed, row in stats.items():
                        check_equal(row[metric], multihop["seeds"][seed]["quality"][config][label],
                                    f"{dataset.name}/{config}/{seed}/{label}")
                    check_equal(average, expected["mean"], f"{dataset.name}/{config}/{label} table mean")
                    # statistics.stdev differs by one ULP across Python versions.
                    if not isclose(sd, expected["sd"], rel_tol=0, abs_tol=1e-12):
                        check_equal(sd, expected["sd"], f"{dataset.name}/{config}/{label} table SD")
                rendered[config].append(number(average) + r"\,$\pm$\," + number(sd))
                means[label] = {"mean": average, "sd": sd}
            by_config[config] = {"by_seed": stats, "generation_mean_sd": means}
        sanity[dataset.key] = {"judge_model": dataset.original_model, "configs": by_config}
    for config, cells in rendered.items():
        check_equal(cells, table[config], f"tab:both-datasets/{config}")
    return sanity


def agreement_majority(votes):
    label = HISTORICAL.majority(votes)
    return label if label in LABELS else NO_MAJORITY


def interjudge(vote_pairs):
    """Count consensus matches with and without the rows lacking valid majorities."""
    counts = Counter((agreement_majority(deepseek), agreement_majority(gemini))
                     for deepseek, gemini in vote_pairs)
    matrix = [[counts[a, b] for b in LABELS] for a in LABELS]
    total = sum(counts.values())
    valid = sum(map(sum, matrix))
    matches = sum(matrix[i][i] for i in range(len(LABELS)))
    expected = sum(sum(matrix[i]) * sum(row[i] for row in matrix) for i in range(len(LABELS)))
    # Use the integer numerator for the degenerate expected agreement of 1.
    kappa = (matches * valid - expected) / (valid ** 2 - expected) if valid ** 2 > expected else None
    return {
        "rows": "DeepSeek", "columns": "Gemini",
        "labels": list(LABELS), "confusion_3x3": matrix,
        "labels_with_no_majority": list(MAJORITIES),
        "confusion_with_no_majority": [[counts[a, b] for b in MAJORITIES] for a in MAJORITIES],
        "n_total": total, "n_both_valid_majority": valid, "n_either_no_majority": total - valid,
        "n_agree": matches, "agreement_valid_pct": 100 * matches / valid if valid else None,
        "agreement_all_pct": 100 * matches / total if total else None, "cohen_kappa": kappa,
    }


def comparison(dataset, ce, set_):
    ce_stats, set_stats = quality(dataset, ce), quality(dataset, set_)
    result = {"CE": ce_stats, "SET": set_stats, "delta_acc_pp": None, "delta_sign": None, "mcnemar": None}
    if ce:
        difference = set_stats["counts"]["accurate"] - ce_stats["counts"]["accurate"]
        majority = LIHUA_MAJORITY if dataset.key == LIHUA_KEY else HISTORICAL.majority
        result.update(
            delta_acc_pp=100 * difference / ce_stats["denominator"],
            delta_sign=(difference > 0) - (difference < 0),
            mcnemar=MHR_BINARY.binary_pair(
                [majority(row) == "accurate" for row in ce],
                [majority(row) == "accurate" for row in set_],
            ),
        )
    return result


def analyze_dataset(dataset, out, sources):
    answers, _ = unique_answers(dataset)
    path = journal_path(dataset, out)
    rows = read_journal(path, sources)
    new = verdict_index(dataset, answers, path, rows)
    result = {
        "dataset": dataset.name, "original_judge_model": dataset.original_model,
        "swapped_judge_model": MODELS[dataset.judge],
        "journal": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "journaled_calls": len(new), "planned_calls": len(answers) * len(PASSES),
        "seeds": {},
    }
    pooled_pairs, pooled_swapped = [], []
    for seed, configs in dataset.answers.items():
        old_votes, new_votes, missing = {}, {}, {}
        for config in CONFIGS:
            rows_by_id = configs[config]
            old_votes[config] = dict(zip(rows_by_id, dataset.original_votes[config][seed]))
            new_votes[config] = {}
            missing[config] = 0
            for qid, answer in rows_by_id.items():
                passes = [new.get((*answer.key, p)) for p in PASSES]
                missing[config] += sum(row is None for row in passes)
                if all(row is not None for row in passes):
                    new_votes[config][qid] = tuple(row["verdict"] for row in passes)
        # Any complete answer contributes to agreement. CE-versus-SET metrics
        # require both configs to have three passes on the same question.
        original_swapped = [(old_votes[config][qid], votes)
                            for config in CONFIGS for qid, votes in new_votes[config].items()]
        pairs = [(b, a) if dataset.judge == "deepseek" else (a, b) for a, b in original_swapped]
        swapped = [b for _, b in original_swapped]
        pooled_pairs.extend(pairs)
        pooled_swapped.extend(swapped)
        paired_ids = sorted(new_votes["CE"].keys() & new_votes["SET"].keys())
        original_judge = "gemini" if dataset.judge == "deepseek" else "deepseek"
        scores = {
            judge: comparison(dataset, *([vote_map[config][qid] for qid in paired_ids] for config in CONFIGS))
            for judge, vote_map in ((original_judge, old_votes), (dataset.judge, new_votes))
        }
        result["seeds"][seed] = {
            "complete": not any(missing.values()),
            "missing_verdict_instances_by_config": missing,
            "n_paired_questions": len(paired_ids),
            "n_questions_per_config": len(configs["CE"]),
            "scores": scores,
            "delta_sign_agrees": (scores["deepseek"]["delta_sign"] == scores["gemini"]["delta_sign"]
                                  if paired_ids else None),
            "inter_judge": interjudge(pairs), "swapped_self_agreement": agreement(swapped),
        }
    result["pooled"] = {"inter_judge": interjudge(pooled_pairs), "swapped_self_agreement": agreement(pooled_swapped)}
    return result


def render(datasets):
    def fmt(value, digits=2, signed=False):
        if value is None:
            return "--"
        return (f"{value:+.{digits}f}".replace(".", ",") if signed else number(value, digits))

    lines = [
        r"% ACC and agreement in percent. Deltas in percentage points.",
        r"% n is the CE/SET matched question count. Agreement uses all complete answer instances.",
        r"% Valid requires a three-class majority from both judges. Cohen's kappa uses the same rows.",
        r"\begingroup\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrrcrrr}\toprule",
        r" & & \multicolumn{3}{c}{DeepSeek ACC} & \multicolumn{3}{c}{Gemini ACC}"
        r" & & \multicolumn{3}{c}{Inter-judge}\\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){10-12}",
        r"Run & $n$ & CE & SET & $\Delta$ & CE & SET & $\Delta$ & Same sign & Valid & All & $\kappa$\\",
    ]
    for dataset in datasets:
        lines.append(rf"\midrule\multicolumn{{12}}{{l}}{{\textbf{{{dataset['dataset']}}}}}\\")
        for seed, result in dataset["seeds"].items():
            cells = [seed, str(result["n_paired_questions"])]
            for judge in ("deepseek", "gemini"):
                score = result["scores"][judge]
                cells.extend([fmt(score[config]["acc_pct"]) for config in CONFIGS])
                cells.append(fmt(score["delta_acc_pp"], signed=True))
            sign = result["delta_sign_agrees"]
            cells.append("--" if sign is None else ("Yes" if sign else "No"))
            inter = result["inter_judge"]
            cells.extend((fmt(inter["agreement_valid_pct"]), fmt(inter["agreement_all_pct"]), fmt(inter["cohen_kappa"], 3)))
            lines.append(" & ".join(cells) + r"\\")
        pooled = dataset["pooled"]["inter_judge"]
        lines.append(" & ".join([r"\multicolumn{9}{l}{Pooled agreement}", fmt(pooled["agreement_valid_pct"]),
                                 fmt(pooled["agreement_all_pct"]), fmt(pooled["cohen_kappa"], 3)]) + r"\\")
    lines.extend([r"\bottomrule\end{tabular}", r"\endgroup"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="Allow seeds with missing swapped verdicts")
    args = parser.parse_args(argv)
    try:
        sources = {}
        datasets = sorted(load_datasets(sources).values(), key=lambda d: d.key)
        sanity = original_sanity(datasets, sources)
        results = [analyze_dataset(dataset, OUT, sources) for dataset in datasets]
        incomplete = [f"{dataset['dataset']} {seed} ({row['n_paired_questions']}/{row['n_questions_per_config']} "
                      "complete pairs)" for dataset in results for seed, row in dataset["seeds"].items()
                      if not row["complete"]]
        if incomplete and not args.partial:
            raise ValueError("Missing verdicts for " + ", ".join(incomplete) + ". Use --partial for incomplete seeds.")
        report = {
            "status": "partial" if incomplete else "complete",
            "passes": list(PASSES),
            "agreement_unit": "One row per question, config and generation run with three recorded swapped verdicts. "
                              "Shared judgments count in every config and run that uses them.",
            "agreement_denominators": "Valid requires a majority in accurate/error/neither from each judge. All keeps "
                                      "every compared row and counts only valid matching majorities as agreements.",
            "self_agreement": "The three passes must match. Answers with any unparsed pass are excluded from "
                              "self-agreement and Fleiss kappa, as in make_judge_agreement.py.",
            "partial_denominator": "Within each run, CE/SET quality and McNemar use the same questions with all "
                                   "swapped passes recorded in both configs. Agreement uses each complete answer.",
            "mcnemar_definition": "Exact two-sided test for accurate versus all other outcomes. LiHua-World "
                                  "uses majority with pass 1 fallback. MultiHop-RAG uses strict majority.",
            "sanity": {"matches_tab_both_datasets": True, "datasets": sanity},
            "datasets": {d["dataset"]: d for d in results},
            "source_sha256": dict(sorted(sources.items())),
        }
        write(OUT / "RESULTS.json", report)
        OUTPUT.mkdir(exist_ok=True)
        (OUTPUT / "cross_judge.tex").write_text(render(results), encoding="utf-8")
    except (AssertionError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cross-judge analysis: {exc}\n")
    print("Original-judge ACC/ERR means and SDs match tab:both-datasets for all five configurations.")
    for dataset in results:
        print(f"{dataset['dataset']}: {dataset['journaled_calls']:,}/{dataset['planned_calls']:,} swapped verdicts")


if __name__ == "__main__":
    main()
