"""Prepare and score one Qwen MultiHop-RAG evaluation over all 2,556 questions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path

from reproduce.mhr_st.common import DATA, JUDGE, ROOT, OUT as ST, records, sha, write, read

OUT = ROOT / "review/mhr_full_20261119"
SEED = 101  # the generation seed shared by both frozen cohorts
ARMS = ("C0", "ST", "F", "CE", "SET")
F = ROOT / "review/mhr_f_20261114"
CE = ROOT / "review/mhr_ce_20261116"
SET = ROOT / "review/mhr_set_20261117"
DEV_C0 = ROOT.parent / "MiniRAG/runs/multihop_rag/dev400/C0_s101"


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def keyed(rows):
    result = {}
    for row in rows:
        key = row["query_id"]
        if key in result:
            raise ValueError(f"Duplicate query: {key}")
        result[key] = row
    return result


def questions():
    whole = keyed(csv_rows(DATA / "processed/questions.csv"))
    dev = keyed(csv_rows(ST / "inputs/dev.csv"))
    test = keyed(csv_rows(ST / "inputs/test.csv"))
    if (len(whole), len(dev), len(test)) != (2556, 400, 2156):
        raise ValueError("Unexpected MultiHop-RAG dataset coverage")
    if set(dev) & set(test) or whole != (dev | test):
        raise ValueError("Frozen cohorts do not reproduce the full question set")
    return whole, dev, test


def test_verdicts(arm):
    root = {"C0": ST, "ST": ST, "F": F, "CE": CE, "SET": SET}[arm]
    return root / f"test/{arm}_s{SEED}/judge_gemini35/per_question.csv"


def sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cached_selections(path, key, dev, folds, pools):
    probabilities = {qid: {} for qid in dev}
    for row in records(path):
        qid, chunk, probability = row["query_id"], row["chunk_id"], row["p"]
        expected = set(folds.values()) - {folds[qid]}
        if (row["fold"] != folds[qid] or set(row["trained_on_folds"]) != expected
                or chunk in probabilities[qid] or not 0 <= probability <= 1):
            raise ValueError(f"Invalid out-of-fold {key} score for {qid}/{chunk}")
        probabilities[qid][chunk] = probability
    for qid, pool in pools.items():
        if set(probabilities[qid]) != set(pool):
            raise ValueError(f"Missing {key} candidates for {qid}")
    return probabilities


def nested_set_params(rows, fold, features, labels, meta, pools, facet_map, chunks):
    import numpy as np
    from reproduce.mhr_f import study as base
    from reproduce.mhr_set import study as selector

    fold_ids = sorted(set(fold.values()))
    assignments = np.array([item[2] for item in meta])
    ids_per_fold = {idx: np.flatnonzero(assignments == idx) for idx in fold_ids}
    mask = {idx: assignments != idx for idx in fold_ids}
    probabilities = {idx: {} for idx in fold_ids}
    # One model per pair of excluded folds supplies both inner validations.
    for pos, first in enumerate(fold_ids):
        for second in fold_ids[pos + 1:]:
            training = mask[first] & mask[second]
            model = base.fit(features[training], labels[training])
            for outer, inner in ((first, second), (second, first)):
                idxs = ids_per_fold[inner]
                scores = model.predict_proba(features[idxs])[:, 1]
                for idx, probability in zip(idxs, scores, strict=True):
                    qid, chunk, _ = meta[idx]
                    probabilities[outer].setdefault(qid, {})[chunk] = float(probability)
    chosen = {}
    for outer in fold_ids:
        inner_rows = [row for row in rows if fold[row["query_id"]] != outer]
        labelled, grid = selector.grid_results(inner_rows, pools, probabilities[outer], facet_map, chunks)
        chosen[outer] = selector.choose(labelled, grid)
        print(f"SET outer fold {outer}: beta, tau = {chosen[outer]}", flush=True)
    return chosen


def prepare():
    whole, dev, test = questions()
    from reproduce.mhr_ce import offline
    from reproduce.mhr_ce.local_runtime import install

    install()
    from reproduce.mhr_ce.run import rebuild
    from reproduce.mhr_f import study as base
    from reproduce.mhr_set.select import facets, select
    from reproduce.mhr_set.offline import csv_write, jsonl

    raw_rows, raw_pools, pool_name, _, folds, features, labels, meta, _ = offline.dev_data()
    if keyed(raw_rows) != dev or pool_name != "native@all":
        raise ValueError("Stored development inputs changed")
    chunks, _, by_time = base.corpus()
    pools = {qid: base.variants(row)[pool_name] if row["status"] == "ok" else []
             for qid, row in raw_pools.items()}
    facet_map = {qid: facets(raw_pools[qid], pool, by_time) for qid, pool in pools.items()}
    probabilities = {
        "F": cached_selections(F / "dev_offline/scores_F.jsonl", "F", dev, folds, pools),
        "CE": cached_selections(CE / "dev_offline/scores_CEF.jsonl", "CE", dev, folds, pools),
    }
    c0_path = DEV_C0 / "contexts.jsonl"
    c0 = keyed(records(c0_path))
    if set(c0) != set(dev):
        raise ValueError("Frozen C0 contexts do not cover all 400 questions")
    # Verify the byte-exact ST template before doing any fits.
    temporal = {}
    for qid, row in c0.items():
        pool = raw_pools[qid]
        if row["status"] != pool["status"] or row["status"] != "ok":
            raise ValueError(f"No verified ST source template for {qid}")
        temporal[qid] = rebuild(row["context"], pool["kept"], chunks)
        if sha_text(temporal[qid]) != pool["context_sha256"]:
            raise ValueError(f"ST source replay differs for {qid}")
    chosen = nested_set_params(raw_rows, folds, features, labels, meta, pools, facet_map, chunks)
    contexts = []
    for qid in (qid for qid in whole if qid in dev):
        template = temporal[qid]
        context = {"C0": c0[qid]["context"], "ST": template}
        for key, prob in probabilities.items():
            ranked = sorted(pools[qid], key=prob[qid].__getitem__, reverse=True)
            context[key] = rebuild(template, base.a1(ranked), chunks)
        beta, tau = chosen[folds[qid]]
        ids, info = select(pools[qid], probabilities["CE"][qid], facet_map[qid], chunks, beta=beta, tau=tau)
        if info["tokens"] > base.BUDGET or len(ids) != len(set(ids)) or set(ids) - set(pools[qid]):
            raise ValueError(f"Invalid SET context for {qid}")
        context["SET"] = rebuild(template, ids, chunks)
        contexts.append({"query_id": qid, "fold": folds[qid], "contexts": context})
    context_path = OUT / "dev_contexts.jsonl"
    csv_write(OUT / "questions_full.csv", list(next(iter(whole.values()))), whole.values())
    jsonl(context_path, contexts)
    write(OUT / "context_manifest.json", {
        "generation_seed": SEED, "n": len(whole), "existing_test": len(test), "new_dev": len(dev),
        "dev_st_replays": len(temporal), "set_nested_params": chosen,
        "contexts_sha256": sha(context_path),
        "test_verdicts_sha256": {arm: sha(test_verdicts(arm)) for arm in ARMS},
    })
    print(f"Verified {len(whole)} questions and prepared {len(dev)} context bundles.")


def majority(labels):
    label, count = Counter(labels).most_common(1)[0]
    return label if count >= 2 else "no_majority"


def summarize():
    whole, dev, test = questions()
    manifest = read(OUT / "context_manifest.json")
    dev_path = OUT / "dev_verdicts.jsonl"
    dev_verdicts = keyed(records(dev_path)) if dev_path.exists() else None
    if dev_verdicts is not None and set(dev_verdicts) != set(dev):
        raise ValueError("Development verdicts must be complete before reporting full-set metrics")
    result = {"generation_seed": SEED, "judge": JUDGE, "n_full": len(whole),
              "n_existing": len(test), "n_pending": len(dev) if dev_verdicts is None else 0, "arms": {}}
    allowed = {"accurate", "error", "neither", "unparsed", "no_majority"}
    for arm in ARMS:
        path = test_verdicts(arm)
        if sha(path) != manifest["test_verdicts_sha256"][arm]:
            raise ValueError(f"Frozen verdicts changed for {arm}")
        old = keyed(csv_rows(path))
        if set(old) != set(test) or any(row["Type"] != test[qid]["Type"] for qid, row in old.items()):
            raise ValueError(f"Test verdict coverage differs for {arm}")
        labels = {qid: row["majority"] for qid, row in old.items()}
        if dev_verdicts is not None:
            labels.update({qid: row["verdicts"][arm] for qid, row in dev_verdicts.items()})
        counts = dict(Counter(labels.values()))
        if counts.keys() - allowed:
            raise ValueError(f"Unknown or incomplete verdicts for {arm}")
        total = len(labels)
        result["arms"][arm] = {"n": total, "counts": counts,
                               "acc": 100 * counts.get("accurate", 0) / total,
                               "err": 100 * counts.get("error", 0) / total}
    write(OUT / "results.json", result)
    print(f"Covered {result['n_full'] - result['n_pending']}/{result['n_full']} questions per configuration.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "answers", "judge", "report"))
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if args.step == "prepare":
        prepare()
    elif args.step == "report":
        summarize()
    else:
        from reproduce import mhr_full_online
        if args.step == "answers":
            mhr_full_online.answers()
        else:
            mhr_full_online.judge(args.rounds)


if __name__ == "__main__":
    main()
