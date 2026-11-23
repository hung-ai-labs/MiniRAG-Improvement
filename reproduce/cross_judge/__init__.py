"""Shared inputs and frozen judging protocol for the CE/SET cross-judge check."""

import ast
import collections
import csv
import json
import math
import sys
from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace

from reproduce.mhr_st.common import JUDGE as GEMINI_MODEL, OUT as ST_OUT, ROOT, SEEDS

# The existing report script imports its sibling by script name.
sys.path.insert(0, str(ROOT / "report/scripts"))
try:
    from make_judge_agreement import (
        ARMS, LABELS, LIHUA, MHR_TEST, MULTIHOP, OUTPUT, PASSES, agreement,
        lihua_votes, multihop_votes, number, read_csv, read_json, read_jsonl,
        source_hash, unique,
    )
finally:
    sys.path.pop(0)

OUT = ROOT / "review/cross_judge_20261124"
CONFIGS = ("CE", "SET")
LIHUA_KEY = "lihua_world"
MHR_KEY = "multihop_rag"
LIHUA_PROMPT = ROOT / "reproduce/Step_2_evaluate.py"
MHR_PROMPT = ST_OUT / "source/reproduce/Step_2_evaluate.py"
MHR_RUNNER = ST_OUT / "source/reproduce/research/judge_run.py"
GEMINI_CLIENT = ST_OUT / "source/minirag/llm/gemini.py"
GEMINI_UTILS = ST_OUT / "source/minirag/utils.py"
DEEPSEEK_RUNNER = ROOT / "reproduce/evidence_set/deepseek_judge.py"
LIHUA_TEMPLATE_LOADER = ROOT / "reproduce/graph_ablation/api.py"
LIHUA_PARSER = ROOT / "reproduce/oracle_headroom/run.py"
LIHUA_PIPELINE = ROOT / "reproduce/evidence_set/pipeline.py"
MHR_REPORTING = ROOT / "reproduce/mhr_set/reporting.py"
MHR_EXACT = ROOT / "reproduce/mhr_f/study.py"
MHR_REPORT = MULTIHOP / "RESULTS.json"
MAIN_TABLE = OUTPUT / "selection_results.tex"
PASS_LABELS = (*LABELS, "unparsed")


def source_symbols(path, *names, **dependencies):
    """Load small protocol definitions without importing their experiment runtimes.

    Those runtimes import model libraries, load dotenv, or change process state
    at import time. Only the named functions and literal assignments are loaded.
    """
    selected = {}
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected[node.name] = node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    dependencies[target.id] = ast.literal_eval(node.value)
    for node in selected.values():
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), dependencies)
    return SimpleNamespace(**{name: dependencies[name] for name in names})


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Answer:
    id: object
    prediction_sha256: str
    question: str
    gold: str
    prediction: str

    @property
    def key(self):
        return self.id, self.prediction_sha256


@dataclass
class Dataset:
    key: str
    name: str
    original_model: str
    judge: str
    template: str
    parser: object
    answers: dict
    original_votes: dict


def load_lihua(sources):
    model, votes = lihua_votes(sources)
    if model != DEEPSEEK.MODEL:
        raise ValueError("Unexpected LiHua-World original judge")
    # judge_template() extracts this literal. Decode explicitly because its
    # unqualified read_text() uses a legacy code page on some Windows hosts.
    template = source_symbols(LIHUA_PROMPT, "JUDGE_PROMPT").JUDGE_PROMPT
    parser = source_symbols(LIHUA_PARSER, "parse", "LABELS").parse
    path = LIHUA.with_name("answers.jsonl")
    rows = unique(read_jsonl(path), lambda r: (str(r["round"]), r["arm"], r["id"]), path)
    ids = sorted({row["id"] for row in read_jsonl(LIHUA.with_name("contexts.jsonl"))})
    seeds = tuple(map(str, read_json(LIHUA)["rounds"]))
    if len(ids) != 435 or seeds != ("1", "2", "3"):
        raise ValueError("Expected LiHua-World non-dev435 and rounds 1, 2, 3")
    answers = {}
    for seed in seeds:
        answers[seed] = {}
        for config, alias in ARMS:
            if config not in CONFIGS:
                continue
            answers[seed][config] = {}
            for qid in ids:
                row = rows[seed, alias, qid]
                answer = Answer(qid, row["prediction_sha256"], row["question"], row["gold"], row["prediction"])
                if digest(answer.prediction) != answer.prediction_sha256:
                    raise ValueError(f"LiHua-World answer hash differs for {config}/{seed}/{qid}")
                answers[seed][config][qid] = answer
    return Dataset(LIHUA_KEY, "LiHua-World", model, "gemini", template, parser, answers, votes)


def load_multihop(sources):
    model, votes = multihop_votes(sources)
    template = source_symbols(MHR_PROMPT, "JUDGE_PROMPT").JUDGE_PROMPT
    if model != GEMINI_MODEL or read_json(MHR_REPORT)["n"] != 2156 or set(votes["CE"]) != set(map(str, SEEDS)):
        raise ValueError("Unexpected MultiHop-RAG judge, test size or seeds")
    answers = {}
    for seed in map(str, SEEDS):
        answers[seed] = {}
        for config, study in (("CE", "mhr_ce_20261116"), ("SET", "mhr_set_20261117")):
            folder = ROOT / "review" / study / f"test/{config}_s{seed}"
            answers_path = folder / "answers.csv"
            provenance_path = folder / "judge_gemini35/provenance.json"
            provenance = read_json(provenance_path)
            source_hash(provenance_path, sources)
            if (source_hash(answers_path, sources) != provenance[f"{config.lower()}_answers_sha256"]
                    or provenance["judge_model"] != model):
                raise ValueError(f"MultiHop-RAG answer provenance differs for {config}/{seed}")
            # CE's pinned manifest records the template hash. SET uses the same
            # frozen launcher via mhr_set/judge_entry.py and mhr_ce/judge_entry.py.
            if config == "CE":
                manifest_path = folder / "judge_delta/run/manifest.json"
                manifest = read_json(manifest_path)
                source_hash(manifest_path, sources)
                if manifest["prompt_sha256"] != digest(template) or manifest["judge_model"] != model:
                    raise ValueError(f"MultiHop-RAG frozen prompt or model differs for CE/{seed}")
            rows = {row["query_id"]: row for row in HISTORICAL.load_rows(MHR_TEST, answers_path, "minirag")}
            # multihop_votes returns CSV row order, which need not be question-file order.
            judged = read_csv(folder / "judge_gemini35/per_question.csv")
            answers[seed][config] = {}
            for row in judged:
                qid = row["query_id"]
                passes = tuple(row[f"pass{p}"] for p in PASSES)
                if any(label not in PASS_LABELS for label in passes) or HISTORICAL.majority(passes) != row["majority"]:
                    raise ValueError(f"MultiHop-RAG original majority differs for {config}/{seed}/{qid}")
                loaded = rows[qid]
                pred = loaded["pred"]
                answers[seed][config][qid] = Answer(qid, digest(pred), loaded["question"], loaded["gold"], pred)
    return Dataset(MHR_KEY, "MultiHop-RAG", model, "deepseek", template, HISTORICAL.parse_label, answers, votes)


def load_datasets(sources):
    for path in (LIHUA_PROMPT, MHR_PROMPT, MHR_RUNNER, GEMINI_CLIENT, GEMINI_UTILS,
                 DEEPSEEK_RUNNER, LIHUA_TEMPLATE_LOADER, LIHUA_PARSER, LIHUA_PIPELINE,
                 MHR_REPORTING, MHR_EXACT, ROOT / "report/scripts/make_judge_agreement.py",
                 ROOT / "report/scripts/make_selection_tables.py", ROOT / "reproduce/mhr_st/judge.py",
                 ROOT / "reproduce/mhr_st/common.py", ROOT / "reproduce/mhr_ce/judge_entry.py",
                 ROOT / "reproduce/mhr_set/judge_entry.py", ROOT / "reproduce/mhr_set/deferred.py"):
        source_hash(path, sources)
    return {dataset.judge: dataset for dataset in (load_multihop(sources), load_lihua(sources))}


def unique_answers(dataset):
    """Union by exact (question id, prediction hash), retaining all uses."""
    answers, uses = {}, collections.defaultdict(list)
    questions = {}
    for seed, configs in dataset.answers.items():
        for config, rows in configs.items():
            for answer in rows.values():
                reference = questions.setdefault(answer.id, (answer.question, answer.gold))
                if reference != (answer.question, answer.gold) or answers.setdefault(answer.key, answer) != answer:
                    raise ValueError(f"Inconsistent frozen answer inputs for {dataset.key}/{answer.id}")
                uses[answer.key].append((seed, config))
    return answers, uses


def journal_path(dataset, out=OUT):
    return out / f"{dataset.key}_{dataset.judge}.jsonl"


def call_key(row):
    return row["id"], row["prediction_sha256"], row["judge_pass"]


def read_journal(path, sources=None):
    if not path.exists():
        return []
    contents = path.read_bytes()
    if contents and not contents.endswith(b"\n"):
        raise ValueError(f"Incomplete final line in {path.name}; refusing to read a damaged journal")
    rows = []
    for number, line in enumerate(contents.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f"Corrupt JSON in {path.name} at line {number}") from None
    if sources is not None:
        name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        sources[name] = sha256(contents).hexdigest()
    return rows


def verdict_index(dataset, answers, path, rows=None):
    indexed = unique(read_journal(path) if rows is None else rows, call_key, path)
    for (qid, digest_value, judge_pass), row in indexed.items():
        if ((qid, digest_value) not in answers or judge_pass not in PASSES
                or row["dataset"] != dataset.key or row["judge_model"] != MODELS[dataset.judge]
                or row["verdict"] not in PASS_LABELS):
            raise ValueError(f"Unexpected verdict identity in {path.name}")
    return indexed


DEEPSEEK = source_symbols(DEEPSEEK_RUNNER, "MODEL", "URL", "PRICE", "cost")
MODELS = {"deepseek": DEEPSEEK.MODEL, "gemini": GEMINI_MODEL}
HISTORICAL = source_symbols(MHR_RUNNER, "LABELS", "STRIP", "parse_label", "majority", "load_rows",
                            collections=collections, csv=csv)
LIHUA_MAJORITY = source_symbols(LIHUA_PIPELINE, "majority", "LABELS").majority
MHR_BINARY = source_symbols(MHR_REPORTING, "binary_pair", base=source_symbols(MHR_EXACT, "exact_p", math=math))
