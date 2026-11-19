# CODEX report: MultiHop-RAG row 5 (+set)

Completed the requested code and offline steps under CODEX_TASK.md and PROTOCOL.md. No existing files were edited, moved, or deleted. All new files are inside reproduce/mhr_set/ or review/mhr_set_20261117/; filesystem writers use exclusive creation (x). Offline audit hooks blocked network connections, credential-file access, overwrites, deletion/rename, and writes outside those two directories. No .env was read, copied, or printed. No endpoint, Gemini, Modal, or other network API was called.

## Files created

Source files under reproduce/mhr_set/:

- `offline.py`
- `select.py`
- `test_select.py`
- `study.py`
- `run.py`
- `deferred.py`
- `judge_entry.py`
- `online.py`
- `reporting.py`
- `verify_offline.py`

Result artifacts under review/mhr_set_20261117/:

- `dev_offline/offline_eval.json` and `dev_offline/selections.jsonl`
- `set_params.json`
- `test/SET_s101/contexts.jsonl` and `context_summary.json`
- `test/SET_s202/contexts.jsonl` and `context_summary.json`
- `test/SET_s303/contexts.jsonl` and `context_summary.json`
- `validation.json` and this `CODEX_REPORT.md`

The pre-existing CODEX_TASK.md, __init__.py, PROTOCOL.md, and codex.log were left untouched. The +CE and +F modules are reused through imports; no historical output root was redirected on disk.

## Commands run

All Python commands ran from `E:/Project/MiniRAG research/MiniRAG-Improvement`, with `PYTHONUTF8=1`, `PYTHONDONTWRITEBYTECODE=1`, and the required interpreter:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
& "E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe" -B -m reproduce.mhr_set.test_select
& "E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe" -B -m reproduce.mhr_set.study
& "E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe" -B -m reproduce.mhr_set.run freeze
& "E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe" -B -m reproduce.mhr_set.run contexts --seed 101
& "E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe" -B -m reproduce.mhr_set.run contexts --seed 202
& "E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe" -B -m reproduce.mhr_set.run contexts --seed 303
& "E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe" -B -m reproduce.mhr_set.verify_offline
```

Every command above exited 0. Additional commands only inspected task/protocol/source files, searched for repository instructions, checked git status/diff, or exclusively created the listed new files. No tracked diff was present.

## Step A: dev400

Pool: native@all; 400 questions, 353 labelled questions, 955 gold facts. All 400 CE+F ordered selections exactly reproduced the saved +CE reference. Every scored row passed probability-range, fold identity and cross-fit exclusion checks.

The following metrics use the 353 labelled questions, matching the +CE stats structure. Tokens are the sum of frozen tiktoken chunk-content counts.

| Arm | Full evidence | Facts hit / 955 | Micro recall | Mean tokens | Median tokens | Mean chunks | Median chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| CE+F | 48 / 353 | 315 | 0.3298 | 3449.13 | 3600 | 3.1303 | 3 |
| SET | 48 / 353 | 314 | 0.3288 | 3354.92 | 3600 | 3.9008 | 4 |
| SET_beta0 | 49 / 353 | 311 | 0.3257 | 3167.45 | 3600 | 3.6629 | 4 |
| SET_tau0 | 51 / 353 | 327 | 0.3424 | 3966.59 | 3972 | 6.8612 | 7 |
| ORACLE | 155 / 353 | 617 | 0.6461 | 3456.91 | 3600 | 3.2663 | 3 |

Token and chunk summaries over all 400 questions are also saved separately:

| Arm | Mean tokens | Median tokens | Mean chunks | Median chunks |
|---|---:|---:|---:|---:|
| CE+F | 3451.95 | 3600.0 | 3.1400 | 3.0 |
| SET | 3203.07 | 3600.0 | 3.7525 | 4.0 |
| SET_beta0 | 3021.03 | 3578.0 | 3.5200 | 4.0 |
| SET_tau0 | 3966.50 | 3972.0 | 6.9725 | 7.0 |
| ORACLE | 3458.30 | 3600.0 | 3.2575 | 3.0 |

Nested choices, using labelled questions in the other four folds:

| Outer fold | SET (beta, tau) | SET_beta0 (beta, tau) | SET_tau0 (beta, tau) |
|---|---|---|---|
| 0 | (1.0, 0.1) | (0.0, 0.05) | (1.0, 0.0) |
| 1 | (0.0, 0.05) | (0.0, 0.05) | (0.0, 0.0) |
| 2 | (0.0, 0.05) | (0.0, 0.05) | (0.0, 0.0) |
| 3 | (0.0, 0.05) | (0.0, 0.05) | (0.0, 0.0) |
| 4 | (0.0, 0.05) | (0.0, 0.05) | (0.0, 0.0) |

SET_tau0 follows evidence_set.evaluate SET_FILL exactly: the nested SET beta with tau forced to zero. SET_beta0 tunes tau independently with beta fixed to zero.

SET vs CE+F full evidence: +5 / -5, exact sign/McNemar p = 1.0. Facts per-question sign test: +24 / -26, p = 0.887724827340783.

Per-type full evidence, micro/macro recall, token/chunk summaries and the full 30-setting grid are in dev_offline/offline_eval.json. Step A is descriptive and did not gate the test context builds.

## Frozen parameters

beta = 0.0; tau = 0.05. Frozen at 2026-11-17T12:24:50.980865+00:00, before any SET test context was built. The all-labelled-dev grid was recomputed and matched Step A exactly. The selected setting has full evidence 49; grid best = 52, with ceil(1% of 353) = 4 tolerance. Choice then minimizes mean tokens, followed by -full count and (beta, tau).

set_params.json records the grid and SHA-256 of scores_CEF.jsonl, model_CE.json, dev pools, and the SET offline_eval.json:

- `scores_CEF_sha256`: `904f30256e16375a5161f48bb03dfdc94c1fe9047d1db01c16fb7ad70687bad6`
- `model_CE_sha256`: `b08f13bd8598ee6c8506d46551a144b036f52cb2161f57d42912b738d2e8e642`
- `pools_sha256`: `cb06faf6dd5b47bf7f2699076285a945b648a23b99336a4e9dc996cd5a4adc6b`
- `offline_eval_sha256`: `c300072ba49b46bacd6332b82eaf38bf2023650d22d035acac736b05916d9c9f`

## Test contexts

| Seed | Questions / ok | Same as CE | Mean tokens | Median tokens | Min / max tokens | Mean chunks |
|---|---:|---:|---:|---:|---:|---:|
| 101 | 2156 / 2156 | 281 | 3219.53 | 3599.0 | 0 / 3999 | 3.8669 |
| 202 | 2156 / 2156 | 274 | 3215.29 | 3599.0 | 0 / 4000 | 3.8502 |
| 303 | 2156 / 2156 | 281 | 3210.40 | 3599.0 | 0 / 4000 | 3.8469 |

All 2,156 questions per seed passed A1(CE probability order) == stored CE chunk_ids and the CE context UTF-8 byte-for-byte rebuild. All SET selections passed budget, uniqueness and pool-membership checks. No context_check_failed.json was created. Each context_summary.json contains input/output SHA-256 values, complete token/chunk summaries, and zero LLM calls.

## Additional verification

Three unit tests passed, including 720 synthetic reference comparisons across the full beta/tau grid, grouped/ungrouped documents, probability clipping, duplicate/missing candidates, exact tie order, zero gain, facets, and oversized-chunk skipping. AST comparison confirms that both the selector port and inline reference have exactly the original select() body after removing its corpus() lookup. All 11 Python files (including the pre-existing empty __init__.py) compile.

verify_offline independently parsed and checked all 6,468 Sources CSV blocks, their row numbering/content/selection order, byte rebuilds, same_as_ce flags, token budgets, duplicates, pool membership and output checksums. The complete frozen index manifest was verified unchanged. Evidence: validation.json.

## Deferred implementation and limits

Implemented answers, execute, finish and the five-arm report, plus the MINIRAG_DOTENV operator launcher. Answers reuse identical CE contexts; judging composes C0/ST/F/CE/SET verdicts and judges only SET answer deltas. The report implements registered SET-vs-CE, secondary SET-vs-F, SET-vs-C0, per-type results, ERR/Neither, evidence recall, full evidence and mean Sources tokens. The registered entity-answer substring checks, Fleiss kappa and deterministic 200-answer CE drift rejudging are included for the later operator run.

To satisfy the hard no-overwrite rule, online retries create immutable attempt directories and resume from successful prior records. Legacy initial w/a writer opens are adapted in memory to x. Progress/failures use immutable status event files; status.json/status_judge.json are created on successful completion. The first serving check uses serving_checks.jsonl, with subsequent immutable checks in serving_checks_parts/.

As required, answers, execute, finish, report and online.py were NOT executed. Their code was syntax-checked only; no online behavior, accuracy result, judge drift result or final RESULTS.json/BANG_SO_SANH.txt is claimed. No requested offline step was blocked or left unfinished.
