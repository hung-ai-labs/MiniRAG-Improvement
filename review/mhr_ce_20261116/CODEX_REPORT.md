# CODEX report: MultiHop-RAG +CE, 2026-11-16

Completed the authorized code and offline steps in `reproduce/mhr_ce/CODEX_TASK.md` under the registered `PROTOCOL.md`.

## Hard-rule compliance

- Created files only in `reproduce/mhr_ce/` and `review/mhr_ce_20261116/`. No pre-existing file was edited, moved, or deleted. Existing +F/+ST code was imported and reused in memory.
- No `.env` was read, copied, or printed. The historical dotenv loader was disabled in memory before package use. Offline utility functions were compiled directly from the frozen source AST to avoid eager ML imports and their filesystem side effects.
- No network API, SLM endpoint, Gemini, or Modal call was made by this task. Offline work ran with a network/credential-file/add-only audit guard. The externally running CE scoring process was neither started nor modified by this task.
- `answers`, `execute`, `finish`, and `report` were not executed. No CE answers, judge journals, serving logs, status journals, RESULTS.json, or BANG_SO_SANH.txt were generated.
- Used the required Python executable from the repository root, with `PYTHONUTF8=1`; also used `PYTHONDONTWRITEBYTECODE=1` and `-B` to avoid modifying bytecode caches.
- The first context command waited for `ce_logits_test_summary.json`, polling every 120 seconds with a 90-minute maximum. All test context builds used the completed summary and checked its score/pool checksums.

## Files created

- `reproduce/mhr_ce/study.py`
- `reproduce/mhr_ce/run.py`
- `reproduce/mhr_ce/offline.py`
- `reproduce/mhr_ce/local_runtime.py`
- `reproduce/mhr_ce/judge_entry.py`
- `reproduce/mhr_ce/verify_offline.py`
- `review/mhr_ce_20261116/dev_offline/offline_eval.json`
- `review/mhr_ce_20261116/dev_offline/scores_CEF.jsonl`
- `review/mhr_ce_20261116/dev_offline/selections.jsonl`
- `review/mhr_ce_20261116/model_CE.json`
- `review/mhr_ce_20261116/offline_validation.json`
- `review/mhr_ce_20261116/test/CE_s101/{contexts.jsonl,context_summary.json}`
- `review/mhr_ce_20261116/test/CE_s202/{contexts.jsonl,context_summary.json}`
- `review/mhr_ce_20261116/test/CE_s303/{contexts.jsonl,context_summary.json}`
- `review/mhr_ce_20261116/CODEX_REPORT.md` (this report)

The pre-existing logits and their summaries, chunk-text snapshot, protocol, task, scoring scripts, and external process logs are inputs, not files created by this task.

## Commands run

Working directory: `E:/Project/MiniRAG research/MiniRAG-Improvement`.

Successful workflow and validation commands (PowerShell):

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe' -B -m reproduce.mhr_ce.study
& 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe' -B -m reproduce.mhr_ce.run --help
& 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe' -B -m reproduce.mhr_ce.run freeze
& 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe' -B -u -m reproduce.mhr_ce.run contexts --seed 101
& 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe' -B -m reproduce.mhr_ce.verify_offline
& 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe' -B -u -m reproduce.mhr_ce.run contexts --seed 202
& 'E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe' -B -u -m reproduce.mhr_ce.run contexts --seed 303
```

Also ran read-only source/protocol/data inspections, `git status`/`git diff --stat`, and inline Python final integrity checks. Git showed no tracked-file diff. The final checks verified all output hashes, unique query counts, same-as-F counts, absence of failure/online artifacts, model-before-context file ordering, and all 10 index-manifest checksums.

Development attempts: the first study invocation was stopped by the audit guard when NumPy queried Windows through the virtual NUL device. A second invocation with the local platform query cached was stopped before filelock could create a temporary probe file outside the allowed directories. Both failed before creating study outputs. The final local-runtime loader avoids those imports; the normal study command then succeeded. An inline, read-only context-format inspection also had a PowerShell quoting SyntaxError and was replaced by a PowerShell JSON inspection. No failed attempt read `.env`, reached the network, or changed a pre-existing file.

## Step A: dev400

Pool: `native@all`; same 5 grouped folds, features, labels, fit(), tie handling, and A1@4000 as +F. There are 353 evidence-labelled questions and 955 gold facts; the other 47 questions are excluded from evidence statistics as in the original study.

| Arm | Full evidence (count / 353) | Full evidence rate | Facts hit / 955 | Micro recall | Macro recall |
|---|---:|---:|---:|---:|---:|
| C0 | 10 / 353 | 2.83% | 157 / 955 | 0.1644 | 0.1830 |
| ST | 13 / 353 | 3.68% | 187 / 955 | 0.1958 | 0.2188 |
| F | 38 / 353 | 10.76% | 301 / 955 | 0.3152 | 0.3404 |
| CE+F | 48 / 353 | 13.60% | 315 / 955 | 0.3298 | 0.3626 |
| CE | 30 / 353 | 8.50% | 262 / 955 | 0.2743 | 0.3052 |
| ORACLE | 155 / 353 | 43.91% | 617 / 955 | 0.6461 | 0.6851 |

Exact paired comparisons:

| Comparison | Full up | Full down | Full exact p | Fact-count up questions | Fact-count down questions | Fact-count sign exact p |
|---|---:|---:|---:|---:|---:|---:|
| CE+F_vs_F | 16 | 6 | 0.0524787902832 | 40 | 24 | 0.059941189567 |
| F_vs_ST | 27 | 2 | 1.62422657013e-06 | 115 | 25 | 5.40817682507e-15 |
| CE_vs_F | 13 | 21 | 0.229481013026 | 61 | 94 | 0.00993266838337 |

CE+F vs F increased full evidence by 10 questions and facts hit by 14. Neither dev p-value is below 0.05. Step A is descriptive; the registered protocol requires the test row regardless of these values unless a code check fails.

| Fold | F AUC | CE+F AUC |
|---|---:|---:|
| 0 | 0.9306 | 0.9505 |
| 1 | 0.8969 | 0.9302 |
| 2 | 0.9571 | 0.9675 |
| 3 | 0.9631 | 0.9691 |
| 4 | 0.9514 | 0.9631 |

F selections matched the saved +F selections for every one of the 400 questions. The validation additionally matched all per-type/all statistics for C0, ST, F, and ORACLE, the F AUCs, fold sizes, and F-vs-ST comparison exactly to the historical results.

`scores_CEF.jsonl` contains 83,845 unique (query, chunk) records. Each record's held-out fold and four training folds were checked. Duplicate resumed logits are merged by query and chunk; latest values win for repeated chunks, and previous non-repeated chunks are retained.

## Freeze

- Features: `ce_logit` followed by the original 18 features, in order; 83,845 training rows and 613 positives.
- Closed-form maximum absolute probability gap against sklearn: `0.0` (required `< 1e-9`).
- Frozen at `2026-11-16T13:07:30.570973+00:00`, before all CE test context outputs. Freeze refuses an existing model file.
- Model SHA-256: `b08f13bd8598ee6c8506d46551a144b036f52cb2161f57d42912b738d2e8e642`.
- Dev pools SHA-256: `cb06faf6dd5b47bf7f2699076285a945b648a23b99336a4e9dc996cd5a4adc6b`.
- Dev logits SHA-256: `803d90f97b44972b365b6bb465d4eb8f1682ef10bf33c1ec349d664169e42907`.
- Offline evaluation SHA-256: `38eae6c2b14dd92382298184020f5cd58de467cc2a2742142e05a98669f0d4a1`.

## Test context summaries

| Seed | Questions | Status ok | Identical to F | Changed from F | F selection mismatches | F byte-rebuild mismatches |
|---|---:|---:|---:|---:|---:|---:|
| 101 | 2156 | 2156 | 570 | 1586 | 0 | 0 |
| 202 | 2156 | 2156 | 590 | 1566 | 0 | 0 |
| 303 | 2156 | 2156 | 559 | 1597 | 0 | 0 |

All 6,468 questions passed both required checks: A1(recomputed F order) equals the saved F chunk IDs, and rebuilding Sources with those IDs reproduces the original F context bytes exactly. CE contexts change only the Sources CSV and preserve the surrounding F context. Each context summary records context/pool/model/logit/chunk/index hashes. All F statuses were ok; the code also preserves non-ok F status/context with empty chunk IDs.

Completed test logits: 2,156 queries and 536,190 unique query/chunk pairs; SHA-256 `3a777cfe43e489c0365195d553aff53ba529465d828470d3ecf294a3f163a213`.

| Seed | CE contexts SHA-256 |
|---|---|
| 101 | `0ed0f019a763edf7df1d982292ddb52d490cf9311b134c5647a7a4dba5758719` |
| 202 | `8cb5492fa3bfce1e2bb09b4866b8294fca8243e5ed9b21277b9087f44155f62e` |
| 303 | `ebb65003a0eb7b0fd9e6d8db7d963252fba42566f5231567c9c3e315cb4ae6ee` |

## Deferred implementation and limits

`run.py` supplies all six requested subcommands. Answers reuse the F journal with `frozen_F_identical_context`, and serving checks are directed to this CE run. Execute skips verified completed context summaries, launches answer subprocesses, retries failures, and maintains status. Finish selects only CE predictions different from C0/ST/F, composes the four journals with the requested origin labels, and maintains judge status. The no-dotenv judge entry point keeps the same pinned judge, three passes, and retry policy. Report provides four rows, primary CE-vs-F and CE-vs-C0 paired tests, exploratory by-type CE-vs-F, four-arm evidence recall, ERR/Neither, and the registered all-three-seeds decision.

The offline validator checked source compilation, baseline reproduction, cross-fit metadata, resumed-logit merging, Unicode/multiline/quoted CSV reconstruction, native-index ties, missing-logit failure, and frozen-model provenance. The network-dependent paths were written and statically checked, but deliberately not exercised. Credentials for a later operator run must already be in that process environment; this implementation does not load credential files.

No authorized offline work remains incomplete. Test answer accuracy, ERR/Neither, verdicts, paired test conclusions, RESULTS.json, and BANG_SO_SANH.txt remain unavailable because their commands were expressly prohibited in this task.
