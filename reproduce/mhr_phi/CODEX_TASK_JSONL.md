# Codex task: JSONL reader fix for Phi outputs + dependency manifest (offline only)

Hard rules: as in reproduce/mhr_phi/CODEX_TASK.md (create/edit only in reproduce/mhr_phi/ and review/mhr_phi_20261118/;
never edit other existing files; never read .env; no network/LLM/Modal calls; Python
"E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe", cwd repo root, PYTHONUTF8=1).

## Problem
Phi wrote the character U+0085 (NEL) inside 2 answers of review/mhr_phi_20261118/c0/archive/qa/s202/C0_s202/answers.jsonl.
The writers (phased_qa.append_jsonl, json.dumps(ensure_ascii=False) + "\n") are correct: the file has exactly 2,156 records,
one per "\n". Three frozen readers split with str.splitlines(), which also splits on \x85, \u2028, \u2029, \x0b, \x0c,
\x1c-\x1e, so they cut a record in two and raise JSONDecodeError:
- review/mhr_st_20261112/source/reproduce/research/phased_qa.py read_jsonl (line ~95)
- review/mhr_st_20261112/source/reproduce/research/judge_run.py read_records (line ~127)
- reproduce/mhr_st/common.py records (line ~35)
(also scan every other module the Phi ladder runs for the same pattern on JSONL/CSV-derived text and include them).

## Fix (in reproduce/mhr_phi only, installed by profile.apply())
Replace these readers in memory with versions that split on "\n" only (strip a trailing "\r"), otherwise byte-for-byte the
same semantics (same skipping of blank lines, same key handling, same error on a genuinely corrupt line, same "last record
wins"). For a file without those characters the result must be identical to the original reader — test that on every
existing Phi and Qwen JSONL the readers touch (C0 dev/s101/s202 answers, contexts, requests, keyword caches; Qwen
archive_c0 and judge journals) and on synthetic files with each separator. Add the case to test_profile or a new
test_jsonl.py. Record in review/mhr_phi_20261118/DEVIATIONS.md a dated entry: reader fix only, no data changed, why, which
files contain the separators (count per file), and that Qwen files are read identically.
Then verify offline that `phased_qa.read_jsonl` under the profile reads s202 answers.jsonl as 2,156 records and that c0.py's
answer step would find nothing left to generate for s202 (do not run anything that contacts the SLM endpoint; if the answer
step always runs the serving check first, just report it — the operator will finish s202 on the next workspace).

## Dependency manifest
Write review/mhr_phi_20261118/ops/QWEN_SIDE_INPUTS.md: every path OUTSIDE review/mhr_phi_20261118 and reproduce/mhr_phi
that the Phi ladder (all stages in RUNBOOK.md, under the profile) reads, found by static analysis of the code plus the
profile's redirect table: path, which stage reads it, required (fails without it) vs test-only, size. Group: code, frozen
source, datasets/indexes, Qwen result files. This decides what must be pushed to git for another machine to continue.

Append a section "JSONL reader" to review/mhr_phi_20261118/CODEX_REPORT.md.
