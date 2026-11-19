# Codex task: run the MultiHop-RAG ladder code on Phi-3.5-mini (profile + C0 stage) — code and offline checks only

Read first: `review/mhr_phi_20261118/PROTOCOL.md` (registered — follow exactly), then the Qwen ladder code
`reproduce/mhr_st/*`, `reproduce/mhr_f/*`, `reproduce/mhr_ce/*`, `reproduce/mhr_set/*`, and the frozen MiniRAG
source they run: `review/mhr_st_20261112/source/` (commit b640765; `reproduce/research/phased_qa.py`, `qa_run.py`,
`run_mhr_qa.sh`, `shard_mhr_qa.py`, `QA_PARALLEL.md`, `run_phased_dev.sh`).

## Hard rules
- ADD-ONLY: create files only in `reproduce/mhr_phi/` and `review/mhr_phi_20261118/`. Never edit, move or delete any
  existing file (all Qwen code and results stay byte-identical). Never read/copy/print any `.env`.
- Never call the SLM, Gemini, Modal or any network API. Write the online steps but do NOT execute them.
- Python `"E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe"`, cwd = repo root, `PYTHONUTF8=1`.
- The Phi index will be at `MultiHop-RAG-phi-ropefix609/` (being built now; a 30-article pilot with the identical
  layout exists at `review/mhr_phi_20261118/index_source/indexes/multihop_rag-phi-pilot30/` for offline tests).

## 1. `reproduce/mhr_phi/profile.py` — one switch that points the whole ladder at Phi
`apply()` must, before any ladder module runs, rebind every module-level constant that names a Qwen artifact, in the
defining module AND in every module that imported it by name (`from .common import OUT` copies the binding — find all
of them with an AST/grep scan, do not guess):
- index `MultiHop-RAG-qwen-full609` → `MultiHop-RAG-phi-ropefix609`
- `mhr_st.common.OUT` → `review/mhr_phi_20261118/st`; `ARCHIVE` → `review/mhr_phi_20261118/c0/archive`;
  the frozen source dir stays `review/mhr_st_20261112/source` (read-only, shared code)
- `mhr_f.pool.OUT` → `review/mhr_phi_20261118/f`; `mhr_ce.offline.OUT`/`mhr_ce.scores.OUT` → `.../ce`;
  `mhr_set.offline.OUT` → `.../set`; model_F/model_CE/set_params paths follow their OUT
- dev C0 reference `runs/multihop_rag/dev400/C0_s101` → `review/mhr_phi_20261118/c0/dev/C0_s101`
- serving spec `serving_qwen_modal.json` → `review/mhr_phi_20261118/serving_phi_ropefix_modal.json` (write it: model
  microsoft/Phi-3.5-mini-instruct, revision 2fe192450127e6a83f7441aef6e3ca586c338b77, vllm 0.11.0, A100-80GB,
  max_model_len 32768, rope fix description from reproduce/modal_slm_phi.py, sampling = vLLM defaults because Phi's
  generation_config sets none). The serving check must still require model name + vLLM version.
- report titles "Qwen2.5-3B" → "Phi-3.5-mini-instruct (LongRoPE fix)"; model strings in manifests likewise.
- Shared read-only inputs keep their paths: dataset/, the question CSVs (copy `review/mhr_st_20261112/inputs/{dev,test}.csv`
  into `review/mhr_phi_20261118/st/inputs/` with sha check), gold labels, the cross-encoder, the judge.
Install an audit hook (like reproduce/mhr_set/offline.guard, but allowing normal appends) that raises on any write
under the Qwen roots: review/mhr_st_20261112, review/mhr_f_20261114, review/mhr_ce_20261116, review/mhr_set_20261117,
MultiHop-RAG-qwen-full609 — and on reading any `.env` except through the launcher below.
`reproduce/mhr_phi/launch.py`: `python -m reproduce.mhr_phi.launch <module> [args]` = load `.env` from
MINIRAG_DOTENV (override=False), set MINIRAG_SLM_URL from PHI_SLM_URL (required for online steps), apply(), then
runpy the module as `__main__`. Subprocesses spawned by ladder code (`-m reproduce.mhr_st...`, judge entries, workers)
must also run under the profile: provide a sitecustomize-free mechanism (e.g. env `MHR_PROFILE=phi` checked by a
`reproduce/mhr_phi/__init__`-level hook is NOT possible without editing existing files — so instead make launch.py
rewrite `sys.executable`-based child commands: wrap subprocess.Popen/run so that `[python, -m, X, ...]` with X in the
ladder packages becomes `[python, -m, reproduce.mhr_phi.launch, X, ...]`). Test this.
`reproduce/mhr_phi/test_profile.py`: import every module of the four packages under the profile and assert that no
module global (Path or str) still contains a Qwen root / 'qwen-full609' / 'serving_qwen', except an explicit
allow-list of shared read-only inputs; assert the subprocess rewrite works; assert writes to Qwen roots raise.

## 2. `reproduce/mhr_phi/c0.py` — the C0 run with Phi (the Qwen C0 came from an archived phase-3 run)
Reproduce exactly how the archived Qwen C0 (review/mhr_st_20261112/archive_c0/qa/s{S}/C0_s{S}/, its keyword caches
review/mhr_st_20261112/s{S}/kw_s{S}.jsonl and inputs/kw_dev_s101.jsonl, and MiniRAG dev400 C0_s101) was produced —
find the runner and its settings in the frozen source (phased_qa phases A keyword → B context → C answer, mode 'mini',
same QueryParam, MINIRAG_* scientific env, worker layout) — with only the SLM endpoint and index changed. Subcommands:
`keywords --split dev|test --seed S` (Phi keyword phase → `review/mhr_phi_20261118/st/s{S}/kw_s{S}.jsonl`, dev →
`st/inputs/kw_dev_s101.jsonl`, same record format as the Qwen caches), `contexts --split --seed` (frozen kw cache,
MINIRAG_KW_CACHE_ONLY=1, 0 LLM calls), `answers --split --seed`, `execute` (dev s101 then test 101/202/303, resumable,
status file), producing the archive layout the ladder expects (`c0/archive/qa/s{S}/C0_s{S}/{answers.jsonl,
context.jsonl, requests.jsonl}` and `c0/dev/C0_s101/...` in the layout `mhr_st.pipeline.reference()` reads).
Serving check before any generation (served names include minirag-slm and microsoft/Phi-3.5-mini-instruct, vLLM 0.11.0).
Every request traced (llm_trace) and refused if response_model is not Phi.
Offline test: run `contexts` for 5 dev questions on the PILOT index with a synthetic keyword cache built by copying the
Qwen dev cache records for those 5 questions into a temp file (never into the Phi kw path) — proves phase B works on a
Phi index layout; delete nothing, write test outputs under review/mhr_phi_20261118/_offline_test/.

## 3. `reproduce/mhr_phi/RUNBOOK.md`
Exact command sequence for the whole Phi ladder through the launcher (C0 execute; ST: mhr_st.run prepare/preflight/
execute + finish; F: mhr_f.pool run dev/test, mhr_f.study, mhr_f.run freeze/execute/finish; CE: mhr_ce.scores dev/test
(Modal A10G), mhr_ce.study, mhr_ce.run freeze/contexts/execute/finish; set: mhr_set.test_select/study/run freeze/
contexts/execute/finish), naming every place where a Qwen-only assumption remains (e.g. any hard-coded check that
compares against a Qwen artifact, a Qwen-specific seed/revision literal, an `is Qwen` assert) and how the profile
handles it. If something cannot be handled without editing an existing file, write it in RUNBOOK.md under
"Blocked" instead of editing.

## 4. Finish
Run test_profile.py and the offline C0 context test. Write `review/mhr_phi_20261118/CODEX_REPORT.md`: files, commands,
test results, remaining Qwen assumptions, anything blocked.
