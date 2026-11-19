# Codex task: keyword-parse-failure guard for the Phi ladder (offline only)

Read: `review/mhr_phi_20261118/DEVIATIONS.md` (first section = the registered rule), `reproduce/mhr_phi/{profile,launch,
runtime,c0}.py`, `review/mhr_st_20261112/source/minirag/operate.py` (minirag_query, lines ~1632-1680),
`review/mhr_st_20261112/source/reproduce/research/mhr_phase3/shared_context.py`, `.../phased_qa.py`,
`reproduce/mhr_st/pipeline.py`, `reproduce/mhr_f/pool.py`.

Hard rules: same as reproduce/mhr_phi/CODEX_TASK.md (add-only: only create/edit files in reproduce/mhr_phi/ and
review/mhr_phi_20261118/; never edit any other existing file; never read .env; no network/LLM/Modal calls).
Python `"E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe"`, cwd repo root, PYTHONUTF8=1.

## Rule to implement (from DEVIATIONS.md)
`bad(reply)`: parse the frozen keyword reply exactly as minirag_query does (json_repair.loads; if that raises
json.JSONDecodeError, the same fallback string surgery then json_repair.loads); bad iff the result is not a dict, or
`answer_type_keywords` contains a non-str item, or `entities_from_query[:5]` contains a non-str item.
For a bad query: no retrieval at all; context status `early_exit`; context text = what MiniRAG returns on a keyword-parse
error (PROMPTS["fail_response"]); the answer step must then emit fail_response WITHOUT an LLM call (phased_qa.answer_phase
already does that for status != ok — verify). Must hold in EVERY Phi context path: C0 dev (phased_qa.context_phase),
C0 test (phase-3 shared prefix/suffix worker in c0.py), mhr_st contexts/ST workers, mhr_f.pool capture (record the
query with an empty pool and status early_exit so F/CE/set keep it as a failure), and any later context builder that
calls minirag_query / the phase-3 prefix. Implement it ONCE in reproduce/mhr_phi (e.g. a wrapper installed by
profile.apply() around the keyword-cache lookup or around minirag_query / _phase3_prefix that consults the frozen cache
reply), not by editing frozen code. Log every guarded query (query_id, seed, split, reason) to
review/mhr_phi_20261118/kwguard_events.jsonl (append) and count them in each cohort's contexts_summary.json.

## Tests (offline) — reproduce/mhr_phi/test_kwguard.py
1. bad() on the real caches: st/inputs/kw_dev_s101.jsonl → 0, st/s101/kw_s101.jsonl → 0, st/s202/kw_s202.jsonl → exactly
   {mhr-0315}; Qwen caches review/mhr_st_20261112/{inputs/kw_dev_s101.jsonl, s*/kw_s*.jsonl} → 0 (read-only).
2. Run the real C0 test context path for seed 202 restricted to mhr-0315 plus 3 normal queries on the full Phi index
   (MultiHop-RAG-phi-ropefix609) into review/mhr_phi_20261118/_offline_test/: mhr-0315 → early_exit/fail_response,
   others → ok and byte-identical to what the unguarded path gives (compare with existing s202 worker outputs if present).
3. Same for the mhr_f.pool capture path under the profile (mhr-0315 → empty pool, early_exit).
4. Guard never triggers on a good reply (all dev s101 / test s101 replies → no event).

## Then (offline, 0 LLM calls)
Resume the production C0 context phase for test seed 202:
`python -B -m reproduce.mhr_phi.launch reproduce.mhr_phi.c0 contexts --split test --seed 202`
(existing partial worker outputs must be reused/resumed exactly as c0.py's resume logic allows; if c0.py refuses because of
the earlier failed attempt, explain why in the report and fix only inside reproduce/mhr_phi without discarding data).
Verify: 2156 contexts, statuses counted, mhr-0315 early_exit, all others unaffected. Do NOT run answers.
Append to review/mhr_phi_20261118/CODEX_REPORT.md a section "Keyword guard": files, tests, s202 context summary.
