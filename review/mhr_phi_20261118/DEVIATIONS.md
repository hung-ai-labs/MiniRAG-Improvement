# Operational deviations / amendments, Phi ladder (review/mhr_phi_20261118)

## 2026-11-19 ~11:45 (UTC+7): keyword output that crashes MiniRAG's own query code (amendment, before any Phi verdict)
- Test seed 202, mhr-0315: Phi's keyword reply is malformed JSON (an unescaped quote inside a list item). MiniRAG's
  minirag_query repairs it with json_repair, which yields a non-string item (the integer 2023) among the first five
  entities_from_query; the embedding call then raises ValueError and the context worker dies (deterministically; 8
  watchdog restarts). Scan of all Phi caches so far: dev s101 0, test s101 0, test s202 1 (this one) of 4,712.
- Rule: a query whose frozen keyword reply, parsed exactly as minirag_query parses it (json_repair.loads, the
  answer_type_keywords list, entities_from_query[:5]), contains a non-string item is treated as MiniRAG's own
  keyword-parse failure: no retrieval, status `early_exit`, answer = PROMPTS["fail_response"] (what minirag_query
  returns on every other keyword-parse error). The keyword reply is not re-asked or edited. The same rule applies in
  every Phi row (C0, ST, F, CE, set share the frozen caches), so the query stays a failure throughout; counts are
  reported per seed. Qwen caches contain no such reply (checked), so the Qwen ladder is unaffected.

## Modal workspace / credit log
- 2026-11-18 22:20 – 2026-11-19 03:47: full Phi index on workspace nguyenhuuanhtai426 (cost of month 6.03 → 19.81 USD).
- 2026-11-19 03:50 – ~09:45: C0 dev s101, test s101, keywords s202 on the same workspace (month cost 25.73 USD at 11:24;
  hard cap 30 USD). Remaining Phi generation moves to another workspace (same app file, logged here when it happens).

## 2026-11-19 ~15:40 (UTC+7): JSONL reader fix (no data changed; before any Phi verdict)
- Test seed 202 C0: Phi wrote the character U+0085 (NEL) inside 2 of 2,156 answers. The writer is correct
  (json.dumps(ensure_ascii=False) + "\n"; 2,156 records, one per "\n"), but three frozen readers split with
  str.splitlines(), which also breaks on U+0085 (and U+2028/2029, \x0b, \x0c, \x1c-\x1e): phased_qa.read_jsonl,
  judge_run.read_records, reproduce.mhr_st.common.records. The s202 answer step therefore failed when it re-read its own
  complete file (14:25-14:56, after all 2,156 answers had been written).
- Fix: reproduce/mhr_phi/jsonl.py replaces the three readers in memory under the Phi profile with copies that split on
  "\n" only. Test (reproduce/mhr_phi/test_jsonl.py): the only file with such characters is s202 answers.jsonl (2) and it
  now reads as 2,156 records; 26 other Phi and Qwen JSONL files read identically with old and new readers. The Qwen
  ladder never met such a character, so its results are unaffected.
- 2026-11-19 15:01: C0 paused by the cost watchdog at 29.30 USD on workspace nguyenhuuanhtai426 (user-set stop 29,
  cap 30). s202 answers are complete on disk; the next `c0 execute` (new workspace) finalizes s202 without new
  generation, then runs s303.
