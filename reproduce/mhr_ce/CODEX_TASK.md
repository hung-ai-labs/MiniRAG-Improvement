# Codex task: MultiHop-RAG ablation row 4 (+CE) — code + offline steps only

Read first: `review/mhr_ce_20261116/PROTOCOL.md` (registered rules — follow exactly),
`reproduce/mhr_f/study.py`, `reproduce/mhr_f/run.py`, `reproduce/mhr_ce/scores.py`.

## Hard rules
- ADD-ONLY. Create files only in `reproduce/mhr_ce/` and `review/mhr_ce_20261116/`. Never edit, move or
  delete any existing file anywhere (including reproduce/mhr_f/*, reproduce/mhr_st/*, review/mhr_f_20261114/*,
  review/mhr_st_20261112/*). Import and reuse their functions instead of copying where possible.
- Never read, copy or print any `.env`. Never call the SLM endpoint, Gemini, Modal or any network API.
  The `answers` and `finish` steps you write must NOT be executed by you.
- Python: `"E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe"`, run from repo root
  `E:/Project/MiniRAG research/MiniRAG-Improvement`, with env `PYTHONUTF8=1`.
- CE logits already exist: `review/mhr_ce_20261116/ce_logits_dev.jsonl` (done) and
  `ce_logits_test.jsonl` (being written by another process — for test steps, wait until
  `review/mhr_ce_20261116/ce_logits_test_summary.json` exists; poll every 2 min, max 90 min).
  Records: `{"query_id", "logits": {chunk_id: float}}`; may contain duplicate query_ids from resume —
  merge them.

## 1. `reproduce/mhr_ce/study.py` (step A, dev400) — write AND run
Mirror `mhr_f/study.py` main(): same dev pools (`review/mhr_f_20261114/dev/s101/pools.jsonl`, check
pools_summary sha), same pool (`native@all`, read from `review/mhr_f_20261114/dev_offline/coverage.json`),
same folds (`study.folds`), same featurize. Arms: C0, ST, F (18 features, cross-fitted), CE+F
(feature vector = [ce_logit] + 18 features, cross-fitted, same fit()), CE (ce logit alone, order by
-logit ties native index), ORACLE. Order = sort by -p ties native index, then `study.a1`.
Output to `review/mhr_ce_20261116/dev_offline/`: offline_eval.json (same stats() structure per arm and
per type, auc per fold for F and CE+F, coef of all-fold CE+F, and CE+F_vs_F, F_vs_ST, CE_vs_F
comparisons with full_up/down/p and facts sign test like study.py), scores_CEF.jsonl, selections.jsonl.
Assert the cross-fitted F arm selections equal `review/mhr_f_20261114/dev_offline/selections.jsonl`
F selections for every question (proves the port reproduces step A of +F).

## 2. `reproduce/mhr_ce/run.py` — subcommands freeze, contexts, answers, execute, finish, report
- `freeze`: like mhr_f.run.freeze → `review/mhr_ce_20261116/model_CE.json` (features list with
  'ce_logit' first, mean, scale, coef, intercept, pool, dev pools sha, ce_logits_dev sha,
  offline_eval sha, frozen_at; closed-form gap < 1e-9). Refuse if it exists. Write AND run.
- `contexts --seed S` (write AND run for 101, 202, 303 after freeze and after test logits exist):
  for each test question: F record = `review/mhr_f_20261114/test/F_s{S}/contexts.jsonl`
  (fields query_id, status, context, chunk_ids, ...); pool record = `.../test/s{S}/pools.jsonl`.
  Checks (fail hard, write context_check_failed.json): (a) A1(mhr_f.run.f_order(pool, model_F)) ==
  F chunk_ids for every ok question; (b) rebuild(F context, F chunk_ids) == F context byte for byte,
  where rebuild replaces the text between "-----Sources-----\n```csv\n" and the next "\n```" with
  `minirag.utils.list_of_list_to_csv([["id","content"]] + [[i, content] for i, c in enumerate(ids)])`
  (check exact header/row format against the stored F contexts; chunk content from
  `MultiHop-RAG-qwen-full609/kv_store_text_chunks.json`). Then CE ids = A1(CE+F order) and CE context =
  rebuild(F context, CE ids). Non-ok F questions: copy F status/context, chunk_ids []. Write
  `review/mhr_ce_20261116/test/CE_s{S}/contexts.jsonl` ({query_id, status, context, chunk_ids, same_as_f})
  and context_summary.json (counts, same_as_f count, check results, sha256s).
- `answers --seed S`: like mhr_f.run.answers but reads CE contexts, reuses F answers
  (`review/mhr_f_20261114/test/F_s{S}/answers.jsonl`) when same_as_f, source tag
  'frozen_F_identical_context'; serving log in review/mhr_ce_20261116/serving_checks.jsonl.
- `execute`: for each seed: contexts (skip if context_summary exists) then answers subprocess; retry
  like mhr_f; status in review/mhr_ce_20261116/status.json.
- `finish --rounds N`: like mhr_f.run.finish/judge_seed: delta = CE pred not in (C0, ST, F) preds;
  compose verdicts from C0 journal, ST delta journal, F delta journal
  (`review/mhr_f_20261114/test/F_s{S}/judge_delta/run/judge.jsonl`), CE delta journal; origin labels
  C0/ST/F_identical_answer, judged_CE. status_judge.json. Then report().
- `report`: like mhr_f.run.report with 4 rows C0, C0+ST, C0+ST+F, C0+ST+F+CE; paired CE_vs_F (primary,
  registered rule), CE_vs_C0; by type CE_vs_F; evidence recall for 4 arms; ERR/Neither.
  RESULTS.json + BANG_SO_SANH.txt in review/mhr_ce_20261116/.

## 3. Finish
Run: study.py, run freeze, run contexts for 101/202/303. Do NOT run answers/execute/finish/report.
Write `review/mhr_ce_20261116/CODEX_REPORT.md`: files created, commands run, key step-A numbers
(full_evidence and micro recall per arm, CE+F vs F p-values, AUCs), context summaries (same_as_f
counts, checks passed), anything you could not do.
