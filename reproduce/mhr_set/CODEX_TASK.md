# Codex task: MultiHop-RAG ablation row 5 (+set) — code + offline steps only

Read first: `review/mhr_set_20261117/PROTOCOL.md` (registered rules — follow exactly), then
`reproduce/evidence_set/select.py` (selection to port), `reproduce/evidence_set/evaluate.py` (tune(), BETAS,
TAUS), `reproduce/evidence_set/nondev.py` (dev_beta_tau), `reproduce/mhr_f/study.py`, `reproduce/mhr_ce/run.py`,
`reproduce/mhr_ce/study.py`, `reproduce/mhr_ce/offline.py`. The +CE package is the template: mirror its
structure, guards and checks; import and reuse its functions instead of copying where possible.

## Hard rules
- ADD-ONLY. Create files only in `reproduce/mhr_set/` and `review/mhr_set_20261117/`. Never edit, move or delete
  any existing file anywhere (reproduce/mhr_ce/*, mhr_f/*, mhr_st/*, evidence_set/*, review/mhr_ce_20261116/*,
  review/mhr_f_20261114/*, review/mhr_st_20261112/*). Writers must refuse to overwrite (open mode 'x').
  Note mhr_ce.offline.OUT points at review/mhr_ce_20261116 — do not write there; use your own OUT.
- Never read, copy or print any `.env`. Never call the SLM endpoint, Gemini, Modal or any network API.
  The `answers`, `execute` and `finish` steps you write must NOT be executed by you.
- Python: `"E:/Project/MiniRAG research/MiniRAG/.venv/Scripts/python.exe"`, run from repo root
  `E:/Project/MiniRAG research/MiniRAG-Improvement`, env `PYTHONUTF8=1`.
- `evidence_set/select.py` imports the LiHua corpus; do NOT call it. Write `reproduce/mhr_set/select.py` with
  the same algorithm (facets(), select()) parameterised by the MultiHop chunks map
  (`mhr_f.study.corpus()`: chunks[c]['doc'|'tokens'], by_time) — line-for-line the same gain, tie and stop logic.
  Facets per PROTOCOL: kw facets from pool record `groups` (support = chunks on seed `path`s that are in the
  pool), date facets from `anchors` via by_time[a] ∩ pool, empty supports dropped, w = log(1+|pool|/|support|).
  Add a unit test script `reproduce/mhr_set/test_select.py` that checks on a small synthetic case that
  your select() gives the same chosen list as evidence_set.select.select's algorithm (re-implement the
  reference inline from that file's text, not by importing it) and that a too-long chunk is skipped, not a stop.

## 1. `reproduce/mhr_set/study.py` (step A, dev400) — write AND run
Dev pools/pool rule/folds exactly as mhr_ce/study.py; p = cross-fitted `review/mhr_ce_20261116/dev_offline/scores_CEF.jsonl`
(assert every scored record's fold is not in its trained_on_folds; p in [0,1]). Grid BETAS x TAUS from
evidence_set.evaluate. Nested choice per outer fold = the tune() rule on labelled questions of the other folds.
Arms: CE+F (A1 over -p order, ties native index; must equal the CE+F selections in
`review/mhr_ce_20261116/dev_offline/selections.jsonl` for every question — assert), SET (nested), SET_beta0
(beta fixed 0, tau nested on its own), SET_tau0 (nested beta, tau=0), ORACLE as in mhr_ce study.
Output `review/mhr_set_20261117/dev_offline/`: offline_eval.json (same stats structure per arm and per type as
mhr_ce offline_eval.json, plus tokens mean/median and chunks mean/median per arm, the per-fold (beta,tau)
choices, the full grid table on all labelled dev, SET_vs_CE+F full-evidence exact sign test and facts sign test),
selections.jsonl (per question ids per arm, selection order).

## 2. `reproduce/mhr_set/run.py` — subcommands freeze, contexts, answers, execute, finish, report
- `freeze`: (beta, tau) by the rule on ALL labelled dev400 (as nondev.dev_beta_tau) →
  `review/mhr_set_20261117/set_params.json` {beta, tau, grid, sha256 of scores_CEF.jsonl, model_CE.json,
  dev pools, offline_eval.json, frozen_at}. Refuse if it exists. Write AND run (after study.py).
- `contexts --seed S` (write AND run for 101, 202, 303): CE record = `review/mhr_ce_20261116/test/CE_s{S}/contexts.jsonl`;
  pool record = `review/mhr_f_20261114/test/s{S}/pools.jsonl`; p = mhr_ce.run.ce_order machinery with
  model_CE.json and the test CE logits (expose p, not only order). Checks (fail hard, write
  context_check_failed.json): (a) A1(CE order) == stored CE chunk_ids for every ok question; (b)
  mhr_ce.run.rebuild(CE context, CE ids) == stored CE context byte for byte. Then SET ids = select(pool, p,
  facets, beta, tau) from set_params.json, SET context = rebuild(CE context, SET ids) (selection order).
  Verify: SET tokens <= 4000, no duplicates, ids ⊂ pool. Non-ok CE questions: copy CE status/context, chunk_ids [].
  Write `review/mhr_set_20261117/test/SET_s{S}/contexts.jsonl` ({query_id, status, context, chunk_ids,
  same_as_ce, tokens}) and context_summary.json (counts, same_as_ce, checks, token stats, sha256s).
- `answers --seed S`: like mhr_ce.run.answers but reads SET contexts, reuses CE answers
  (`review/mhr_ce_20261116/test/CE_s{S}/answers.jsonl`) when same_as_ce, source 'frozen_CE_identical_context';
  serving log review/mhr_set_20261117/serving_checks.jsonl.
- `execute`: like mhr_ce.run.execute; status review/mhr_set_20261117/status.json.
- `finish --rounds N`: like mhr_ce.run.finish/judge_seed: delta = SET pred not in (C0, ST, F, CE) preds;
  verdicts composed from C0 journal, ST delta, F delta, CE delta
  (`review/mhr_ce_20261116/test/CE_s{S}/judge_delta/run/judge.jsonl`), SET delta; origins
  {C0,ST,F,CE}_identical_answer / judged_SET; judge via a `reproduce/mhr_set/judge_entry.py` identical in
  behaviour to mhr_ce's; status_judge.json, step name 'judge_SET_delta'. Then report().
- `report`: 5 rows C0, C0+ST, C0+ST+F, C0+ST+F+CE, C0+ST+F+CE+set; paired SET_vs_CE (primary, registered rule),
  SET_vs_F (secondary), SET_vs_C0; by type SET_vs_CE; evidence recall + full evidence + mean Sources tokens for
  all 5 arms; ERR/Neither; RESULTS.json + BANG_SO_SANH.txt in review/mhr_set_20261117/.
- Also write `reproduce/mhr_set/online.py`: the same launcher as reproduce/mhr_ce/online.py (load .env path
  from MINIRAG_DOTENV, then run.main()). Do not run it.

## 3. Finish
Run: test_select.py, study.py, run freeze, run contexts for 101/202/303. Do NOT run answers/execute/finish/report.
Write `review/mhr_set_20261117/CODEX_REPORT.md`: files created, commands run, step-A numbers (full evidence,
micro recall, tokens per arm; per-fold and frozen beta/tau; SET vs CE+F p), context summaries (same_as_ce counts,
checks passed, token stats), anything you could not do.
