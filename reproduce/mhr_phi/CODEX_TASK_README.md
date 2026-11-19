# Codex task: write review/MULTIHOP_README.md (overview of the MultiHop-RAG ablation study)

Create ONE new file `review/MULTIHOP_README.md`. Do not edit, move or delete any other file. No network. Never read `.env`.
Do not run git commands.

Audience: team members and the advisor who open the GitHub branch `phi-multihop-ladder` and need to understand the
MultiHop-RAG study without having followed it: what was tested, what was found, how sure we are, where every artifact is,
how to continue the Phi part. Write in Vietnamese (like the other docs in review/), concise, tables where useful,
relative links to files. Every number must be copied from a file in the repo (cite the file next to each table);
do not compute new statistics, do not round differently from the source, do not invent anything.

Sources to read:
- Qwen ladder results: review/mhr_set_20261117/BANG_SO_SANH.txt and RESULTS.json (all 5 rows, SET vs CE / F / C0, by type,
  secondary checks), review/mhr_ce_20261116/{BANG_SO_SANH.txt,RESULTS.json,PROTOCOL.md,secondary_checks/output.txt,
  secondary_checks/judge_drift/RESULT.json}, review/mhr_f_20261114/{BANG_SO_SANH.txt,RESULTS.json,judge_validity_s101/},
  review/mhr_st_20261112/{BANG_SO_SANH.txt,RESULTS.json,protocol.json}, review/mhr_set_20261117/{PROTOCOL.md,DEVIATIONS.md},
  step-A dev files review/mhr_{f_20261114,ce_20261116,set_20261117}/dev_offline/offline_eval.json, review/mhr_f_20261114/dev_offline/coverage.json.
- Phi ladder: review/mhr_phi_20261118/{PROTOCOL.md,DEVIATIONS.md,GATE_I.json,HANDOFF.md}, reproduce/mhr_phi/RUNBOOK.md,
  review/mhr_phi_20261118/c0/status.json.
- Code docstrings of reproduce/mhr_st, mhr_f, mhr_ce, mhr_set for one-line descriptions of each row.

Sections:
1. Mục tiêu và thiết kế: dataset (2,156 test / 400 dev, question types), ladder rows C0 → +ST → +F → +CE → +set
   (one line each: what changes), SLM Qwen2.5-3B (vLLM 0.11.0 on Modal), judge gemini-3.5-flash-lite × 3 passes,
   3 generation seeds, registered decision rule (all 3 seeds, exact McNemar p < 0.05 vs previous row), reuse rules
   (identical context → reuse answer; identical answer → share verdict), pre-registration practice (PROTOCOL before numbers).
2. Kết quả Qwen: the 5-row table (ACC ± SD, ERR, Neither, evidence recall, full evidence, tokens), per-row decision table
   with per-seed Δ and p, by-type summary, step-A dev summary (full-evidence counts per arm, chosen pool, frozen β/τ),
   one-paragraph interpretation that only states what the registered tests support.
3. Độ tin cậy của chấm điểm: Fleiss κ per row, judge drift checks, judge-free entity metric, the yes/no judge-only
   limitation, `unparsed` amendment.
4. Thang Phi-3.5-mini: why (second SLM), how (Hùng's LongRoPE fix, own index, re-fit scorers), gate I numbers,
   current status table (from c0/status.json and HANDOFF.md), deviations so far (one line each with link), differences
   from Qwen that limit cross-SLM comparison (sampling defaults, RoPE override), link to HANDOFF.md as the place to start.
5. Bản đồ thư mục: table path → content for every review/mhr_* folder, reproduce/mhr_* packages, the two indexes, dataset;
   say which per-question files exist (test/<ROW>_s<seed>/answers.csv and judge_gemini35/per_question.csv) and what is
   deliberately not in git (snapshot/, answer_work/, intermediate pools/contexts of Qwen, the Qwen index
   MultiHop-RAG-qwen-full609, raw logs) and that they can be regenerated or obtained from the authors.
6. Hạn chế đã biết (bullet list, from the protocols/deviations).
