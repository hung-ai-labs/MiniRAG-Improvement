# MultiHop-RAG ablation row 5: C0 + ST + F + CE + set (registered 2026-11-17, before any set number)

Port of the LiHua "+ budgeted set selection" row (reproduce/evidence_set/select.py, arm CE0F_SET;
beta/tau rule from evidence_set/evaluate.tune and evidence_set/nondev.dev_beta_tau) onto the +CE run
(review/mhr_ce_20261116). Written after the CE test result was known (CE vs F not confirmed) and
before any set selection was computed on dev or test.

## What changes
Only which chunks of the candidate pool enter the Sources block, and their order.
Everything else is the +CE run: index, keyword caches, Entities, pool (native@all, saved in
mhr_f_20261114/{dev,test}/s<seed>/pools.jsonl), chunk probabilities p (CE+F scorer: cross-fitted
dev scores mhr_ce_20261116/dev_offline/scores_CEF.jsonl; frozen model_CE.json on test), prompt,
SLM (Qwen2.5-3B, vLLM 0.11.0 on Modal, same seeds), judge (gemini-3.5-flash-lite, 3 passes), budget 4000.

## Selection (fixed in advance, copied from evidence_set/select.py)
Greedy by marginal gain per 1,000 tokens of the monotone submodular objective

    F(S) = sum_d [1 - prod_{c in S, doc(c)=d} (1 - p_c)]
         + beta * sum_f w_f [1 - prod_{c in S, c supports f} (1 - p_c)]

- d = article (full_doc_id of the chunk); tokens = the tiktoken count used by +F/+CE (study.corpus()).
- Facets, label-free, from the saved pool record of the question:
  keyword facet kw:<k> for each keyword group k in `groups`, support = chunks on its seed paths that
  are in the pool; date facet date:<a> for each anchor a in `anchors`, support = pool chunks of
  articles published on a (study.corpus() by_time, the same map as the date_match feature).
  Facets with empty support are dropped. w_f = log(1 + |pool| / |support(f)|).
- A chunk that does not fit the remaining budget is skipped (no A1 stop). Stop when nothing fits, or
  when the best gain per 1k tokens < tau (tau > 0 also stops on zero gain). Ties: first in pool order.
- Output order = selection order; Sources rows numbered 0..n-1 in that order.

## beta, tau
Grid beta in (0, 0.25, 0.5, 1, 2) x tau in (0, 0.05, 0.1, 0.2, 0.5, 1) (LiHua grid, unchanged).
Rule (LiHua): among settings whose full-evidence count is within ceil(1% of the labelled questions
used) of the best, take the fewest mean Sources tokens, then the higher full-evidence count, then the
smaller (beta, tau). Uses gold chunk labels of dev questions only; test labels are never used for any
choice.

## Step A (dev400, seed-101 pools, cross-fitted CE+F p, 0 LLM calls)
Same 5 folds as +F/+CE (study.folds). For each outer fold, (beta, tau) is chosen by the rule on the
labelled questions of the other four folds and applied to the fold (nested). Reported for CE+F (A1),
SET, and two ablations, SET beta=0 (tau chosen nested on its own) and SET tau=0: full evidence, fact
micro/macro recall, mean/median tokens and chunks, by type; SET vs CE+F exact sign test on full
evidence. Step A is descriptive: the row runs on test whatever it shows (as on LiHua), unless a code
check fails.

## Freeze
(beta, tau) = the rule applied to all labelled dev400 questions with the cross-fitted p (as
nondev.dev_beta_tau). Written to set_params.json with the hashes of scores_CEF.jsonl and
model_CE.json before any test set context is built.

## Step B (test 2156 x seeds 101/202/303)
- Contexts offline from the saved test pools: p from model_CE.json (closed form) on the stored test CE
  logits and the 18 F features. Check: A1 over the pool sorted by that p must reproduce the stored CE
  chunk_ids and every stored CE context byte for byte. Sources block = CSV of the SET selection, rows
  0..n-1, written into the CE context in place of its Sources block. Non-ok CE questions keep their
  CE status and context.
- Answers: a SET context identical to the CE context reuses the CE answer.
- Judge: a SET answer identical to the C0, ST, F or CE answer of the same query/seed shares that
  verdict; the rest is judged in SET_s<seed>/judge_delta.

## Registered decision
SET improves on CE (the previous ladder row) iff ACC rises in all three seeds and exact McNemar
(SET vs CE) p < 0.05 in each. Because CE was not confirmed over F, SET vs F is reported with the same
statistics as a secondary comparison (not the decision). Also reported: SET vs C0, by type
(exploratory, no multiplicity correction), ERR, Neither, evidence recall, Sources tokens.

## Secondary checks
- Judge-free metric on entity-answer questions (gold not yes/no, type not null): normalized gold
  substring (mhr_ce_20261116/secondary_checks/check.py rule); SET vs CE and SET vs F exact McNemar.
- Judge drift: re-judge 200 CE seed-101 answers (random.Random(13).sample(range(2156), 200)) on the SET
  judging date; report agreement with the stored CE verdicts.
- Fleiss kappa over the 3 passes for the SET verdicts.
- Known limitation carried over: yes/no answers are scored only by the LLM judge.
