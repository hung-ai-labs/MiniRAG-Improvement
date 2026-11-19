# MultiHop-RAG ablation row 4: C0 + ST + F + CE (registered 2026-11-16, before any CE number)

Port of the LiHua "+ semantic signal" row (reproduce/evidence_set/quick_scorer.py, arm CE0+F)
onto the +F run (review/mhr_f_20261114). Written before any cross-encoder score was computed.

## What changes
Only the order of the candidate pool before A1@4000, hence only the Sources block.
Everything else is the +F run: index, keyword caches, Entities, pool (native@all, saved in
mhr_f_20261114/{dev,test}/s<seed>/pools.jsonl), prompt, SLM (Qwen2.5-3B, vLLM 0.11.0 on Modal,
same seeds), judge (gemini-3.5-flash-lite, 3 passes), budget 4000.

## Scorer CE+F (fixed in advance, copied from quick_scorer.py)
- Cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2`, revision 233902d25c440f23af6f7d6e94d2946bac0bee0a,
  zero-shot, sentence-transformers CrossEncoder(max_length=512) -> input truncated to 512 wordpieces,
  raw logit (Identity activation), pair = (question text, chunk content). fp32.
  Run on a Modal GPU (local CPU measured at ~2 pairs/s). Same code path for dev and test.
- Feature vector = [ce_logit] + the 18 F features (study.FEATURES, same order).
- StandardScaler + LogisticRegression(C=1, max_iter=5000, random_state=13). No grid, no tuning.
- Order: pool sorted by -p, ties by native index; then A1@4000 (stop at first overflow).

## Step A (dev400, seed 101 pools, 0 LLM calls)
Cross-fitted with the +F folds (study.folds). Report full-evidence, fact micro/macro recall for
ST, F, CE+F, ORACLE, and CE+F vs F (exact sign tests). Step A is descriptive: the row is part of
the ladder and runs on test whatever step A shows (as on LiHua), unless a code check fails.

## Freeze
Refit on all dev400 rows; store closed-form scaler + coefficients in model_CE.json (max gap to
sklearn < 1e-9) before any test CE context is built.

## Step B (test 2156 x seeds 101/202/303)
- Contexts built offline from the saved test pools. Sources block = CSV of A1(CE+F order),
  rows numbered 0..n-1, written into the F context in place of its Sources block.
  Check: the same rebuild applied to the F chunk_ids must reproduce every stored F context byte
  for byte; F chunk_ids must equal A1(F order) from model_F.json. Non-ok F questions keep their
  F status and context.
- Answers: a CE context identical to the F context reuses the F answer.
- Judge: a CE answer identical to the C0, ST or F answer of the same query/seed shares that
  verdict; the rest is judged in CE_s<seed>/judge_delta.

## Registered decision
CE improves on F iff ACC rises in all three seeds and exact McNemar (CE vs F) p < 0.05 in each.
Reported also: CE vs C0, by type (exploratory, no multiplicity correction), ERR, Neither,
evidence recall.

## Secondary checks (added 2026-11-16, still before any CE number was seen)
- Judge-free metric: on entity-answer questions (gold not yes/no, type not null), normalized gold
  substring of the answer (script: review/mhr_f_20261114/judge_validity_s101/check.py rule);
  CE vs F exact McNemar per seed. Reported next to the judge result, not part of the decision.
- Judge drift: re-judge a fixed random sample of 200 F answers (random.Random(13), seed 101) on the
  CE judging date; report agreement with the stored F verdicts.
- Judge consistency: Fleiss kappa over the 3 passes for the CE verdicts.
- Known limitation carried over: yes/no answers are scored only by the LLM judge.
