# Operational deviations, row 5 (+set)

## 2026-11-17 ~20:05 (UTC+7): SLM serving workspace moved
- Modal disabled workspace `nguyenhuuanhtai36` ("modal-http: workspace ... is disabled", HTTP 404) during
  seed 202 answer generation. Seed 101 answers (all 2156) and the first 42 new seed-202 answers were
  generated there before the outage. No answer was written from a failed request: answer_phase raises on
  any non-overflow error; all 1,924 answers then on disk had context_status ok and non-empty text.
- At 21:52 the unchanged serving app (MiniRAG/reproduce/modal_slm.py, git-clean at 33bd0ef: Qwen/Qwen2.5-3B-Instruct,
  vLLM 0.11.0, A10G, max_model_len 32768, same secret name) was redeployed on workspace `nguyenhuuanhtai426`
  (modal client 1.6.2, MODAL_IMAGE_BUILDER_VERSION=2025.06 because that client does not know the workspace
  default 2026.10). The remaining seed-202 and all seed-303 answers come from that deployment
  (MINIRAG_SLM_URL overridden in the environment of the launcher; .env untouched).
- The registered serving check (served model names and vLLM version, reproduce/mhr_f/run.configure_slm) passed
  on the new deployment before any answer was generated; requests.jsonl in each answer attempt records
  response_model = Qwen/Qwen2.5-3B-Instruct.
- Effect: same model, engine version, GPU type and sampling seeds; vLLM batching is not bit-deterministic across
  runs on either deployment, which is why identical CE contexts reuse CE answers instead of regenerating.
  Reported as a limitation; no change to the protocol or the decision rule.

## 2026-11-18 ~07:00 (UTC+7): permanently unparseable judge replies (amendment, before any SET verdict was read)
- Seed 101 SET delta: mhr-0862 (type null, gold "Insufficient information.") got 37 consecutive judge replies that
  reason at length and end with a label (all three passes end with "error"), which the strict one-label contract
  of judge_run marks parse_failed. The CE/F pipeline has no rule for this (it never happened there), so the run
  looped all night (watchdog restart budget exhausted 01:50-06:50).
- Rule added: a (query, pass) still parse_failed after 5 attempt cycles (= 15 asks) is recorded as label
  `unparsed`. Its majority is `unparsed` (or no_majority), which counts as NOT accurate and as neither ERR nor
  Neither. This is conservative for the arm under test (SET). The parser and prompt are unchanged; no free-text
  reply is mapped to a label. Number of unparsed records is reported per seed; Fleiss kappa is computed over
  rows with three parsed labels and reports how many rows were excluded.
- Written before any SET majority, ACC or McNemar number was computed or viewed.

## Online write guard
Codex's exclusive-create guard (reproduce/mhr_set/offline.py) would have crashed library appends (MiniRAG
logging.FileHandler, judge_run, phased_qa). Relaxed for online steps before any answer was generated; the
credential-file block stays. Every online attempt writes to a fresh attempt directory.
