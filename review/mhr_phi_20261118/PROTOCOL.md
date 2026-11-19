# MultiHop-RAG ablation ladder with Phi-3.5-mini-instruct (registered 2026-11-18, before any Phi call)

Second SLM for the MultiHop-RAG ladder C0 → +ST → +F → +CE → +set (Qwen2.5-3B results:
review/mhr_{st_20261112,f_20261114,ce_20261116,set_20261117}). Built the way the LiHua Phi run was fixed
(README §4.5, reproduce/evidence_set/slm.py `phi_ropefix`): Phi does everything the SLM does in MiniRAG
(extraction + gleaning, query keywords, answers) on its own index, and every learned component is re-fit on
Phi's own dev pools with the rules already registered for Qwen. Nothing is shared with the Qwen ladder except
the code, the corpus and its chunking (same chunk ids), the questions, the gold labels, the seeds and the judge.
Written after all five Qwen rows were known.

## Serving (fixed)
- `reproduce/modal_slm_phi.py` unchanged (ROPE_FIX = True: `--hf-overrides` short_factor := long_factor; the
  LongRoPE batch-switch bug of vLLM 0.11.0, A/B in review/phi_rope_ab_20261112). vLLM 0.11.0, A100-80GB,
  max_model_len 32768, served names `microsoft/Phi-3.5-mini-instruct` and `minirag-slm`.
- Weights: Hugging Face main of microsoft/Phi-3.5-mini-instruct, commit 2fe192450127e6a83f7441aef6e3ca586c338b77
  (read 2026-11-18; the app does not pass --revision, so the serving check records the commit served).
- Sampling: as in the LiHua phi_ropefix run, no override. Phi's generation_config.json sets no sampling
  values, so vLLM's defaults apply (temperature 1.0, top_p 1.0), while Qwen ran with its own generation_config
  (0.7 / 0.8 / top_k 20 / repetition 1.05). Per-request seeds 101/202/303 as for Qwen. Reported as a
  difference between the two SLM ladders; Phi and Qwen are never tested against each other.
- The positional-encoding override is an intervention on the model and is reported.

## Index (gate I)
- Code: MiniRAG commit d90d892 (the commit the Qwen index MultiHop-RAG-qwen-full609 was started from, 1 min
  later), copied to review/mhr_phi_20261118/index_source. Its extraction path equals the +ST source snapshot;
  the recorded source digest of the Qwen manifest (8a3913b6…) could not be reproduced from any commit
  (likely an uncommitted file at that time) — reported.
- Same docs.jsonl (sha 5816d2d4…), all 609 articles, chunk 1200/100 tiktoken gpt-4o-mini, gleaning 1,
  MINIRAG_MAX_TOKENS 4096, local embeddings, batch 8 docs, llm_max_async 8 — as the Qwen index.
- Pilot first (index_run --subset pilot, 30 articles, seed 13). Gate: reproduce.index_health zero-entity chunks
  ≤ 5% overall and the run's verification all true. Then full 609. Gate on the full index: verification all
  true and zero-entity chunks ≤ 5% (Qwen full609: 5/1656 without graph source). A failed gate stops the run.
- The index can be resumed on another Modal workspace (identity excludes the endpoint); any switch is logged
  with time and workspace in DEVIATIONS.md.

## Ladder (same definitions and decision rules as the Qwen rows)
For each row the registered protocol of the Qwen row applies verbatim, with "Qwen" read as "Phi" and every
artifact rebuilt from the Phi index and Phi keyword caches:
- C0, +ST: review/mhr_st_20261112 protocol (source-time anchors, test 2156 × seeds 101/202/303).
- +F: review/mhr_f_20261114 rules — pool rule (smallest native cut within ceil(2%) of uncut, else all) chosen
  on Phi dev400 pools; 18 features; StandardScaler+LogReg C=1; grouped folds (same construction); freeze.
- +CE: review/mhr_ce_20261116/PROTOCOL.md — same cross-encoder revision; features [ce_logit]+18; freeze.
- +set: review/mhr_set_20261117/PROTOCOL.md — same grid and (β, τ) rule on Phi dev; freeze; including the
  2026-11-18 `unparsed` amendment (≥ 5 parse_failed cycles → not accurate) from the start.
- Step A of each learned row runs on Phi dev400 (seed 101 pools) before any Phi test context of that row is
  built; step A is descriptive.
- Reuse rules as for Qwen (identical context → reuse answer; identical answer → share verdict), within Phi only.

### Clarification (2026-11-18 23:00, before any Phi keyword, pool or answer existed)
The CE and set protocols name the pool "native@all" because that is what the registered pool rule chose on Qwen dev.
For Phi, "the pool" in +F, +CE and +set means the pool the registered rule selects on Phi dev400 (native@all or a
smaller native cut), frozen once in the +F step and used unchanged by +CE and +set. C0 for Phi is run fresh with the
frozen phase-3 C0 code and settings (review/mhr_st_20261112/source, commit b640765; reproduce/mhr_phi/c0.py), since
the Qwen C0 came from that archived run.

## Decisions
Per row: the new row improves on the previous Phi row iff ACC rises in all three seeds and exact McNemar
p < 0.05 in each. Reported also vs C0, by type (exploratory), ERR, Neither, evidence recall, tokens; secondary
checks as for Qwen (entity-substring metric, judge drift, Fleiss κ; yes/no judge-only limitation).
Cross-SLM statements are descriptive only ("the F gain replicates / does not replicate on Phi").

## Budget
Modal spend is reported per stage (index, dev, test). If a workspace's credit runs low, the run is paused at a
resumable point and continued on another workspace with the identical app; logged in DEVIATIONS.md.
