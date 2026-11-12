# Table 1 — Phi-3.5-mini-instruct (LongRoPE fix), nondev (435 questions)

3 generation round(s) × 3 deepseek-flash judge passes. ACC/ERR = mean over rounds × passes, ± = sd between generation rounds. McNemar on the per-round majority verdict, one entry per round (became / ceased, exact p).

| # | Configuration | Full evidence | ACC | ERR | Neither | ACC vs baseline | ACC vs row above | ERR vs row above |
|---|---|---:|---:|---:|---:|---|---|---|
| 1 | C0 (MiniRAG original) | 128/385 | 42.30 ± 0.80 | 29.22 ± 0.61 | 28.48 | 0/18, p=7.6e-06; 1/18, p=7.6e-05; 1/15, p=0.00052 | — | — |
| 2 | + source-time (baseline) | 147/385 | 46.03 ± 0.62 | 28.02 ± 0.31 | 25.95 | — | 18/0, p=7.6e-06; 18/1, p=7.6e-05; 15/1, p=0.00052 | 1/8, p=0.039; 3/8, p=0.23; 2/5, p=0.45 |
| 3 | + learned scorer: graph/time/structure features | 202/385 | 53.72 ± 0.61 | 26.39 ± 1.35 | 19.90 | 70/37, p=0.0018; 64/31, p=0.00092; 77/42, p=0.0017 | 70/37, p=0.0018; 64/31, p=0.00092; 77/42, p=0.0017 | 45/54, p=0.42; 50/48, p=0.92; 47/60, p=0.25 |
| 4 | + semantic signal (cross-encoder) | 211/385 | 55.33 ± 1.59 | 26.23 ± 1.46 | 18.44 | 69/35, p=0.0011; 68/28, p=5.5e-05; 82/33, p=5.6e-06 | 29/28, p=1; 34/27, p=0.44; 40/26, p=0.11 | 36/27, p=0.31; 35/50, p=0.13; 39/39, p=1 |
| 5 | + budgeted set selection (full method) | 212/385 | 55.43 ± 1.35 | 26.41 ± 0.64 | 18.16 | 72/27, p=6.9e-06; 72/34, p=0.00028; 78/35, p=6.4e-05 | 29/18, p=0.14; 27/29, p=0.89; 20/26, p=0.46 | 24/35, p=0.19; 39/30, p=0.34; 30/25, p=0.59 |

## ACC by question type

| Configuration | Single (347) | Multi (43) | Null (45) |
|---|---:|---:|---:|
| C0 (MiniRAG original) | 39.51 | 34.37 | 71.36 |
| + source-time (baseline) | 44.28 | 34.37 | 70.62 |
| + learned scorer: graph/time/structure features | 56.00 | 26.36 | 62.22 |
| + semantic signal (cross-encoder) | 56.29 | 36.43 | 65.93 |
| + budgeted set selection (full method) | 56.10 | 37.21 | 67.65 |

Judge noise: sd between passes of baseline ACC = 0.22.
Rows 3–5 score the first 45 candidates of the baseline ranking. Widening 30 → 45 without re-scoring leaves every context identical to the baseline. Scores: all-fold models fit on dev200 only.
