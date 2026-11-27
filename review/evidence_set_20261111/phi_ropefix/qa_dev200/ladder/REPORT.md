# Table 1 — Phi-3.5-mini-instruct (LongRoPE fix), dev200 (200 questions)

3 generation round(s) × 3 deepseek-flash judge passes. ACC/ERR = mean over rounds × passes, ± = sd between generation rounds. McNemar on the per-round majority verdict, one entry per round (became / ceased, exact p).

| # | Configuration | Full evidence | ACC | ERR | Neither | ACC vs baseline | ACC vs row above | ERR vs row above |
|---|---|---:|---:|---:|---:|---|---|---|
| 1 | C0 (MiniRAG original) | 60/177 | 43.83 ± 0.17 | 30.67 ± 2.74 | 25.50 | 4/10, p=0.18; 1/11, p=0.0063; 2/10, p=0.039 | — | — |
| 2 | + source-time (baseline) | 71/177 | 47.83 ± 0.76 | 28.50 ± 2.25 | 23.67 | — | 10/4, p=0.18; 11/1, p=0.0063; 10/2, p=0.039 | 3/6, p=0.51; 0/7, p=0.016; 1/4, p=0.38 |
| 3 | + learned scorer: graph/time/structure features | 97/177 | 59.39 ± 0.51 | 21.83 ± 1.69 | 18.78 | 35/10, p=0.00025; 36/13, p=0.0014; 31/10, p=0.0015 | 35/10, p=0.00025; 36/13, p=0.0014; 31/10, p=0.0015 | 14/31, p=0.016; 13/31, p=0.0096; 13/23, p=0.13 |
| 4 | + semantic signal (cross-encoder) | 97/177 | 56.61 ± 1.83 | 26.06 ± 1.92 | 17.33 | 38/14, p=0.0012; 29/13, p=0.02; 28/16, p=0.096 | 15/16, p=1; 12/19, p=0.28; 11/20, p=0.15 | 16/15, p=1; 25/11, p=0.029; 24/10, p=0.024 |
| 5 | + budgeted set selection (full method) | 96/177 | 53.89 ± 2.08 | 23.61 ± 1.71 | 22.50 | 29/19, p=0.19; 30/17, p=0.079; 30/15, p=0.036 | 7/21, p=0.013; 10/13, p=0.68; 16/13, p=0.71 | 17/12, p=0.46; 8/15, p=0.21; 8/22, p=0.016 |

## ACC by question type

| Configuration | Single (159) | Multi (21) | Null (20) |
|---|---:|---:|---:|
| C0 (MiniRAG original) | 40.46 | 50.79 | 63.33 |
| + source-time (baseline) | 45.91 | 49.74 | 61.11 |
| + learned scorer: graph/time/structure features | 60.17 | 54.50 | 58.33 |
| + semantic signal (cross-encoder) | 57.30 | 44.97 | 63.33 |
| + budgeted set selection (full method) | 54.30 | 48.68 | 56.11 |

Judge noise: sd between passes of baseline ACC = 0.00.
Rows 3–5 score the first 45 candidates of the baseline ranking. Widening 30 → 45 without re-scoring leaves every context identical to the baseline. Scores: cross-fitted (5 folds); beta/tau nested per fold.
