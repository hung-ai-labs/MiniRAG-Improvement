# Table 1 — Qwen2.5-3B-Instruct, dev200 (200 questions)

3 generation round(s) × 3 deepseek-flash judge passes. ACC/ERR = mean over rounds × passes, ± = sd between generation rounds. McNemar on the per-round majority verdict, one entry per round (became / ceased, exact p).

| # | Configuration | Full evidence | ACC | ERR | Neither | ACC vs baseline | ACC vs row above | ERR vs row above |
|---|---|---:|---:|---:|---:|---|---|---|
| 1 | C0 (MiniRAG original) | 72/177 | 46.06 ± 0.95 | 30.72 ± 0.10 | 23.22 | 1/14, p=0.00098; 0/13, p=0.00024; 0/14, p=0.00012 | — | — |
| 2 | + source-time (baseline) | 88/177 | 52.67 ± 1.17 | 27.61 ± 0.25 | 19.72 | — | 14/1, p=0.00098; 13/0, p=0.00024; 14/0, p=0.00012 | 1/7, p=0.07; 0/7, p=0.016; 0/7, p=0.016 |
| 3 | + learned scorer: graph/time/structure features | 103/177 | 56.22 ± 1.67 | 28.00 ± 1.20 | 15.78 | 24/22, p=0.88; 25/15, p=0.15; 28/19, p=0.24 | 24/22, p=0.88; 25/15, p=0.15; 28/19, p=0.24 | 20/16, p=0.62; 16/15, p=1; 21/23, p=0.88 |
| 4 | + semantic signal (cross-encoder) | 110/177 | 59.94 ± 1.02 | 27.72 ± 1.62 | 12.33 | 35/24, p=0.19; 32/13, p=0.0066; 35/20, p=0.058 | 21/12, p=0.16; 18/9, p=0.12; 18/12, p=0.36 | 20/18, p=0.87; 14/17, p=0.72; 17/17, p=1 |
| 5 | + budgeted set selection (full method) | 109/177 | 61.00 ± 1.76 | 26.50 ± 2.67 | 12.50 | 38/20, p=0.025; 36/20, p=0.044; 36/18, p=0.02 | 21/14, p=0.31; 14/17, p=0.72; 12/9, p=0.66 | 12/24, p=0.065; 16/12, p=0.57; 17/16, p=1 |

## ACC by question type

| Configuration | Single (159) | Multi (21) | Null (20) |
|---|---:|---:|---:|
| C0 (MiniRAG original) | 46.19 | 24.34 | 67.78 |
| + source-time (baseline) | 54.58 | 22.75 | 68.89 |
| + learned scorer: graph/time/structure features | 57.86 | 33.86 | 66.67 |
| + semantic signal (cross-encoder) | 61.50 | 43.92 | 64.44 |
| + budgeted set selection (full method) | 62.61 | 48.15 | 61.67 |

Judge noise: sd between passes of baseline ACC = 0.17.
Rows 3–5 score the first 45 candidates of the baseline ranking. Widening 30 → 45 without re-scoring leaves every context identical to the baseline. Scores: cross-fitted (5 folds); beta/tau nested per fold.
