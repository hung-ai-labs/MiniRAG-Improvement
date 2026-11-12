# Table 1 — Phi-3.5-mini-instruct (LongRoPE fix), dev200 (200 questions)

1 generation round(s) × 3 gemini-flash-lite-latest judge passes. ACC/ERR = mean over rounds × passes, ± = sd between generation rounds. McNemar on the per-round majority verdict, one entry per round (became / ceased, exact p).

| # | Configuration | Full evidence | ACC | ERR | Neither | ACC vs baseline | ACC vs row above | ERR vs row above |
|---|---|---:|---:|---:|---:|---|---|---|
| 1 | C0 (MiniRAG original) | 60/177 | 47.67 | 29.17 | 23.17 | 3/11, p=0.057 | — | — |
| 2 | + source-time (baseline) | 71/177 | 51.50 | 26.83 | 21.67 | — | 11/3, p=0.057 | 3/8, p=0.23 |
| 3 | + learned scorer: graph/time/structure features | 97/177 | 64.50 | 21.00 | 14.50 | 35/7, p=1.5e-05 | 35/7, p=1.5e-05 | 11/23, p=0.058 |
| 4 | + semantic signal (cross-encoder) | 97/177 | 63.67 | 21.33 | 15.00 | 39/14, p=0.0008 | 14/17, p=0.72 | 17/15, p=0.86 |
| 5 | + budgeted set selection (full method) | 96/177 | 58.00 | 21.00 | 21.00 | 32/19, p=0.092 | 9/21, p=0.043 | 13/15, p=0.85 |

## ACC by question type

| Configuration | Single (159) | Multi (21) | Null (20) |
|---|---:|---:|---:|
| C0 (MiniRAG original) | 43.61 | 52.38 | 75.00 |
| + source-time (baseline) | 49.27 | 46.03 | 75.00 |
| + learned scorer: graph/time/structure features | 65.62 | 49.21 | 71.67 |
| + semantic signal (cross-encoder) | 64.99 | 49.21 | 68.33 |
| + budgeted set selection (full method) | 59.12 | 49.21 | 58.33 |

Judge noise: sd between passes of baseline ACC = 0.50.
Rows 3–5 score the first 45 candidates of the baseline ranking. Widening 30 → 45 without re-scoring leaves every context identical to the baseline. Scores: cross-fitted (5 folds); beta/tau nested per fold.
