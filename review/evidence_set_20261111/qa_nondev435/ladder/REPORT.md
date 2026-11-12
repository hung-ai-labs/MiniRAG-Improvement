# Table 1 — Qwen2.5-3B-Instruct, nondev (435 questions)

3 generation round(s) × 3 DeepSeek judge passes. ACC/ERR = mean over rounds × passes, ± = sd between generation rounds. McNemar on the per-round majority verdict, one entry per round (became / ceased, exact p).

| # | Configuration | Full evidence | ACC | ERR | Neither | ACC vs baseline | ACC vs row above | ERR vs row above |
|---|---|---:|---:|---:|---:|---|---|---|
| 1 | C0 (MiniRAG original) | 167/385 | 47.20 ± 0.96 | 27.05 ± 0.33 | 25.75 | 3/27, p=8.4e-06; 1/26, p=4.2e-07; 0/26, p=3e-08 | — | — |
| 2 | + source-time (baseline) | 196/385 | 52.92 ± 0.95 | 25.72 ± 0.35 | 21.35 | — | 27/3, p=8.4e-06; 26/1, p=4.2e-07; 26/0, p=3e-08 | 2/6, p=0.29; 2/9, p=0.065; 2/8, p=0.11 |
| 3 | + learned scorer: graph/time/structure features | 229/385 | 56.02 ± 0.78 | 25.13 ± 1.13 | 18.85 | 52/40, p=0.25; 52/36, p=0.11; 57/42, p=0.16 | 52/40, p=0.25; 52/36, p=0.11; 57/42, p=0.16 | 41/47, p=0.59; 44/46, p=0.92; 49/51, p=0.92 |
| 4 | + semantic signal (cross-encoder) | 238/385 | 59.05 ± 1.00 | 26.39 ± 0.92 | 14.56 | 58/36, p=0.03; 67/30, p=0.00022; 61/36, p=0.014 | 41/31, p=0.29; 43/22, p=0.013; 37/27, p=0.26 | 47/38, p=0.39; 38/40, p=0.91; 32/27, p=0.6 |
| 5 | + budgeted set selection (full method) | 241/385 | 58.93 ± 0.20 | 26.16 ± 1.17 | 14.92 | 61/37, p=0.02; 66/34, p=0.0018; 65/35, p=0.0035 | 32/30, p=0.9; 27/32, p=0.6; 32/27, p=0.6 | 43/40, p=0.83; 35/34, p=1; 30/34, p=0.71 |

## ACC by question type

| Configuration | Single (347) | Multi (43) | Null (45) |
|---|---:|---:|---:|
| C0 (MiniRAG original) | 46.30 | 29.72 | 70.86 |
| + source-time (baseline) | 53.70 | 29.72 | 69.14 |
| + learned scorer: graph/time/structure features | 58.50 | 26.87 | 64.69 |
| + semantic signal (cross-encoder) | 61.67 | 35.40 | 61.48 |
| + budgeted set selection (full method) | 61.51 | 28.17 | 68.40 |

Judge noise: sd between passes of baseline ACC = 0.32.
Rows 3–5 score the first 45 candidates of the baseline ranking. Widening 30 → 45 without re-scoring leaves every context identical to the baseline. Scores: all-fold models fit on dev200 only.
