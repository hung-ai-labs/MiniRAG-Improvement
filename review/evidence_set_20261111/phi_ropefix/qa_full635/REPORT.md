# Table 1 — Phi-3.5-mini-instruct (LongRoPE fix), full LiHua-World (635 questions)

dev200 (outer-CV: cross-fitted (5 folds); beta/tau nested per fold) + nondev435 (held out: all-fold models fit on dev200 only). 3 generation round(s) × 3 deepseek-flash judge passes. ACC/ERR = mean over rounds × passes, ± = sd between generation rounds. McNemar on the per-round majority verdict, pooled over both splits, one entry per round (became / ceased, exact p).

**The main result stays the 435 held-out questions.** The method was designed by looking at dev200; outer-CV removes leakage from training and tuning, not from design choices.

| # | Configuration | Full evidence | ACC | ERR | Neither | ACC vs baseline | ACC vs row above | ERR vs row above |
|---|---|---:|---:|---:|---:|---|---|---|
| 1 | C0 (MiniRAG original) | 188/562 | 42.78 ± 0.55 | 29.68 ± 1.26 | 27.54 | 4/28, p=1.9e-05; 2/29, p=4.6e-07; 3/25, p=2.7e-05 | — | — |
| 2 | + source-time (baseline) | 218/562 | 46.60 ± 0.18 | 28.17 ± 0.91 | 25.23 | — | 28/4, p=1.9e-05; 29/2, p=4.6e-07; 25/3, p=2.7e-05 | 4/14, p=0.031; 3/15, p=0.0075; 3/9, p=0.15 |
| 3 | + learned scorer: graph/time/structure features | 299/562 | 55.50 ± 0.45 | 24.95 ± 0.75 | 19.55 | 105/47, p=2.9e-06; 100/44, p=3.5e-06; 108/52, p=1.1e-05 | 105/47, p=2.9e-06; 100/44, p=3.5e-06; 108/52, p=1.1e-05 | 59/85, p=0.037; 63/79, p=0.21; 60/83, p=0.065 |
| 4 | + semantic signal (cross-encoder) | 308/562 | 55.73 ± 0.84 | 26.18 ± 0.42 | 18.09 | 107/49, p=3.9e-06; 97/41, p=2.1e-06; 110/49, p=1.5e-06 | 44/44, p=1; 46/46, p=1; 51/46, p=0.68 | 52/42, p=0.35; 60/61, p=1; 63/49, p=0.22 |
| 5 | + budgeted set selection (full method) | 308/562 | 54.94 ± 0.89 | 25.53 ± 0.52 | 19.53 | 101/46, p=6.7e-06; 102/51, p=4.6e-05; 108/50, p=4.6e-06 | 36/39, p=0.82; 37/42, p=0.65; 36/39, p=0.82 | 41/47, p=0.59; 47/45, p=0.92; 38/47, p=0.39 |

## ACC by question type

| Configuration | Single (506) | Multi (64) | Null (65) |
|---|---:|---:|---:|
| C0 (MiniRAG original) | 39.81 | 39.76 | 68.89 |
| + source-time (baseline) | 44.80 | 39.41 | 67.69 |
| + learned scorer: graph/time/structure features | 57.31 | 35.59 | 61.03 |
| + semantic signal (cross-encoder) | 56.61 | 39.24 | 65.13 |
| + budgeted set selection (full method) | 55.53 | 40.97 | 64.10 |
