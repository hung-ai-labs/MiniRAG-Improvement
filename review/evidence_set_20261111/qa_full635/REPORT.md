# Table 1 — Qwen2.5-3B-Instruct, full LiHua-World (635 questions)

dev200 (outer-CV: cross-fitted (5 folds); beta/tau nested per fold) + nondev435 (held out: all-fold models fit on dev200 only). 3 generation round(s) × 3 deepseek-flash judge passes. ACC/ERR = mean over rounds × passes, ± = sd between generation rounds. McNemar on the per-round majority verdict, pooled over both splits, one entry per round (became / ceased, exact p).

**The main result stays the 435 held-out questions.** The method was designed by looking at dev200; outer-CV removes leakage from training and tuning, not from design choices.

| # | Configuration | Full evidence | ACC | ERR | Neither | ACC vs baseline | ACC vs row above | ERR vs row above |
|---|---|---:|---:|---:|---:|---|---|---|
| 1 | C0 (MiniRAG original) | 239/562 | 46.84 ± 0.90 | 28.21 ± 0.26 | 24.95 | 4/41, p=9.3e-09; 1/39, p=7.5e-11; 0/40, p=1.8e-12 | — | — |
| 2 | + source-time (baseline) | 284/562 | 52.84 ± 0.94 | 26.32 ± 0.27 | 20.84 | — | 41/4, p=9.3e-09; 39/1, p=7.5e-11; 40/0, p=1.8e-12 | 3/13, p=0.021; 2/16, p=0.0013; 2/15, p=0.0023 |
| 3 | + learned scorer: graph/time/structure features | 332/562 | 56.08 ± 0.95 | 26.04 ± 1.10 | 17.88 | 76/62, p=0.27; 77/51, p=0.027; 85/61, p=0.057 | 76/62, p=0.27; 77/51, p=0.027; 85/61, p=0.057 | 61/63, p=0.93; 60/61, p=1; 70/74, p=0.8 |
| 4 | + semantic signal (cross-encoder) | 348/562 | 59.34 ± 1.00 | 26.81 ± 1.03 | 13.86 | 93/60, p=0.0095; 99/43, p=3e-06; 96/56, p=0.0015 | 62/43, p=0.078; 61/31, p=0.0023; 55/39, p=0.12 | 67/56, p=0.37; 52/57, p=0.7; 49/44, p=0.68 |
| 5 | + budgeted set selection (full method) | 350/562 | 59.58 ± 0.42 | 26.26 ± 0.45 | 14.16 | 99/57, p=0.00097; 102/54, p=0.00015; 101/53, p=0.00014 | 53/44, p=0.42; 41/49, p=0.46; 44/36, p=0.43 | 55/64, p=0.46; 51/46, p=0.68; 47/50, p=0.84 |

## ACC by question type

| Configuration | Single (506) | Multi (64) | Null (65) |
|---|---:|---:|---:|
| C0 (MiniRAG original) | 46.27 | 27.95 | 69.91 |
| + source-time (baseline) | 53.97 | 27.43 | 69.06 |
| + learned scorer: graph/time/structure features | 58.30 | 29.17 | 65.30 |
| + semantic signal (cross-encoder) | 61.62 | 38.19 | 62.39 |
| + budgeted set selection (full method) | 61.86 | 34.72 | 66.32 |
