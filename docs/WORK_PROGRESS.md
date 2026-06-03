# DM2026 Final Project — Work Progress Summary

**Last updated:** June 3, 2026  
**Competition:** [data-mining-2026-final-project](https://www.kaggle.com/competitions/data-mining-2026-final-project)  
**Metric:** MAE (0–5, lower is better)  
**Team:** Muhammad Rayhan Athaillah (313540001), NYCU Data Mining Spring 2026

---

## Executive summary

| Item | Value |
|------|--------|
| **Current best public MAE** | **0.8092** |
| **Best file** | `output/daily_candidates/prob_blend_best83_ord17.csv` |
| **Method** | Probability-space blend: 83% soft-wrapped season/LGBM anchor + 17% ordinal class probabilities |
| **Previous best (Jun 2)** | 0.8112 — `lgbm_blend_w15_w15_20260531_133212.csv` |
| **Improvement (Jun 3 sweep)** | −0.0020 vs 0.8112 anchor; −0.0009 vs first prob blend (92/8 @ 0.8101) |
| **Gap to Baseline 3** | ~0.0036 (estimate; baseline 3 ≈ 0.8128) |

**Breakthrough (June 3):** Cached probability blending — convert regression anchor to class probabilities (mean-preserving), blend with real ordinal XGB threshold probabilities, convert back to 5-week predictions. Public LB improved monotonically from 5% → 17% ordinal weight; plateau at **17–18%**.

---

## 1. Competition & data

- **Task:** 91-day met windows → predict next 5 weekly severity scores (0–5) per region  
- **Train:** 12.3M rows, 2248 regions, 14 meteorological features  
- **Test:** Last 91 days per region (same region IDs as train)  
- **Submission:** Rows must match `data/sample_submission.csv` order exactly  
- **Practical submit budget:** ~3–10/day (use one clear signal per slot when exploring)

---

## 2. Score timeline (major milestones)

| Date | Approach | Public MAE | File / notes |
|------|----------|------------|----------------|
| May 18 | Small 1D CNN | 0.8222 | `cnn_1d_20260518_180837.csv` |
| May 22 | CNN-GRU 25% blend | 0.8167 | GRU plateau branch |
| May 23 | Tiny tree 2% into GRU anchor | 0.8159 | `w25_tree_hybrid_blend_w02_*` |
| May 24 | Ordinal classification 1% | 0.8153 | Breakthrough structure |
| May 25 | Tree4 + ordinal 1.5% | 0.8145 | `tree4_ordinal_w015_*` |
| May 30 | Season-tree 20% into tree+ordinal | 0.8120 | Same-season history |
| **May 31** | **LGBM 15% into season anchor** | **0.8112** | `lgbm_blend_w15_w15_20260531_133212.csv` |
| Jun 2 | Grid XGB 1% into best | 0.8112 | Tied; CV models overfit standalone |
| **Jun 3** | **Prob blend 83/17** | **0.8092** | `prob_blend_best83_ord17.csv` |

### Failed / stopped branches (selected)

| Approach | Public MAE | Why stopped |
|----------|------------|-------------|
| PatchTST | 1.04–1.21 | Severe overfit |
| Meta-stacking Ridge | 1.16 | Trivial predictions |
| V2 CNN | 0.89+ | Val/public disconnect |
| Ordinal/season **full replacement** | 0.83–0.85 | Distribution shift |
| Standalone grid XGB / LGBM | 0.85–0.87 | CV overfit |
| Historical severity standalone | 0.878 | Too different from anchor |
| Seed42 GRU fine-tuning | ~0.8167 plateau | Marginal / wrong direction |
| Stale Gaussian soft prob cache | — | Mean shift ~+0.1; do not use |

---

## 3. June 3 — Probability blend (full session)

### 3.1 Idea

1. **Soft anchor:** Map current best CSV predictions to class probabilities `P(0)…P(5)` per (region, week) using **linear interpolation** between `floor(p)` and `ceil(p)` so the **expected value equals the regression prediction** (mean-preserving).  
2. **Ordinal cache:** Train cumulative-threshold XGBoost ordinal models → true class probabilities.  
3. **Blend:** `P_final = w_soft * P_soft + w_ord * P_ord` (renormalize), then `E[score] = sum_k k * P_final(k)`.  
4. **Align regions:** Ordinal cache uses test `groupby` order; soft cache uses submission CSV order → `reorder_class_probs()` in blend script.

### 3.2 Code added (mostly uncommitted)

| Path | Role |
|------|------|
| `model/probability_blend.py` | Soft probs, blend, reorder, cache I/O, probs→predictions |
| `scripts/cache_ordinal_probabilities.py` | Train ordinal → `.npz` |
| `scripts/cache_submission_soft_probs.py` | CSV → soft `.npz` (`--temperature` optional) |
| `scripts/blend_prob_submissions.py` | Weighted multi-cache blend → CSV |
| `tests/test_probability_blend.py` | 9 tests passing |

**Committed earlier (Jun 2):** `same_season.py`, `ordinal_tree.py`, `severity_history.py`, many `generate_*` scripts (`94acffc`).

### 3.3 Active caches

| File | Shape | Use |
|------|-------|-----|
| `output/prob_cache/soft_lgbm_best.npz` | 2248×5×6 | Soft wrap of `lgbm_blend_w15` (mean-preserving) |
| `output/prob_cache/ordinal_hybrid_20260603_145531.npz` | 2248×5×6 | Ordinal hybrid features |
| ~~`soft_best_20260602_prob.npz`~~ | — | **Stale** (old Gaussian soft) |

### 3.4 Bugs fixed

1. `NameError: np` in `cache_ordinal_probabilities.py`  
2. Region order mismatch between caches → `reorder_class_probs`  
3. Gaussian soft mapping inflated means → default **linear** soft mapping

### 3.5 Complete public sweep (June 3, 2026)

All blends: `soft_lgbm_best.npz` + `ordinal_hybrid_20260603_145531.npz`.

| Soft % | Ordinal % | File | Public MAE |
|--------|-----------|------|------------|
| 95 | 5 | `prob_blend_best0.95_ord0.05.csv` | 0.8105 |
| 94 | 6 | `prob_blend_best0.94_ord0.06.csv` | 0.8104 |
| 93 | 7 | `prob_blend_best93_ord07.csv` | 0.8102 |
| 92 | 8 | `prob_blend_best92_ord08.csv` | 0.8101 |
| 91.5 | 8.5 | `prob_blend_best915_ord085.csv` | 0.8100 |
| 91 | 9 | `prob_blend_best91_ord09.csv` | 0.8100 |
| 90 | 10 | `prob_blend_best90_ord10.csv` | 0.8099 |
| 89 | 11 | `prob_blend_best89_ord11.csv` | 0.8098 |
| 88 | 12 | `prob_blend_best88_ord12.csv` | 0.8097 |
| 87 | 13 | `prob_blend_best87_ord13.csv` | 0.8096 |
| 86 | 14 | `prob_blend_best86_ord14.csv` | 0.8095 |
| 85 | 15 | `prob_blend_best85_ord15.csv` | 0.8094 |
| 84 | 16 | `prob_blend_best84_ord16.csv` | 0.8093 |
| **83** | **17** | **`prob_blend_best83_ord17.csv`** | **0.8092** |
| 82 | 18 | `prob_blend_best82_ord18.csv` | 0.8092 (tie → stop) |

**Conclusion:** Optimum ≈ **17% ordinal** in probability space. Below 8% and above 18% do not beat 0.8092 on public LB.

### 3.6 Current best — reproduction

```bash
# Rebuild soft cache from scalar anchor (if anchor CSV changes)
PYTHONPATH=. python scripts/cache_submission_soft_probs.py \
  --submission output/daily_candidates/lgbm_blend_w15_w15_20260531_133212.csv \
  --output-path output/prob_cache/soft_lgbm_best.npz

# Best blend
PYTHONPATH=. python scripts/blend_prob_submissions.py \
  --cache output/prob_cache/soft_lgbm_best.npz:0.83 \
  --cache output/prob_cache/ordinal_hybrid_20260603_145531.npz:0.17 \
  --output-path output/daily_candidates/prob_blend_best83_ord17.csv

# Local gates vs previous best
PYTHONPATH=. python scripts/compare_candidate_distribution.py \
  --candidate output/daily_candidates/prob_blend_best83_ord17.csv \
  --reference output/daily_candidates/prob_blend_best92_ord08.csv
```

### 3.7 Optional micro-sweep (not submitted)

- 83.5/16.5, 82.5/17.5 around plateau  
- Rebuild `soft_lgbm_best.npz` from **0.8092 CSV** as new anchor for next prob round (may allow lower ordinal % with same LB)

---

## 4. Model architecture evolution (condensed)

| Phase | Period | Highlight | Best MAE |
|-------|--------|-----------|----------|
| 1 CNN | May 18–19 | Small CNN seed42 | 0.8222 |
| 2 Blend | May 20–22 | CNN-GRU into CNN | 0.8167 |
| 3 Tree/ordinal | May 23–26 | 2% tree, ordinal %, residual, blackout | 0.8144 |
| 4 Season | May 28–30 | Month-of-year severity | 0.8120 |
| 5 Grid + LGBM | May 31–Jun 1 | LGBM 15% into season anchor | 0.8112 |
| 6 Prob blend | **Jun 3** | **Probability-space ordinal mix** | **0.8092** |

### Best scalar anchor composition (pre–prob-blend)

```
tree4+ordinal1.5%  →  +20% season_tree  →  season_w20 anchor (0.8120)
                              →  +15% LightGBM  →  lgbm_blend_w15 (0.8112)
```

### Feature sets (reference)

| Set | Dims | Notes |
|-----|------|-------|
| Extended baseline | 168 | 14×12 stats |
| Hybrid temporal | 294 | Rolling + seasonal |
| Same-season | +4 | Month-of-year severity |
| Ordinal probs | 6 classes | Used in prob blend cache |

### Grid search winners (CV only — blend, don’t submit standalone)

- **XGBoost:** depth=7, trees=300, lr=0.05, subsample=0.8, colsample=0.9, reg α=0.1, λ=1.0 → CV MAE 0.2888  
- **LightGBM:** num_leaves=70, depth=6, lr=0.05, trees=300 → CV MAE 0.3153  

**LGBM scalar blend into season anchor:**

| LGBM weight | Public MAE |
|-------------|------------|
| 10% | 0.8113 |
| **15%** | **0.8112** |
| 20% | 0.8114 |

---

## 5. Key findings

### What works

1. **Ordinal structure** — cumulative thresholds / class probabilities match discrete 0–5 targets  
2. **Probability-space blending** — smoother than scalar blend for ordinal signal (Jun 3)  
3. **Same-season history** — strongest tabular signal before prob blend  
4. **Conservative scalar blends** — 10–15% new model into anchor (May 31)  
5. **GroupKFold by region** — no leakage across windows  
6. **Distribution gates** — `compare_candidate_distribution.py` before submit  

### What does not work

1. Standalone high-CV models on public LB  
2. Full replacement anchors (ordinal-only, season-only)  
3. Deep nets without distribution match (PatchTST, V2 CNN)  
4. Stale soft-prob caches (Gaussian mapping)  
5. Many submission weights same day without reading LB (Jun 3 used many slots intentionally for sweep)

### Critical insights

- **CV ≠ public** for standalone models; blending and prob-space mix generalize better  
- **Higher ordinal % in prob blend helped up to 17%** — opposite of tiny 1–2% scalar ordinal blends (different mechanism: full 6-way distribution vs small scalar nudge)  
- **Mean-preserving soft wrap** is required or predictions drift high  
- **Region alignment** between caches is mandatory  

---

## 6. Repository & git state (June 3)

- **Branch:** `main` (ahead of origin)  
- **Committed:** `94acffc` pipelines (season, ordinal, severity, generate scripts)  
- **Local-only:** `AGENTS.md` (gitignored; experiment log)  
- **Uncommitted (prob blend):** `model/probability_blend.py`, `scripts/cache_*`, `scripts/blend_prob_submissions.py`, `tests/test_probability_blend.py`, many candidates under `output/daily_candidates/`  

### Core scripts inventory

| Script | Purpose |
|--------|---------|
| `blend_prob_submissions.py` | Probability cache blend |
| `cache_submission_soft_probs.py` | CSV → soft npz |
| `cache_ordinal_probabilities.py` | Train ordinal → npz |
| `blend_submissions.py` | Scalar CSV blend |
| `compare_candidate_distribution.py` | Pre-submit safety |
| `generate_lgbm_submission.py` | LGBM grid |
| `grid_search_xgboost.py` | XGB grid |
| `generate_ordinal_tree_submission.py` | Ordinal tree CSV |
| `generate_season_tree_submission.py` | Season features |

### Tests

| File | Status |
|------|--------|
| `tests/test_probability_blend.py` | 9 passed |
| `tests/test_blend_submissions.py` | passing |
| `tests/test_ordinal_tree.py` | passing |
| Broader suite | 56+ passed; `test_cnn_gru` needs PyTorch |

---

## 7. Cached training data

| File | Description |
|------|-------------|
| `data/train_X_temporal_hybrid_52.npy` | Train features 119144×294 |
| `data/train_y_temporal_hybrid_52.npy` | Labels 119144×5 |
| `data/test_X_temporal_hybrid_0.npy` | Test 2248×294 |

---

## 8. Next steps (prioritized)

1. **Commit** prob-blend pipeline (`probability_blend.py`, scripts, tests).  
2. **Micro-tune** 16.5–17.5% ordinal if submission slots remain.  
3. **New anchor:** rebuild `soft_lgbm_best.npz` from `prob_blend_best83_ord17.csv`; re-sweep 5–12% ordinal.  
4. **Combine signals:** prob-blend best + tiny scalar residual/season (only if distribution-safe).  
5. **Update** `docs/PRELIMINARY_REPORT.md` (still cites 0.8113 in places).  
6. **Log** new results in local `AGENTS.md` after each Kaggle feedback.  

---

## 9. Submission checklist

```bash
# Validate format
PYTHONPATH=. python -c "
from model import experiments
import pandas as pd
s = pd.read_csv('data/sample_submission.csv')
sub = pd.read_csv('output/daily_candidates/prob_blend_best83_ord17.csv')
print(experiments.validate_submission(sub, s))
"

# Submit
kaggle competitions submit -c data-mining-2026-final-project \
  -f output/daily_candidates/prob_blend_best83_ord17.csv \
  -m "prob blend 83/17 best"
```

---

## 10. Environment

- Python 3.10, numpy, pandas, scikit-learn, xgboost, lightgbm  
- `pip install -e .` for package layout  
- Thread limits: `OMP_NUM_THREADS=2` (etc.) when training  
- Kaggle auth: `~/.kaggle/kaggle.json`

---

## 11. Documentation map

| File | Role |
|------|------|
| **`docs/WORK_PROGRESS.md`** | **This file — single consolidated progress log** |
| `AGENTS.md` | Local-only detailed experiment table (gitignored) |
| `docs/PRELIMINARY_REPORT.md` | Course report draft |
| `docs/superpowers/plans/` | Archived implementation plans |
| `docs/RESEARCH_2026-05-23_PLATEAU_ALTERNATIVES.md` | Research notes |

---

## 12. Contact

- **Instructor:** Jyun-Yu Jiang — wu80623@gmail.com (subject: "DM final")  
- **Competition:** https://www.kaggle.com/competitions/data-mining-2026-final-project  

---

*End of work progress summary.*
