# DM2026 Final Project — Work Progress Summary

**Date:** June 2, 2026
**Best Public MAE:** 0.8112 (unchanged since May 31)
**Gap to Baseline 3:** 0.0056
**Total Submissions:** 58+

---

## 1. Competition Overview

- Competition: `data-mining-2026-final-project` on Kaggle
- Metric: MAE (Mean Absolute Error, range 0-5)
- Dataset: 12.3M training rows, 2248 regions, 14 meteorological features, 91-day windows
- Task: Predict 5-week severity scores (0-5) per region
- Team: Muhammad Rayhan Athaillah (313540001), NYCU Data Mining Spring 2026

---

## 2. Submission History & Results

### Best Results by Approach

| Rank | Approach | Public MAE | Date |
|------|----------|------------|------|
| 1 | LightGBM 15% + Season-tree anchor | **0.8112** | May 31 |
| 2 | LightGBM 10% + Season-tree anchor | 0.8113 | May 31 |
| 3 | Grid XGB + LightGBM + Season-tree 10% | 0.8113 | May 31 |
| 4 | Grid XGB 2% + Season-tree anchor | 0.8114 | May 31 |
| 5 | LightGBM 20% + Season-tree anchor | 0.8114 | May 31 |
| 6 | LightGBM 5% + Season-tree anchor | 0.8116 | May 31 |
| 7 | Grid XGB + LightGBM 5% | 0.8116 | May 31 |
| 8 | Grid XGB + LightGBM 2% | 0.8118 | May 31 |
| 9 | Grid XGB 1% + Season-tree anchor | 0.8118 | May 31 |
| 10 | Season-tree 20% blend | 0.8120 | May 30 |

### Failed Approaches

| Approach | Public MAE | Reason |
|----------|------------|--------|
| PatchTST | 1.04-1.21 | Massive overfitting |
| Meta-stacking Ridge | 1.16 | Trivial predictions |
| CatBoost standalone | 0.8149-0.8156 | Underperformed |
| Ordinal anchor (full replacement) | 0.8279-0.8314 | Distribution shift |
| Grid XGB standalone | 0.849 | Overfitting |
| LightGBM standalone | 0.847 | Overfitting |
| Season anchor (full replacement) | 0.850-0.851 | Distribution shift |
| Ordinal tree variants (today) | 0.827-0.831 | Worse than baseline |
| Aggressive XGBoost (today) | 0.816-0.817 | Marginal or worse |
| Historical severity standalone (today) | 0.878 | Very different distribution |

---

## 3. Model Architecture Evolution

### Phase 1: Basic Approaches (May 18-19)
- **Small 1D CNN** (seed 42, 25 epochs): Public MAE **0.8222**
- **V2 CNN**: Public MAE 0.8901-0.8967 (overfit)
- **XGBoost baseline**: Public MAE 0.8434-0.8509

### Phase 2: Blending Era (May 20-22)
- **CNN-GRU blend** (seed42 25%): Public MAE **0.8167**
- **Multi-GRU**: Public MAE 0.8178 (stopped)
- **Seed-21 GRU**: Public MAE 0.8181 (stopped)
- **Tiny tree correction** (2%): Public MAE **0.8159**

### Phase 3: Ordinal Breakthrough (May 24-25)
- **Ordinal classification** (1% blend): Public MAE **0.8153**
- **Tree3 + ordinal1.5%**: Public MAE **0.8149**
- **Tree4 + ordinal1.5%**: Public MAE **0.8145**

### Phase 4: Season Features (May 28-30)
- **Same-season features** (month-of-year history): Public MAE **0.8120** at 20%
- **Ordinal + season tree**: Public MAE **0.8120**

### Phase 5: Grid Search (May 31 - June 1)
- **XGBoost grid search** (108 combos): Best CV 0.2888 (depth=7, trees=300, lr=0.05)
- **LightGBM grid search** (81 combos): Best CV 0.3153 (num_leaves=70, depth=6, lr=0.05)
- **LightGBM 15% blend**: Public MAE **0.8112** (current best)

### Phase 6: Aggressive Attempts (June 2) — All Failed
- Ordinal tree standalone: 0.827-0.831
- Grid XGB + LightGBM: 0.848
- Season anchor: 0.850-0.851
- Historical severity: 0.878

---

## 4. Key Technical Details

### Feature Engineering

| Feature Set | Dimensions | Description |
|-------------|------------|-------------|
| Original baseline | 126 | 14 met features × 9 stats |
| Extended baseline | 168 | 14 met features × 12 stats |
| Hybrid temporal | 294 | Original + rolling windows + seasonal patterns |
| Same-season | 4 | Month-of-year severity history |
| Ordinal classification | 6 | Class probabilities (0-5) |

### Best Model Parameters

**XGBoost (grid search winner):**
```python
{
    'n_estimators': 300,
    'max_depth': 7,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_jobs': 2
}
```

**LightGBM (grid search winner):**
```python
{
    'num_leaves': 70,
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'n_jobs': 2,
    'verbose': -1
}
```

### Best Anchor Composition
```
tree4+ordinal1.5% anchor + 20% season tree = season_w20_w20_20260530_122421.csv (0.8120)
```

### Best Submission Composition
```
85% season-tree anchor (0.8120) + 15% LightGBM (0.8470) = lgbm_blend_w15_w15_20260531_133212.csv (0.8112)
```

---

## 5. Grid Search Results

### XGBoost Grid Search (108 combinations)

**Stage 1 (27 combos):**
- Best: depth=7, trees=300, lr=0.05, CV=0.2969

**Stage 2 (27 combos):**
- Best: subsample=0.8, colsample=0.9, CV=0.2932

**Stage 3 (27 combos):**
- Best: reg_alpha=0.1, reg_lambda=1.0, CV=0.2929

**Stage 4 (27 combos):**
- Best: depth=7, trees=300, lr=0.05, subsample=0.8, colsample=0.9, alpha=0.1, lambda=1.0, CV=0.2888

**Top 5 configurations:**

| Rank | Depth | Trees | LR | Subsample | Colsample | Alpha | Lambda | CV MAE |
|------|-------|-------|-----|-----------|-----------|-------|--------|--------|
| 1 | 7 | 300 | 0.05 | 0.8 | 0.9 | 0.1 | 1.0 | 0.2888 |
| 2 | 7 | 300 | 0.05 | 0.9 | 0.8 | 0.1 | 1.0 | 0.2890 |
| 3 | 7 | 300 | 0.05 | 0.8 | 0.8 | 0.1 | 1.0 | 0.2891 |
| 4 | 7 | 300 | 0.05 | 0.9 | 0.9 | 0.1 | 1.0 | 0.2893 |
| 5 | 7 | 200 | 0.05 | 0.8 | 0.9 | 0.1 | 1.0 | 0.2895 |

### LightGBM Grid Search (81 combinations)

**Best:** num_leaves=70, depth=6, lr=0.05, trees=300, CV=0.3153

**Top 5 configurations:**

| Rank | num_leaves | depth | lr | trees | CV MAE |
|------|------------|-------|-----|-------|--------|
| 1 | 70 | 6 | 0.05 | 300 | 0.3153 |
| 2 | 70 | 7 | 0.05 | 300 | 0.3155 |
| 3 | 50 | 6 | 0.05 | 300 | 0.3160 |
| 4 | 70 | 6 | 0.05 | 200 | 0.3162 |
| 5 | 50 | 7 | 0.05 | 300 | 0.3164 |

---

## 6. Blend Weight Optimization

### LightGBM Blend Weights (into season-tree anchor)

| Weight | Public MAE | Delta |
|--------|------------|-------|
| 0% | 0.8120 | baseline |
| 2% | 0.8118 | -0.0002 |
| 5% | 0.8116 | -0.0004 |
| 10% | 0.8113 | -0.0007 |
| **15%** | **0.8112** | **-0.0008** |
| 20% | 0.8114 | -0.0006 |
| 25% | 0.8117 | -0.0003 |
| 30% | 0.8123 | +0.0003 |

**Conclusion:** Optimal weight is 12-15%. Improvement curve flattens and reverses after 15%.

### Grid XGB + LightGBM + Season-tree (10%)

| Weight | Public MAE |
|--------|------------|
| 2% | 0.8118 |
| 5% | 0.8116 |
| **10%** | **0.8113** |
| 15% | 0.8114 |
| 20% | 0.8118 |

---

## 7. File Inventory

### Core Scripts

| File | Purpose |
|------|---------|
| `scripts/grid_search_xgboost.py` | Staged XGBoost grid search |
| `scripts/generate_lgbm_submission.py` | LightGBM grid search |
| `scripts/blend_submissions.py` | Blend two CSVs by weight |
| `scripts/generate_submission.py` | Generate submission from model |
| `scripts/compare_candidate_distribution.py` | Compare prediction distributions |

### Model Implementations

| File | Purpose |
|------|---------|
| `model/temporal_features.py` | Hybrid temporal features (294 dims) |
| `model/temporal_tree.py` | XGBoost per-week models |
| `model/same_season.py` | Month-of-year severity features |
| `model/ordinal_tree.py` | Ordinal classification helpers |
| `model/severity_history.py` | Score history features |
| `model/experiments.py` | clip, blend, build, validate |

### Test Files

| File | Status |
|------|--------|
| `tests/test_grid_search.py` | 4 tests, passing |
| `tests/test_lgbm.py` | 2 tests, passing |
| `tests/test_blend_submissions.py` | 4 tests, passing |

### Documentation

| File | Purpose |
|------|---------|
| `AGENTS.md` | Project state, submission history |
| `docs/WORK_PROGRESS.md` | This file |
| `docs/PRELIMINARY_REPORT.md` | Full preliminary report |
| `docs/superpowers/plans/` | Implementation plans |
| `docs/superpowers/specs/` | Design specs |

---

## 8. Cached Data Files

| File | Size | Description |
|------|------|-------------|
| `data/train_X_temporal_hybrid_52.npy` | 268MB | Training features (119144×294) |
| `data/train_y_temporal_hybrid_52.npy` | 4.6MB | Training labels (119144×5) |
| `data/train_regions_temporal_hybrid_52.npy` | 953KB | Training region IDs |
| `data/test_X_temporal_hybrid_0.npy` | 2.2MB | Test features (2248×294) |
| `data/test_regions_temporal_hybrid_0.npy` | 18KB | Test region IDs |

---

## 9. Key Findings & Insights

### What Works
1. **Ordinal classification** — treats severity as ordered classes, aligns with problem structure
2. **Same-season features** — month-of-year historical severity patterns
3. **Conservative blending** — 10-15% new models into proven anchor
4. **Staged grid search** — systematic parameter space coverage
5. **GroupKFold by region** — prevents data leakage

### What Doesn't Work
1. **Standalone grid search models** — overfit to CV, poor public MAE
2. **Full model replacement** — causes distribution shift
3. **Neural networks** (PatchTST, V2 CNN) — massive overfitting
4. **Meta-stacking** — produces trivial predictions
5. **Aggressive blending** (>20%) — degrades performance
6. **CatBoost** — underperformed XGBoost/LightGBM

### Critical Insights
- **CV MAE is not predictive of public MAE** — standalone grid search models overfit
- **Conservative blending is essential** — prevents overfitting, maintains distribution
- **Ordinal classification aligns with problem structure** — severity scores are ordered classes
- **Same-season features are the strongest new signal** — month-of-year patterns matter
- **GroupKFold by region prevents leakage** — windows from same region stay together
- **Season anchor (20%) is better than ordinal anchor** — ordinal causes distribution shift
- **Grid search helps, but only when blended** — standalone overfits, blend works

---

## 10. Current State & Next Steps

### Current Best Submission
- **File:** `output/daily_candidates/lgbm_blend_w15_w15_20260531_133212.csv`
- **Public MAE:** 0.8112
- **Composition:** 85% season-tree anchor + 15% LightGBM
- **Anchor:** tree4+ordinal1.5% + 20% season tree

### New Candidates Ready for Tomorrow
- `alt_lgbm_w10_w10_*.csv` — Alt LightGBM params (num_leaves=50, depth=5, lr=0.04) at 10%
- `residual_w05_w05_*.csv` — Residual correction at 5%
- `residual_w10_w10_*.csv` — Residual correction at 10%
- `residual_w15_w15_*.csv` — Residual correction at 15%

### Tomorrow Commands
```bash
kaggle competitions submit -c data-mining-2026-final-project -f output/daily_candidates/alt_lgbm_w10_w10_20260602_154539.csv -m "alt lgbm w10"
kaggle competitions submit -c data-mining-2026-final-project -f output/daily_candidates/residual_w05_w05_20260602_154543.csv -m "residual w05"
kaggle competitions submit -c data-mining-2026-final-project -f output/daily_candidates/residual_w10_w10_20260602_154540.csv -m "residual w10"
kaggle competitions submit -c data-mining-2026-final-project -f output/daily_candidates/residual_w15_w15_20260602_154545.csv -m "residual w15"
```

### Replanning Needed
All aggressive approaches failed. Current methods are plateaued. Need fundamentally different:
- **Features** — volatility, trends, non-linear interactions
- **Models** — stacking with OOF predictions, different architectures
- **Approaches** — region-specific models, temporal attention mechanisms

---

## 11. Lessons Learned

1. **Start with conservative baselines** — ordinal classification was the breakthrough
2. **Grid search is valuable, but only when blended** — standalone overfits
3. **GroupKFold by region is essential** — prevents data leakage
4. **Conservative blending prevents overfitting** — 10-15% is optimal
5. **Same-season features are powerful** — month-of-year patterns matter
6. **Distribution checks are critical** — don't trust CV alone
7. **Iterative improvement works** — small refinements compound
8. **Don't chase marginal gains** — plateau means need new approach

---

## 12. Environment & Configuration

**Hardware:**
- CPU: 12 cores (using 2 threads)
- RAM: 32GB
- GPU: NVIDIA (currently unavailable)

**Software:**
- Python 3.10
- XGBoost, LightGBM, scikit-learn, pandas, numpy
- Jupyter notebooks

**Environment Variables:**
```bash
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
MKL_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
```

---

## 13. Submission Guidelines

- **Budget:** 10 submissions/day
- **Format:** CSV with columns matching `data/sample_submission.csv`
- **Naming:** `method_variant_YYYYMMDD_HHMMSS.csv`
- **Validation:** Always run `experiments.validate_submission()` before submitting
- **Distribution check:** Compare against current best using `compare_candidate_distribution.py`

---

## 14. Contact & Resources

- **Instructor:** Jyun-Yu Jiang
- **Email:** wu80623@gmail.com (put "DM final" in subject)
- **Competition:** https://www.kaggle.com/competitions/data-mining-2026-final-project
- **Course:** NYCU Data Mining Spring 2026

---

*Last updated: June 2, 2026*
