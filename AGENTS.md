# DM2026-Final-Project — Natural Disaster Severity

## Quick Start

```bash
jupyter nbconvert --to notebook --execute --inplace Project_Analysis.ipynb
# or open Jupyter and Run All Cells
```

## User-run validation (copy-paste safe)

Do not hand users indented `python <<'PY'` blocks — they break on paste. Use:

```bash
PYTHONPATH=. python scripts/validate_prob_cache.py output/prob_cache/<cache>.npz
PYTHONPATH=. python scripts/gate_and_validate_submissions.py --reference output/daily_candidates/prob_blend_recycle8089_ord08.csv --candidate <candidate.csv>
```

First run is slow (builds window features). Subsequent runs use `.npy` cache.

## Architecture

```
model/
├── train.py       # train_xgboost, cv_evaluate (GroupKFold by region)
├── utils.py       # _aggregate_array (168 features), load_train_data, load_test_data, generate_submission
└── experiments.py # submission utilities: clipping, blending, building, validation
```

**`model/train.py`**: XGBoost multi-output regression via `MultiOutputRegressor(XGBRegressor)`. Safe default params: 200 trees, max_depth=5, lr=0.05, reg_alpha=0.1, reg_lambda=1.0, n_jobs=2. `cv_evaluate()` uses GroupKFold-5 by region (prevents data leakage — windows from same region stay together) and accepts `params_override` for experiments.

**`model/utils.py`**: 
- `_aggregate_array()`: 168 features = 14 met features × 12 stats (mean, std, min, max, q25, q50, q75, last7_mean, last30_mean, trend, skew, kurt)
- `load_train_data(path, max_windows_per_region=None)`: Sliding 91-day windows → 5-week labels. `None` = ALL windows. Caches to `.npy`.
- `load_test_data(path)`: Takes last 91 days per region, aggregates.
- `generate_submission()`: Writes timestamped CSVs (`name_YYYYMMDD_HHMMSS.csv`) and appends to `output/SUBMISSIONS.md`.

**`model/experiments.py`**: submission utilities shared across pipelines.
- `clip_predictions()` clips predictions to `[0, 5]` before submission.
- `blend_predictions()` blends two prediction arrays by weight.
- `build_submission()` and `validate_submission()` enforce exact sample-submission columns and row order.

**Data**: 12.3M rows, 2248 regions, 5480 days each, 14 meteorological features. Weekly severity scores 0-5. Test = 91 days per region (predict next 5 weeks).

## Kaggle

- Competition: `data-mining-2026-final-project` (invite-only)
- Metric: **MAE** (Mean Absolute Error, range 0-5) — **lower is better**
- Treat the practical budget as **3 submissions/day**.
- Output: `output/submission_*.csv`

## Experiment Logging Rule

Always update this `AGENTS.md` file when experiments or Kaggle results change. Record candidate name, file path, features/windows, parameters, offline CV if available, Kaggle public MAE, and any rejection/error reason.

CSV naming rule: use date-time timestamps only, not `_vN` suffixes. Example: `xgb_a_depth4_300_20260518_143012.csv`.

## Submission History

| File | MAE | Features | Windows/region | XGBoost params |
|------|-----|----------|----------------|----------------|
| submission_xgb.csv | 0.8509 | 126 (9 stats) | 52 | 200 trees, lr=0.1, md=6 |
| submission_xgb_v1.csv | 0.8437 | 168 (12 stats) | 52 | 200 trees, lr=0.1, md=6 |
| submission_xgb_v2_fixed.csv | **0.8434** | 168 (12 stats) | 52 | 200 trees, lr=0.05, md=5, n_jobs=2 |
| cnn_1d_20260518_180837.csv | **0.8222** | raw 91-day sequence | 52 | Small 1D CNN, 25 epochs, val MAE 0.3614 |
| cnn_1d_small_20260519_044615.csv | 0.8282 | raw 91-day sequence | 52 | Small 1D CNN, 40 epochs, scheduler, val MAE 0.34218 |
| cnn_1d_v2_20260519_045002.csv | 0.8901 | raw 91-day sequence | 52 | V2 CNN, 100 epochs, scheduler, val MAE 0.24172; bad public generalization |
| cnn_1d_v2_20260519_044515.csv | 0.8967 | raw 91-day sequence | 52 | V2 CNN, 40 epochs, scheduler, val MAE 0.24182; bad public generalization |
| temporal_tree_hybrid_20260518_181718.csv | 0.8403 | 168 baseline + targeted temporal deltas | 52 | GPU XGBoost, 300 trees, lr=0.04, md=5, CV MAE 0.36316 |
| temporal_tree_blocks_20260518_172813.csv | Not submitted | ~938 block features | 52 | CV MAE 0.46336; worse than baseline |
| cnn_1d_small_blend_20260520_161232.csv | 0.8258 | raw 91-day sequence blend | 52 | 70/30 blend: May 18 best + May 20 seed 123 small CNN; corr 0.99307 vs best, diff 0.06413 |
| cnn_1d_small_20260520_161056.csv | 0.8512 | raw 91-day sequence | 52 | Small CNN, seed 123, 30 epochs, dropout 0.20, wd 0.0005, scheduler, val MAE 0.34531 |
| cnn_1d_small_20260520_160954.csv | 0.8812 | raw 91-day sequence | 52 | Small CNN, seed 7, 25 epochs, dropout 0.15, wd 0.001, scheduler, val MAE 0.36375 |

**Key insight**: Small CNN is still the best approach (0.8222). V2 CNN had much better local validation (0.2417-0.2418) but much worse public MAE (0.8901-0.8967), so the local split is not reliable for larger CNN architecture selection. May 20 small-CNN seed/training variants also underperformed publicly despite acceptable distribution checks; the 70/30 blend was closest at 0.8258 but did not beat the May 18 best. Hybrid temporal tree features beat the previous XGBoost baseline (0.8403 vs 0.8434), but wide temporal block features overfit/underfit badly (CV 0.46336) and should not be submitted.

## Current State

- **Best submitted file**: `output/daily_candidates/prob_blend_recycle8089_ord08.csv`
- **Best public MAE**: **0.8088**
- **Method**: Recycled anchor — 92% soft wrap of `prob_blend_recycle8092_ord10.csv` (0.8089) + 8% ordinal
- **Previous best**: `prob_blend_recycle8092_ord10.csv` @ 0.8089 (Jun 4)
- **Windows/region**: 52
- **Submission format note**: rows must match `data/sample_submission.csv` order.

### Prob blend — June 3 sweep (lgbm soft anchor)

| File | Blend | Public MAE | vs 0.8112 |
|------|-------|------------|-----------|
| `prob_blend_best83_ord17.csv` | 83% soft + 17% ordinal | **0.8092** | -0.0020 |
| `prob_blend_best82_ord18.csv` | 82% / 18% ordinal | 0.8092 | -0.0020 (tie) |
| `prob_blend_best92_ord08.csv` | 92% / 8% ordinal | 0.8101 | -0.0011 |

**Plateau (Jun 3):** 17–18% ordinal on `soft_lgbm_best` anchor; higher ordinal % helped monotonically until 83/17.

### Prob blend — June 4 (recycled-anchor family)

Soft anchor = mean-preserving wrap of prior best CSV + `ordinal_hybrid_best.npz`.

| File | Blend | Public MAE | Decision |
|------|-------|------------|----------|
| `prob_blend_best825_ord175.csv` | 82.5% lgbm-soft + 17.5% ord | 0.8092 | Tie 83/17; stop micro-tune on first anchor. |
| `prob_blend_recycle8092_ord10.csv` | 90% wrap(0.8092) + 10% ord | 0.8089 | Breakthrough: recycle anchor. |
| **`prob_blend_recycle8089_ord08.csv`** | **92% wrap(0.8089) + 8% ord** | **0.8088** | **Current best** (tied with 7%). |
| `prob_blend_recycle8089_ord07.csv` | 93% wrap(0.8089) + 7% ord | 0.8088 | Tie 8%; plateau 7–8%. |
| `prob_blend_recycle8089_ord09.csv` | 91% wrap(0.8089) + 9% ord | 0.8089 | Worse than 8%. |
| `prob_blend_recycle8089_ord12.csv` | 88% + 12% ord | — | **Not submitted**; gate fail (shift 0.021 > 0.02). |
| `prob_blend_recycle8089_ord06.csv` | 94% wrap(0.8089) + 6% ord | — | Superseded by recycle-from-8088 path. |
| `prob_blend_recycle8088_ord06.csv` | 94% wrap(0.8088) + 6% ord | 0.8090 | Worse than 7–8%; stop lowering below 7%. |
| `prob_blend_recycle8088_season_ord08.csv` | 92% soft(8088) + 8% season-ordinal | 0.8103 | **Worse**; season ordinal in prob space does not help. Stop branch. |
| `prob_blend_8089_mae_ord03.csv` | 97% soft(8089) + 3% MAE-tree OOF-cal | 0.8098 | Gated safe; **worse** than 0.8088. Stop MAE-tree prob branch. |
| `prob_blend_8089_oofcal_ord08_full.csv` | 92% soft(8089) + 8% OOF-cal hybrid ord | 0.8091 | Gated safe diff=0.0049; OOF MAE 0.378→0.349; **worse** than 0.8088. Stop OOF-cal ordinal at 8%. |
| `scalar_cal_holdout_full.csv` | Holdout week-affine on 8088 anchor (strength 0.4) | 0.8112 | Gated safe diff=0.016; holdout proxy MAE 0.319→0.306; mean shift down; **worse** than 0.8088. Stop scalar temporal calibration. |
| `prob_blend_recycle8088_blackout_ord07.csv` | 93% soft(8088) + 7% blackout-ordinal | 0.8135 | **Much worse**; gated safe offline. Stop blackout-ordinal prob branch. |
| `prob_blend_recycle8088_blackout_ord08.csv` | 92% soft(8088) + 8% blackout-ordinal | — | **Not submitted**; 7% failed LB. |
| `prob_blend_recycle8089_ord11.csv` | 89% + 11% ord | — | Gated safe offline; not submitted. |
| `prob_best_residual_w0025.csv` | 0.25% residual scalar | — | Not submitted; fails vs lgbm (diff 0.050). |

### Prob blend — June 5 (hybrid_full ordinal + recycle2)

| File | Blend | Public MAE | Offline gate | Decision |
|------|-------|------------|--------------|----------|
| `prob_blend_recycle8089_ord08.csv` | 92% wrap(0.8089) + 8% hybrid ord | **0.8088** | identity | Re-submit; still best. |
| `prob_blend_recycle2_8088_w0.97.csv` | 97% soft(0.8088) + 3% hybrid ord | 0.8089 | safe, diff ~0.006 | Worse; stop second recycle on hybrid ord. |
| `prob_blend_8089_fullord03.csv` | 97% soft8089 + 3% hybrid_full ord | **0.8109** | safe, diff 0.020 | **Worse**; stop hybrid_full prob branch. |
| `prob_blend_recycle8089_fullord03_ord02.csv` | 98% wrap(fullord03) + 2% full ord | **0.8123** | safe, diff 0.025 | **Much worse**; stop fullord recycle. |
| `prob_blend_recycle8089_fullord08.csv` | 92% + 8% full ord | — | **gate fail** (week shift 0.023) | Not submitted. |

**Plateau (Jun 4 recycle):** On recycled anchors, **7–8% hybrid ordinal** → **0.8088** best; 6% from 8088 → 0.8090 (worse); 9–10% → 0.8089. **Do not go below 7% hybrid ordinal** on current cache.

**Stopped (Jun 5):** `hybrid_full` ordinal (weather+season+blackout concat) in prob space — gate-safe at 3% but LB +0.0021; second recycle +0.0035. Same failure mode as season/blackout ordinal substitutes. **Gap to target 0.8056:** 0.0032; needs new model family (region-state/hurdle/joint retrain), not more cache blends.

**Active caches:** `soft_lgbm_best.npz`, `soft_prob_best8092.npz`, `soft_prob_best8089.npz`, `soft_prob_best8088.npz`, `ordinal_hybrid_best.npz`, `ordinal_full_hybrid_best.npz` (fullord branch stopped), `ordinal_season_hybrid_best.npz`, `ordinal_blackout_hybrid_best.npz` (stopped).

**Reproduce current best:**
```bash
PYTHONPATH=. python scripts/cache_submission_soft_probs.py \
  --submission output/daily_candidates/prob_blend_recycle8092_ord10.csv \
  --output-path output/prob_cache/soft_prob_best8089.npz
PYTHONPATH=. python scripts/blend_prob_submissions.py \
  --cache output/prob_cache/soft_prob_best8089.npz:0.92 \
  --cache output/prob_cache/ordinal_hybrid_best.npz:0.08 \
  --output-path output/daily_candidates/prob_blend_recycle8089_ord08.csv
```

### Day 1 queue (2026-06-02, pending public MAE)

| Slot | File | Notes |
|------|------|-------|
| A exploit | `lgbm_blend_w14_w14_20260602_164850.csv` | 14% LGBM into season anchor; strict-safe vs best |
| B explore | `best_lgbm_grid_xgb_w01_w01_20260602_152433.csv` | 1% grid-XGB into current best |
| C explore | `residual_w05_w05_20260602_154543.csv` | 5% residual into current best |

Deferred: `lgbm_blend_w16_*` (Day 2), `alt_lgbm_w10` (diff too high).

### Cached probability blend (2026-06-02)

Scripts: `cache_ordinal_probabilities.py`, `cache_submission_soft_probs.py`, `blend_prob_submissions.py`, `model/probability_blend.py`.

```bash
# 1) Cache real ordinal probs (slow — trains 25 XGB classifiers)
PYTHONPATH=. python scripts/cache_ordinal_probabilities.py

# 2) Cache soft probs from existing best submission (fast)
PYTHONPATH=. python scripts/cache_submission_soft_probs.py \
  --submission output/daily_candidates/lgbm_blend_w15_w15_20260531_133212.csv \
  --output-path output/prob_cache/soft_best.npz

# 3) Blend in probability space, then export CSV
PYTHONPATH=. python scripts/blend_prob_submissions.py \
  --cache output/prob_cache/soft_best.npz:0.85 \
  --cache output/prob_cache/ordinal_hybrid_TIMESTAMP.npz:0.15 \
  --output-path output/daily_candidates/prob_blend_w15_ord15.csv
```

### Grid Search + LightGBM Results (May 31 - June 1)

XGBoost grid search (108 combos) found best params: depth=7, trees=300, lr=0.05, subsample=0.8, colsample=0.9, reg_alpha=0.1, reg_lambda=1.0. CV MAE=0.2888.

LightGBM grid search (81 combos) found best params: num_leaves=70, depth=6, lr=0.05, trees=300. CV MAE=0.3153.

**Standalone scores were bad** (XGBoost 0.849, LightGBM 0.847) — overfitting to CV. But **blending at 10-15% into anchor gave new best**.

| Blend Weight | Public MAE | Delta |
|---:|---:|---:|
| 2% | 0.8118 | -0.0002 |
| 5% | 0.8116 | -0.0004 |
| 10% | 0.8113 | -0.0007 |
| **15%** | **0.8112** | **-0.0001** |
| 20% | 0.8114 | +0.0002 |
| 25% | 0.8117 | +0.0003 |
| 30% | 0.8123 | +0.0006 |

**Optimal LightGBM weight: 12-15%** (all tied at 0.8112). Improvement curve flattens and reverses after 15%.

### Previous ranking

Current ranking: lgbm blend 15% (2026-06-01) **0.8112**, lgbm/gs blend 10% 0.8113, lgbm blend 20% 0.8114, lgbm/gs blend 5% 0.8116, lgbm blend 25% 0.8117, lgbm/gs blend 2% 0.8118, season tree 20% (2026-05-30) 0.8120, season tree 15% 0.8122, season tree 25% 0.8123, season tree 12% 0.8124, season tree 10% 0.8127, season tree 9% 0.8128, season tree 8% 0.8130, season tree 7% 0.8131, season tree 6% 0.8133, season tree 5% 0.8134, season tree 4% 0.8136, season tree 3%+residual 1% 0.8137, season tree 3% 0.8138, season tree 2% 0.8140, residual 1% 0.8143, residual 0.5% 0.8144, blk_temp_tree 0.25% 0.8146, tree4+ord1.5% 0.8145, tree3+ord2% 0.8146, per-week ordinal inc 0.8147, tree3+ord1.5% 0.8149, blackout ordinal 0.25% 0.8150.

### Key Finding

Same-season (month-of-year) severity history is the strongest new signal discovered since ordinal classification. It improves linearly from 2% to 10% blend weight and may continue higher. The gap to baseline 3 is 0.0071. Further improvements should combine season-tree with other proven methods (ordinal, residual, blackout temporal tree) at higher blend weights.

Ordinal classification is the breakthrough method. Tree refinements and absolute error tree also helped. Severity prior was too different from anchor to help at 2%.

Pure seed42 GRU-weight tuning is plateaued. Tiny tree correction and seed30 GRU are the only May 23 positive signals; future submissions should refine those and explore historical severity / ordinal-MAE methods rather than more w25.x tweaks.

Previous small-CNN follow-up commands were tried on May 20 and did not improve the public score:

```bash
python scripts/generate_cnn_submission.py --model small --max-windows-per-region 52 --epochs 25 --seed 7 --dropout 0.15 --scheduler
python scripts/generate_cnn_submission.py --model small --max-windows-per-region 52 --epochs 25 --seed 123 --dropout 0.15 --scheduler
```

Submit only if predictions are close to the current best small CNN distribution and the CSV passes sample-submission checks. Do not use V2 validation MAE alone as a submission signal.

### Next Candidate Commands

Calendar-Small-CNN candidate:

```bash
python scripts/generate_cnn_submission.py --model small --calendar --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.001
```

CNN-GRU candidate:

```bash
python scripts/generate_cnn_submission.py --model cnn_gru --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.001
```

Distribution gate after generation uses the timestamped candidate path printed by `generate_cnn_submission.py`. For example, after a Calendar-Small-CNN run:

```bash
python scripts/compare_candidate_distribution.py output/daily_candidates/cnn_1d_small_YYYYMMDD_HHMMSS.csv output/daily_candidates/cnn_1d_20260518_180837.csv
```

After a CNN-GRU run:

```bash
python scripts/compare_candidate_distribution.py output/daily_candidates/cnn_1d_cnn_gru_YYYYMMDD_HHMMSS.csv output/daily_candidates/cnn_1d_20260518_180837.csv
```

Submit at most one Calendar-Small-CNN and at most one CNN-GRU candidate. Save the remaining daily submission for a blend or combined follow-up after public feedback. Do not submit if temporal backtest and distribution checks disagree.

### GRU Best-Blend Candidates (May 22)

Blended CNN-GRU (`cnn_1d_cnn_gru_20260522_154103.csv`, backtest 0.184923) into current best at three weights:

| Blend | Correlation | Mean Diff | Week Shift | Safe |
|---|---:|---:|---:|---|
| w10 (`cnn_gru_blend_w10_20260522_160258.csv`) | 0.997 | 0.050 | 0.019 | Yes |
| w15 (`cnn_gru_blend_w15_20260522_160258.csv`) | 0.992 | 0.074 | 0.029 | Yes |
| w20 (`cnn_gru_blend_w20_20260522_160258.csv`) | 0.985 | 0.099 | 0.039 | Yes |

Decision: submit `w20` as highest safe weight at the time. w20 public MAE: **0.8171** (improved from 0.8222).

### Higher GRU Blend Search (May 22)

Tested weights beyond w20 against relaxed old-best thresholds (corr >= 0.975, diff <= 0.13, shift <= 0.055) and incremental w20 thresholds (corr >= 0.995, diff <= 0.035, shift <= 0.02):

| Blend | corr vs old | diff vs old | shift vs old | corr vs w20 | diff vs w20 | Safe |
|---:|---:|---:|---:|---:|---:|---|
| w22 (`cnn_gru_blend_w22_20260522_163740.csv`) | 0.982 | 0.109 | 0.043 | 0.9998 | 0.010 | Yes |
| w24 (`cnn_gru_blend_w24_20260522_163741.csv`) | 0.979 | 0.119 | 0.047 | 0.9993 | 0.020 | Yes |
| w25 (`cnn_gru_blend_w25_20260522_163741.csv`) | 0.977 | 0.124 | 0.049 | 0.9990 | 0.025 | Yes, public 0.8167 |
| w26 (`cnn_gru_blend_w26_20260522_170137.csv`) | 0.975 | 0.129 | 0.051 | corr vs w25=0.99996 | diff vs w25=0.005 | Borderline, public 0.8167 |
| w27 (`cnn_gru_blend_w27_20260522_163914.csv`) | 0.973 | 0.134 | 0.053 | 0.998 | 0.035 | No |
| w30 (`cnn_gru_blend_w30_20260522_163914.csv`) | 0.966 | 0.149 | 0.058 | 0.996 | 0.050 | No |

Decision: **w25 and w26 tied at public 0.8167**. Keep w25/w26 as current best. w27 fails corr-vs-old and diff-vs-old; w30 fails all gates.

### Fine-Grained GRU Blends (May 22)

Fractional blends around the plateau:

| Blend | corr vs old | diff vs old | shift vs old | corr vs w25 | Safe |
|---:|---:|---:|---:|---:|---|
| w25.5 (`cnn_gru_blend_w25p5_w26_20260522_171753.csv`) | 0.976 | 0.126 | 0.050 | 0.99999 | Yes |
| w26.5 (`cnn_gru_blend_w26p5_w26_20260522_171753.csv`) | 0.974 | 0.131 | 0.052 | 0.99991 | No (corr-old) |

Recommendation: submit **w25.5** (`cnn_gru_blend_w25p5_w26_20260522_171753.csv`). w26.5 fails corr-vs-old at 0.974.

### Tiny 40-Epoch CNN Blends (May 22)

Blended 40-epoch small CNN (`0.8282` public) into current best `w25` (`0.8167`):

| Blend | corr vs w25 | diff vs w25 | shift vs w25 | Safe |
|---|---:|---:|---:|---|
| 95% w25 + 5% 40epoch (`w25_40epoch_blend_w05_20260522_171947.csv`) | 0.9999 | 0.008 | 0.002 | Yes |
| 90% w25 + 10% 40epoch (`w25_40epoch_blend_w10_20260522_171948.csv`) | 0.9996 | 0.015 | 0.005 | Yes |

Recommendation: submit **10% blend** (`w25_40epoch_blend_w10_20260522_171948.csv`). Both weights are safe; prefer 10% per decision rule.

### Seed-21 GRU Blend (May 22)

Trained CNN-GRU seed-21 (`cnn_1d_cnn_gru_20260522_175824.csv`, val_mae 0.252). Standalone distribution vs w25: corr=0.705, diff=0.424, safe=False.

Blended into w25:

| Blend | corr vs w25 | diff vs w25 | shift vs w25 | Safe |
|---|---:|---:|---:|---|
| 10% (`w25_seed21_gru_blend_w10_20260522_180929.csv`) | 0.997 | 0.042 | 0.008 | No |
| 15% (`w25_seed21_gru_blend_w15_20260522_180929.csv`) | 0.993 | 0.064 | 0.011 | No |

Recommendation: do not submit any seed-21 GRU blend. Both weights exceed the 0.035 diff threshold.

### Latest Public Feedback (May 22)

| Candidate | Public MAE | Decision |
|---|---:|---|
| `cnn_gru_blend_w25p5_w26_20260522_171753.csv` | 0.8167 | Tied best; confirms plateau around 25-26% seed42 GRU. |
| `w25_40epoch_blend_w10_20260522_171948.csv` | 0.8171 | Worse than best; stop 40-epoch CNN blend branch. |
| `w25_seed21_gru_blend_w05_20260522_190847.csv` | 0.8181 | Worse than best; stop seed21 GRU branch. |

### May 23 Public Feedback

All six prepared candidates were submitted. The 2% tree correction became the new best.

#### Seed30 CNN-GRU

Trained `cnn_gru` seed 30: `cnn_1d_cnn_gru_20260523_034152.csv`, val_mae 0.257374.

Standalone vs w25: corr=0.794, diff=0.388, safe=False (but best standalone GRU corr yet — seed42 was 0.643, seed21 was 0.705).

Blends into w25:

| Blend | corr vs w25 | diff vs w25 | shift vs w25 | Safe | Public MAE |
|---|---:|---:|---:|---|---:|
| 5% (`w25_seed30_gru_blend_w05_20260523_035626.csv`) | 0.9993 | 0.019 | 0.002 | Yes | Not submitted |
| **7.5%** (`w25_seed30_gru_blend_w07p5_w08_20260523_035629.csv`) | 0.9985 | 0.029 | 0.004 | **Yes** | 0.8165 |
| 10% (`w25_seed30_gru_blend_w10_20260523_035631.csv`) | 0.9973 | 0.039 | 0.005 | No | Not submitted |

Decision: seed30 is a useful positive signal, but weaker than the 2% tree correction. Consider combining a smaller seed30 component with the new tree anchor.

#### Multi-GRU Average

Averaged seed42 GRU + seed30 GRU 50/50, then blended into old CNN at 25%:
`oldcnn_gruavg_blend_w25_20260523_041011.csv`.

| Comparison | corr | diff | shift | Safe |
|---|---:|---:|---:|---|
| vs w25 | 0.9940 | 0.055 | 0.015 | No (corr < 0.995) |
| vs old CNN | 0.9847 | 0.098 | 0.034 | Yes |

Public MAE: 0.8178. Decision: stop multi-GRU average branch.

#### Week-Specific GRU Weights

Per-column seed42 GRU weights into old CNN anchor:

| Candidate | corr vs w25 | diff vs w25 | shift vs w25 | Safe | Public MAE |
|---|---:|---:|---:|---|---:|
| increasing [0.24,0.24,0.25,0.26,0.26] (`seed42_gru_weekly_increasing_w24_24_25_26_26_20260523_041043.csv`) | 1.000 | 0.004 | 0.002 | Yes | 0.8170 |
| late-lower [0.25,0.25,0.25,0.24,0.24] (`seed42_gru_weekly_late_lower_w25_25_25_24_24_20260523_041045.csv`) | 1.000 | 0.002 | 0.000 | Yes | 0.8167 |

Decision: stop weekly GRU branch. Changes were too small or in the wrong direction.

New script created: `scripts/blend_weekly_submission.py` with tests `tests/test_blend_weekly_submission.py` (2 tests, PYTHONPATH=. required).

#### Tiny Tree Blend

Blended tree hybrid into w25:

| Blend | corr vs w25 | diff vs w25 | shift vs w25 | Safe | Public MAE |
|---|---:|---:|---:|---|---:|
| **2%** (`w25_tree_hybrid_blend_w02_20260523_041047.csv`) | 1.000 | 0.009 | 0.008 | Yes | **0.8159** |
| 5% (`w25_tree_hybrid_blend_w05_20260523_041048.csv`) | 1.000 | 0.023 | 0.020 | No (shift >= 0.02) | Not submitted |

Decision: set 2% tree blend as new anchor. Refine nearby tree weights before spending more training time.

#### Stability Ensemble

One-third each of w25, w26, w25.5: `w25_w26_w25p5_stability_w33_20260523_041055.csv`.

| corr vs w25 | diff vs w25 | shift vs w25 | Safe | Public MAE |
|---|---:|---:|---:|---|---:|
| 1.000 | 0.002 | 0.001 | Yes | 0.8167 |

Decision: tied old best; fallback/private robustness only.

#### Why The Plateau Is Hard To Break

Observed after May 23:

- The new best mean (`0.7205`) moved closer to the recent cached train-label mean (`0.7284`) than w25 did (`0.7137`).
- Very safe candidates often changed predictions by only `0.002-0.004` mean absolute difference, so their possible public movement was tiny.
- The only positive May 23 signals were structurally different: tiny tree correction (`0.8159`) and seed30 GRU (`0.8165`).
- Train and test share all 2248 region IDs. Current weather-only CNN/tree inputs mostly ignore explicit historical severity, so region/season severity priors are now the highest-priority missing signal.

Next high-upside methods: refine ordinal tree weight around 1%, combine ordinal with absolute error tree, add severity history features to ordinal tree training.

### May 24 Public Feedback

Six candidates submitted. Ordinal classification breakthrough.

| File | Public MAE | Decision |
|---|---:|---|
| `anchor_ordinal_tree_w01_20260524.csv` | **0.8153** | New best; ordinal classification is the breakthrough method. |
| `w25_tree_refine_w030_20260524.csv` | 0.8156 | Tree 3.0% beat previous 2.0% best. |
| `anchor_abs_tree_w02_20260524.csv` | 0.8156 | Absolute error tree tied with tree 3.0%. |
| `w25_tree_refine_w025_20260524.csv` | 0.8158 | Tree 2.5% beat previous 2.0% best. |
| `tree_anchor_seed30_w050_20260524.csv` | 0.8158 | Seed30 combo no better than tree alone. |
| `anchor_severity_prior_w020_20260524.csv` | 0.8177 | Worse; prior too different from anchor at 2%. |

Key insight: treating scores as ordered classes `0..5` via cumulative threshold classifiers gave the best improvement. This aligns the model structure with the discrete ordinal target.

### May 25 Candidate Generation

Wave 1: three ordinal/tree refinements gated against current best `anchor_ordinal_tree_w01_20260524.csv`:

| Candidate | File | Gate vs current best | Public MAE | Decision |
|---|---|---:|---:|---|
| tree3 + ordinal1.5% | `tree3_ordinal_w015_20260525.csv` | safe=True | **0.8149** | **New best**; tree3+ordinal1.5% beats tree2+ordinal2%. |
| current_best + abs tree 1% | `current_best_abs_tree_w010_20260525.csv` | safe=True | 0.8151 | Tied with tree3+ordinal1%; abs tree complement confirmed. |
| history ordinal 0.5% | `current_best_history_ordinal_w005_20260525.csv` | safe=True | 0.8157 | Worse; history features alone as a blend aren't sufficient yet. |

Remaining generated but not submitted: `tree4_ordinal_w010_20260525.csv` (safe), `current_best_history_ordinal_w010_20260525.csv` (safe), `history_ordinal_tree_expected_20260525.csv` (standalone).

### May 26 Expanded Candidate Generation

20 candidates across 6 families generated. 19/20 pass strict gates against current best `tree4_ordinal_w015_20260525.csv` (0.8145). Affine calibration (#20) failed gate (week shift 0.0127).

| # | Candidate | File | corr | diff | Safe | Public |
|---:|---|---|---:|---:|---|---|
| 1 | tree4 + ordinal2% | `tree4_ordinal_w020_20260526.csv` | 0.999996 | 0.0021 | Yes | — |
| 2 | tree5 + ordinal1% | `tree5_ordinal_w010_20260526.csv` | 0.999993 | 0.0027 | Yes | — |
| 3 | tree5 + ordinal1.5% | `tree5_ordinal_w015_20260526.csv` | 0.999981 | 0.0046 | Yes | — |
| 4 | tree5 + ordinal2% | `tree5_ordinal_w020_20260526.csv` | 0.999961 | 0.0066 | Yes | — |
| 5a| per-week ordinal inc | `tree4_ordinal_weekly_...16454.csv` | 0.999996 | 0.0014 | Yes | 0.8147 |
| 5b| per-week ordinal late-hi | `tree4_ordinal_weekly_...16455.csv` | 0.999993 | 0.0017 | Yes | — |
| 7 | tree4 + abs1% + ord1.5% | `tree4_abs1_ordinal_w015_20260526.csv` | 0.999978 | 0.0036 | Yes | — |
| 8 | tree5 + abs1% + ord1.5% | `tree5_abs1_ordinal_w015_20260526.csv` | 0.999928 | 0.0076 | Yes | — |
| 9 | tree4 + hbord0.25 + ord1.5% | `tree4_hbord0p25_ordinal_w015_20260526.csv` | 0.999998 | 0.0010 | Yes | — |
| 10 | tree4 4-way combo | `tree4_hb_abs_ord_4way_20260526.csv` | 0.999990 | 0.0025 | Yes | 0.8145 (tied prev best) |
| 11 | seed30 + tree4 + ord1.5% | `tree4_seed30w2p5_ordinal_w015_20260526.csv` | 0.999835 | 0.0096 | Yes | — |
| 12 | seed30 4-way | `tree4_seed30_abs_ord_4way_20260526.csv` | 0.999817 | 0.0099 | Yes | — |
| 14 | blk_temp_tree 0.25% into best | `best_hb_temp_w0025_20260526.csv` | 0.999998 | 0.0011 | Yes | **0.8146** |
| 15 | blk_temp_tree 0.5% into best | `best_hb_temp_w005_20260526.csv` | 0.999992 | 0.0023 | Yes | — |
| 16 | blk_temp_tree 1.0% into best | `best_hb_temp_w010_20260526.csv` | 0.999969 | 0.0045 | Yes | — |
| 17 | blk_temp_tree 2.0% into best | `best_hb_temp_w020_20260526.csv` | 0.999875 | 0.0090 | Yes | — |
| 18 | residual 0.5% into best | `best_residual_w005_20260526.csv` | 0.999996 | 0.0018 | Yes | **0.8144** |
| 19 | residual 1.0% into best | `best_residual_w010_20260526.csv` | 0.999985 | 0.0037 | Yes | — |

New scripts: `generate_history_blackout_temporal_tree_submission.py`, `generate_residual_correction_submission.py`, `generate_affine_calibration_submission.py`.

History features now cut scores at `first_label_pos - 91` to mimic test availability.

| Candidate | File | corr vs best | diff vs best | Safe | Public |
|---|---:|---:|---|---:|
| tree3 + ordinal2% | `tree3_ordinal_w020_20260525.csv` | 0.999996 | 0.0021 | Yes | 0.8146 |
| tree4 + ordinal1.5% | `tree4_ordinal_w015_20260525.csv` | 0.999981 | 0.0046 | Yes | **0.8145** |
| best + history_blackout_ordinal 0.25% | `best_history_blackout_ordinal_w0025_20260525.csv` | 0.999998 | 0.0010 | Yes | 0.8150 |
| best + history_blackout_ordinal 0.5% | `best_history_blackout_ordinal_w005_20260525.csv` | 0.999992 | 0.0021 | Yes | 0.8151 |
| best + history_blackout_ordinal 1.0% | `best_history_blackout_ordinal_w010_20260525.csv` | 0.999968 | 0.0041 | Yes | 0.8152 |

Blackout history ordinal helped more than the old history ordinal (0.8157). Best was 0.8150 at 0.25%, but tree/ordinal refinements still dominate (0.8145, 0.8146).

### May 24 Candidate Generation

All candidates generated offline; public scores pending Kaggle submission window.

#### Tree Weight Refinements

Generated deterministic blends from old w25 + temporal tree hybrid. Compared against new anchor (`w25_tree_hybrid_blend_w02_20260523_041047.csv`):

| Weight | File | corr vs anchor | diff vs anchor | Safe |
|---|---:|---|---:|---|
| 1.0% | `w25_tree_refine_w010_20260524.csv` | 1.000 | 0.005 | Yes |
| 1.5% | `w25_tree_refine_w015_20260524.csv` | 1.000 | 0.002 | Yes |
| **2.5%** | `w25_tree_refine_w025_20260524.csv` | 1.000 | 0.002 | Yes | **0.8158** |
| 3.0% | `w25_tree_refine_w030_20260524.csv` | 1.000 | 0.005 | Yes | 0.8156 |
| 4.0% | `w25_tree_refine_w040_20260524.csv` | 1.000 | 0.009 | Yes | Not submitted |

Recommendation: tree refinements confirmed useful. 3.0% beat 2.5%.

#### Tree Anchor + Seed30 GRU

Blended seed30 standalone GRU into the new tree anchor:

| Weight | File | corr vs anchor | diff vs anchor | Safe |
|---|---:|---|---:|---|
| 2.5% | `tree_anchor_seed30_w025_20260524.csv` | 0.9998 | 0.010 | Yes |
| **5.0%** | `tree_anchor_seed30_w050_20260524.csv` | 0.9993 | 0.019 | **Yes** | 0.8158 |
| 7.5% | `tree_anchor_seed30_w075_20260524.csv` | 0.9985 | 0.029 | Yes | Not submitted |

Recommendation: seed30 combo no better than tree alone. Stop seed30 branch.

#### Severity History Prior

Created `model/severity_history.py` and `tests/test_severity_history.py` (3 tests). Built prior-only diagnostic from recent-20-score mean: `severity_prior_recent_mean_20260524.csv`.

Standalone vs anchor: corr=0.563, diff=0.555, mean=0.514, safe=False. Structurally very different (lower mean). Blends:

| Weight | File | corr vs anchor | diff vs anchor | Safe |
|---|---:|---|---:|---|
| 1.0% | `anchor_severity_prior_w010_20260524.csv` | 1.000 | 0.006 | Yes |
| 2.0% | `anchor_severity_prior_w020_20260524.csv` | 1.000 | 0.011 | Yes | 0.8177 |
| 3.0% | `anchor_severity_prior_w030_20260524.csv` | 0.9995 | 0.017 | Yes | Not submitted |

Recommendation: prior too different from anchor at 2%. Try lower weights (0.5-1%) if revisited.

#### Absolute Error Tree

Trained XGBoost with `reg:absoluteerror` objective on hybrid temporal features. CV MAE 0.401. Script: `scripts/generate_quantile_tree_submission.py`.

Standalone vs anchor: corr=0.764, diff=0.377, safe=False (best standalone XGBoost corr yet). Blends:

| Weight | File | corr vs anchor | diff vs anchor | Safe |
|---|---:|---|---:|---|
| 1.0% | `anchor_abs_tree_w01_20260524.csv` | 1.000 | 0.004 | Yes |
| 2.0% | `anchor_abs_tree_w02_20260524.csv` | 1.000 | 0.008 | Yes | 0.8156 |
| 3.0% | `anchor_abs_tree_w03_20260524.csv` | 1.000 | 0.011 | Yes | Not submitted |

Recommendation: absolute error tree tied with tree 3.0%. Useful but not superior to tree refinement.

Additional quantile variants trained: quantile050 (CV 0.427), quantile045 (CV 0.434), quantile055 (CV 0.426). Blends also generated.

#### Ordinal Classification Tree

Created `model/ordinal_tree.py` and `tests/test_ordinal_tree.py` (2 tests). Trained cumulative threshold classifiers per week on hybrid temporal features. Script: `scripts/generate_ordinal_tree_submission.py`.

Standalone vs anchor: corr=0.772, diff=0.430, safe=False (means 0.91-1.10, much higher than anchor). Blends:

| Weight | File | corr vs anchor | diff vs anchor | Safe |
|---|---:|---|---:|---|
| 0.5% | `anchor_ordinal_tree_w00_20260524.csv` | 1.000 | 0.002 | Yes | Not submitted |
| **1.0%** | `anchor_ordinal_tree_w01_20260524.csv` | 1.000 | 0.006 | Yes | **0.8153** |
| 2.0% | `anchor_ordinal_tree_w02_20260524.csv` | 1.000 | 0.009 | Yes | Not submitted |

Recommendation: ordinal classification is the breakthrough. Set 1% ordinal blend as new anchor. Refine ordinal weight and combine with tree/abs tree next.

#### New Source Files

- `model/severity_history.py` — score-history feature extraction
- `model/ordinal_tree.py` — ordinal probability conversion helpers
- `scripts/generate_severity_prior_submission.py` — prior diagnostic
- `scripts/generate_quantile_tree_submission.py` — abs/quantile tree variants
- `scripts/generate_ordinal_tree_submission.py` — ordinal threshold tree
- `tests/test_severity_history.py` (3 tests)
- `tests/test_ordinal_tree.py` (2 tests)

All 14 tests pass (PYTHONPATH=. required).

### May 22 Generated Candidates (Non-Blend)

| Candidate | File | val_mae | Backtest | Distribution | Submit? |
|---|---|---|---|---|---|
| Calendar-Small-CNN (no scheduler) | `cnn_1d_small_20260522_143903.csv` | 0.362 | (crashed) | corr=0.879, diff=0.239, safe=False | No |
| CNN-GRU | `cnn_1d_cnn_gru_20260522_154103.csv` | 0.243 | 0.185 | corr=0.643, diff=0.496, safe=False | No |

### Regularized CNN-GRU Variants (May 22)

Stronger regularization did not close the distribution gap vs current best:

| Variant | File | val_mae | Correlation | Diff | Shift | Safe |
|---|---|---|---|---|---|---|
| A: do 0.25, wd 0.001 | `cnn_1d_cnn_gru_20260522_160528.csv` | 0.248 | 0.678 | 0.451 | 0.102 | No |
| B: do 0.30, wd 0.002 | `cnn_1d_cnn_gru_20260522_160711.csv` | 0.255 | 0.692 | 0.458 | 0.099 | No |
| C: do 0.25, wd 0.002, lr 0.0005, sched | `cnn_1d_cnn_gru_20260522_160852.csv` | 0.259 | 0.662 | 0.507 | 0.200 | No |

Decision: stop standalone GRU direction. Use the w20 blend for submission.

### Calendar-Small-CNN Scheduler (May 22)

Lightweight temporal backtest with scheduler:

```bash
python scripts/run_temporal_backtest.py --mode cnn --model small --calendar --recent-cutoffs 2 --max-windows-per-region 52 --epochs 3 --batch-size 128 --scheduler
```

Backtest MAE: `0.239607` vs anchor `0.189607` — clearly worse. Full generation skipped.

Decision: abandon standalone calendar features. They consistently hurt temporal generalization (prior non-scheduler candidate also had poor distribution).

## Temporal Backtest

Terminal temporal backtest smoke checks completed on May 21:

| Mode | Command summary | Overall MAE | Output |
|------|-----------------|-------------|--------|
| tree | `--mode tree --recent-cutoffs 2 --max-windows-per-region 52` | 0.300622 | `output/backtests/temporal_backtest_20260520_215534.csv` |
| cnn small | `--mode cnn --model small --recent-cutoffs 1 --max-windows-per-region 52 --epochs 2 --batch-size 256` | 0.249133 | `output/backtests/temporal_backtest_20260521_004824.csv` |

Use temporal backtest scores for relative candidate ranking before future Kaggle submissions. The absolute MAE is not directly comparable to Kaggle public MAE because the public set has additional distribution shift and unknown labels.

### Backtest Calibration Against Known Public Scores

| Candidate | Public MAE | Backtest MAE | Notes |
|---|---:|---:|---|
| small CNN seed 42, 25 epochs | 0.8222 | 0.189607 | Current best anchor. |
| small CNN seed 42, 40 epochs, scheduler | 0.8282 | 0.171465 | Backtest over-rewards longer training. |
| small CNN seed 123, 30 epochs | 0.8512 | 0.199184 | Backtest correctly weaker. |
| tree/XGBoost-ish | 0.8434-ish | 0.300622 | Backtest correctly weaker than CNN. |
| V2 CNN 5 epochs smoke | V2 public 0.8901-0.8967 | 0.205499 | Backtest correctly unattractive. |

Conclusion: backtest is useful for **rejecting bad candidates**, but not sufficient as the only selection rule. Lower backtest MAE alone does not guarantee better public score (40-epoch example). Combine backtest with prediction distribution checks.

### Seed Calibration For Seed 42

| Seed | Backtest MAE |
|---:|---:|
| 42 | 0.189607 |
| 21 | 0.196103 |
| 30 | 0.196760 |
| 0 | 0.201149 |
| 7 | 0.203010 |

Seed 42 remains the strongest. Seed variance is real but not dramatic.

### Conservative Seed-42 Variant Experiment (May 21)

Tested dropout 0.20 (anchor=0.15), weight decay 0.0005 (anchor=0.001), and scheduler variants with seed 42, 25 epochs.

| Variant | Backtest MAE | vs Anchor | Generated? | Distribution vs Best |
|---|---:|---:|---|---|
| dropout 0.20 | 0.185155 | better | `cnn_1d_small_20260521_182510.csv` | corr=0.958, diff=0.155, safe=False |
| weight decay 0.0005 | 0.193362 | slightly worse | not generated | — |
| scheduler | 0.189607 | identical | not generated | — |

Dropout 0.20 had better backtest but predictions drifted too far from current best (higher week means, correlation 0.958 < 0.98). **Not submitted.**

## Running v2

```bash
git pull
rm -f data/*_X_*.npy data/*_y_*.npy data/*_regions_*.npy  # clear cache
# Run All Cells in Project_Analysis.ipynb
```

## Dependencies

- numpy, pandas, matplotlib, scikit-learn, xgboost
- Install: `pip install -e .` (runs setup.py which also downloads Kaggle data)
- Kaggle auth: `~/.kaggle/kaggle.json` or `~/.kaggle/access_token` (auto-converted)
