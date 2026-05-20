---
title: "Natural Disaster Severity Prediction - Progress Report"
subtitle: "NYCU Data Mining 2026 Final Project"
author: "Project Progress Summary"
date: "2026-05-20"
geometry: margin=0.75in
fontsize: 10pt
---

# 1. Project Goal

The final project predicts natural disaster severity for each region over the next five weeks. The Kaggle metric is mean absolute error (MAE), where lower is better.

The dataset contains:

| Item | Value |
|---|---:|
| Regions | 2,248 |
| Training rows | 12.3 million |
| Days per region | 5,480 |
| Meteorological features | 14 |
| Test input | Last 91 days per region |
| Target | 5 weekly severity scores |

The main goal was to build a robust prediction pipeline, improve Kaggle public MAE, and understand why local validation did not always match public leaderboard performance.

# 2. Pipeline Summary

We built several model families and submission utilities.

| Component | Purpose |
|---|---|
| Aggregate XGBoost | Baseline model using summary statistics over 91-day windows |
| Temporal tree features | XGBoost with targeted temporal deltas and block features |
| Small 1D CNN | Learns directly from raw 91-day meteorological sequences |
| V2 CNN | Larger CNN architecture with batch normalization |
| Submission validation | Enforces exact `sample_submission.csv` order and schema |
| Experiment tracking | Logs every submitted file, score, model setting, and notes |
| Temporal backtest | New validation tool to better match Kaggle's future-horizon task |

Important reliability improvements:

- Fixed submission row order to match `sample_submission.csv`.
- Added timestamped submission filenames.
- Added progress bars and safer training defaults.
- Added tests for submission format, experiment generation, scripts, CNN utilities, and temporal backtest logic.
- Avoided full all-window local training after it froze the PC.

## Pipeline Details

### Data And Windowing Pipeline

The raw training data is organized by `region_id` and date. Each region has daily meteorological observations and sparse weekly severity labels. We convert this into supervised learning examples by taking a 91-day meteorological window and predicting the next five available weekly severity scores.

Flow:

| Step | Description |
|---|---|
| Group by region | Keep each region's time series separate |
| Find labeled weeks | Locate rows where `score` is available |
| Build input window | Take the 91 daily rows before the first target week |
| Build target | Use the next 5 weekly scores as a 5-output target |
| Limit windows | Use a safe recent-window setting to avoid local memory/CPU overload |

This framing matches the competition target shape: one row per region and five future weekly predictions.

### Aggregate XGBoost Pipeline

The first modeling pipeline compresses each 91-day window into engineered summary features, then trains a multi-output XGBoost regressor.

Input and transformation:

| Stage | Details |
|---|---|
| Input | 91 days × 14 meteorological features |
| Aggregation | Mean, standard deviation, min, max, quantiles, recent means, trend, skew, kurtosis |
| Feature size | 168 features after adding trend/skew/kurtosis/quantiles |
| Model | `MultiOutputRegressor(XGBRegressor)` predicting 5 weeks |
| Validation | GroupKFold by region |

What we learned:

- XGBoost is stable and fast enough for repeated experiments.
- Better aggregate features improved public MAE from 0.8509 to 0.8434.
- After that, parameter-only tuning gave diminishing returns.

### Temporal Tree Pipeline

The temporal tree pipeline keeps the XGBoost model family but adds more information about when patterns occur inside the 91-day window.

Two feature styles were tested:

| Feature style | Result |
|---|---|
| Hybrid temporal features | Improved tree baseline to 0.8403 public MAE |
| Wide temporal block features | CV MAE 0.46336, not submitted |

The hybrid version kept the useful aggregate features and added targeted temporal deltas, such as recent-vs-full-window changes. This worked better than adding many wide block features.

What we learned:

- Targeted temporal information helps tree models slightly.
- Too many block features can hurt validation and are not worth submission slots.
- Tree models remain useful baselines, but they did not beat the CNN family.

### Small 1D CNN Pipeline

The CNN pipeline avoids manual aggregation and learns directly from the raw 91-day sequence.

Flow:

| Stage | Details |
|---|---|
| Input | 91 × 14 raw meteorological sequence |
| Preprocessing | Standardize features before training/prediction |
| Model | Small 1D convolutional network |
| Pooling | Global average pooling over the time dimension |
| Output | Five severity predictions, one per target week |

This became the strongest pipeline. The best submission, `cnn_1d_20260518_180837.csv`, reached public MAE 0.8222.

What we learned:

- Raw sequence learning captured patterns missed by aggregate tree features.
- The small CNN generalized better than the larger V2 CNN.
- More epochs or different seeds did not automatically improve public score.

### V2 CNN Pipeline

The V2 CNN added more convolution channels, batch normalization, and a deeper dense head.

The motivation was to increase model capacity, but the result showed a validation problem:

| Model | Local validation | Public MAE |
|---|---:|---:|
| Small CNN best | 0.3614 | **0.8222** |
| V2 CNN | 0.2417 | 0.8901 |

What we learned:

- Lower local validation MAE did not guarantee better public MAE.
- The larger model likely overfit the old validation split.
- Architecture changes should wait until validation better matches the public task.

### Submission And Tracking Pipeline

The submission pipeline was important because early submissions failed or used the wrong order.

Reliability steps:

| Step | Purpose |
|---|---|
| Match sample columns | Avoid schema rejection |
| Match sample row order | Avoid wrong region-to-prediction mapping |
| Clip predictions | Keep outputs inside severity range `[0, 5]` |
| Timestamp filenames | Preserve experiment history |
| Log submissions | Track public score, MD5 prefix, model settings, and notes |

What we learned:

- Correct formatting matters as much as model performance.
- The sample submission order differs from raw test group order.
- Tracking every result made it clear which experiments were actually useful.

### Temporal Backtest Validation Pipeline

The latest pipeline is not a submission model. It is validation infrastructure designed to avoid wasting Kaggle slots.

Goal:

| Item | Description |
|---|---|
| Backtest shape | One validation row per region per recent cutoff |
| Input | 91 days before the validation target horizon |
| Target | Next 5 known weekly scores |
| Supported models | Aggregate tree path and CNN sequence path |
| Main purpose | Rank candidates before public submission |

What we expect it to solve:

- Better detect overfitting like the V2 CNN case.
- Reduce reliance on misleading region-only validation.
- Make terminal label drift visible before using submission slots.
- Help decide whether tomorrow's candidates are worth submitting.

\newpage

# 3. Experiment Timeline

## XGBoost Baselines

The first successful models used engineered aggregate features from each 91-day window.

| Submission | Public MAE | Notes |
|---|---:|---|
| `submission_xgb.csv` | 0.8509 | Initial XGBoost baseline with 84 features |
| `submission_xgb_v1.csv` | 0.8437 | Added skew, kurtosis, trend, and quantile features |
| `submission_xgb_v2_fixed.csv` | 0.8434 | Safer params and fixed sample order |

Result: XGBoost was stable, but improvements became small after better aggregate features and safer parameters.

## Temporal Tree Experiments

We then tested whether richer time structure improved tree models.

| Submission | Public MAE | Notes |
|---|---:|---|
| `temporal_tree_hybrid_20260518_181718.csv` | 0.8403 | Best tree result, using baseline features plus targeted temporal deltas |
| `temporal_tree_blocks_20260518_172813.csv` | Not submitted | CV MAE 0.46336, worse than baseline |

Result: targeted temporal deltas helped slightly, but wide block features were not useful.

## CNN Experiments

The strongest model family was the small 1D CNN trained on raw 91-day sequences.

| Submission | Public MAE | Notes |
|---|---:|---|
| `cnn_1d_20260518_180837.csv` | **0.8222** | Current best, small CNN, 25 epochs |
| `cnn_1d_small_20260519_044615.csv` | 0.8282 | Small CNN, 40 epochs, worse public score |
| `cnn_1d_small_blend_20260520_161232.csv` | 0.8258 | 70/30 blend with current best, close but not better |
| `cnn_1d_small_20260520_161056.csv` | 0.8512 | Seed 123, 30 epochs, worse public score |
| `cnn_1d_small_20260520_160954.csv` | 0.8812 | Seed 7, 25 epochs, worse public score |
| `cnn_1d_v2_20260519_045002.csv` | 0.8901 | V2 CNN, strong local validation but poor public score |
| `cnn_1d_v2_20260519_044515.csv` | 0.8967 | V2 CNN, same generalization problem |

Result: the small CNN produced the best score, but later seed, epoch, and architecture changes did not reliably improve public MAE.

\newpage

# 4. Best Result So Far

The current best Kaggle submission is:

| Field | Value |
|---|---|
| File | `cnn_1d_20260518_180837.csv` |
| Public MAE | **0.8222** |
| Model | Small 1D CNN |
| Input | Raw 91-day meteorological sequence |
| Windows per region | Safe local setting, documented as 52 |
| Validation MAE | 0.3614 |

Ranking of best public scores:

| Rank | Submission | Public MAE |
|---:|---|---:|
| 1 | `cnn_1d_20260518_180837.csv` | **0.8222** |
| 2 | `cnn_1d_small_blend_20260520_161232.csv` | 0.8258 |
| 3 | `cnn_1d_small_20260519_044615.csv` | 0.8282 |
| 4 | `temporal_tree_hybrid_20260518_181718.csv` | 0.8403 |
| 5 | `submission_xgb_v2_fixed.csv` | 0.8434 |

# 5. Main Lesson Learned

The main bottleneck is not just model capacity. The main issue is that local validation was not matching the Kaggle test setup.

Current validation methods:

| Model family | Old validation method | Problem |
|---|---|---|
| XGBoost | GroupKFold by region | Tests unseen-region generalization, not terminal future prediction |
| CNN | One GroupShuffleSplit by region | Too noisy for architecture and seed selection |
| V2 CNN | Same CNN validation | Looked excellent locally but failed publicly |

Evidence:

| Candidate | Local validation | Public MAE | Interpretation |
|---|---:|---:|---|
| Small CNN best | 0.3614 | **0.8222** | Best public model despite weaker local score |
| V2 CNN | 0.2417 | 0.8901 | Local validation badly overestimated performance |
| Small CNN seed 123 | 0.3453 | 0.8512 | Better local score, worse public score |
| Small CNN blend | distributionally safe | 0.8258 | Reduced risk, but did not improve best score |

Conclusion: we need a validation method that mimics the actual Kaggle task: predicting a future 5-week horizon from the terminal 91-day window.

\newpage

# 6. Why Improvements Became Hard

We identified several reasons for the performance plateau.

## Validation Mismatch

Kaggle uses the same 2,248 regions and asks for future predictions. Our old validation held out regions or used a single region split. That does not fully test the same kind of extrapolation.

## Strong Temporal Drift

Recent historical labels are much lower than the overall training distribution. This means random or region-based validation can be misleading.

Observed label means:

| Distribution | Mean severity |
|---|---:|
| Overall training score mean | 0.8357 |
| Cached training-window target mean | 0.7284 |
| Last 5 known labels mean | 0.3423 |

## CNN Overfitting Risk

The larger V2 CNN produced excellent local validation scores but poor Kaggle scores. This suggests the local split rewarded patterns that did not transfer to the public test horizon.

## Limited Diversity In Blends

The May 20 blend was highly correlated with the best CNN. It reduced damage from a worse model but did not add enough complementary signal to beat the current best.

## Tree Model Ceiling

Aggregate XGBoost is stable, but compressed summary statistics lose timing information. Wide temporal blocks added too many features and degraded validation.

# 7. Current Work: Better Validation

We started implementing a temporal backtest system.

Goal:

- validate models on recent historical future horizons
- use one validation row per region per cutoff
- input = 91 days before the validation horizon
- target = next 5 known weekly scores
- support both tree and CNN pipelines

Progress so far:

| Component | Status |
|---|---|
| Split builder | Implemented and tested |
| Tree backtest evaluator | Implemented and tested |
| CNN backtest evaluator | Implemented and tested |
| CLI script | Implemented and tested |
| Unit tests | 28 tests passed in latest full regression run |
| Tree smoke run | In progress or pending |

This tool should help decide which candidates are worth future Kaggle submissions.

\newpage

# 8. Next Steps

Immediate next steps:

1. Finish smoke-testing the temporal backtest CLI.
2. Compare old validation ranking against temporal backtest ranking.
3. Re-evaluate CNN variants using terminal-style validation before spending more Kaggle submissions.
4. Avoid submitting simple seed variants unless temporal backtest supports them.
5. Prefer candidates with genuinely different error profiles, such as CNN plus temporal tree or calibrated residual models.

Potential modeling directions after validation improves:

| Direction | Reason |
|---|---|
| Recency-aware CNN pooling | Current small CNN uses global average pooling, which may weaken recent-day signal |
| Date/season features | Public target months are not uniformly distributed |
| CNN plus temporal-tree blend | More diverse than same-CNN seed blending |
| Region-level priors | Train and test share the same region universe |
| Calibration by week | May reduce systematic week-by-week bias |

# 9. Five-Minute Presentation Flow

Use this order for a short presentation:

| Time | Topic |
|---:|---|
| 0:00-0:45 | Problem setup: predict 5-week disaster severity, MAE metric |
| 0:45-1:30 | Pipeline: XGBoost aggregates, temporal trees, CNNs, submission validation |
| 1:30-2:30 | Experiment results: XGBoost 0.8434, temporal tree 0.8403, best CNN 0.8222 |
| 2:30-3:30 | Failed improvements: V2 CNN and seed variants looked good locally but failed publicly |
| 3:30-4:30 | Root cause: validation mismatch and terminal label drift |
| 4:30-5:00 | Next step: temporal backtest before future submissions |

# 10. Summary

The project improved from an initial XGBoost public MAE of 0.8509 to a best public MAE of 0.8222 using a small 1D CNN. The largest remaining issue is not simply finding a bigger model, but building validation that predicts leaderboard behavior more reliably. The new temporal backtest work directly addresses this issue and should guide the next round of experiments.
