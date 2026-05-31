# DM2026 Final Project: Natural Disaster Severity Prediction

## Preliminary Report

**Author:** Muhammad Rayhan Athaillah (賴睿涵)  
**Student ID:** 313540001  
**Course:** Data Mining (Spring 2026), National Yang Ming Chiao Tung University  
**Date:** May 31, 2026  
**Competition:** Data Mining 2026 Final Project (Kaggle)  
**Metric:** Mean Absolute Error (MAE), range 0-5, lower is better  
**GitHub Repository:** [DM2026-Final-Project]

---

## 1. Executive Summary

This report presents our approach to predicting natural disaster severity scores (0-5) for 2,248 regions over 5 weeks. Our final model achieves **0.8113 MAE** on the public leaderboard, representing a **4.6% improvement** over the XGBoost baseline (0.8509).

**Key Contributions:**
- Ordinal classification breakthrough: treating severity as ordered classes via cumulative thresholds
- Same-season severity features: month-of-year historical severity patterns
- Ensemble blending: combining XGBoost, LightGBM, and season tree corrections

**Connection to Course Assignments:**
This project builds on skills developed across three prior assignments:
- **Assignment 1** (Linear/Logistic Regression): Foundation in regression and classification metrics
- **Assignment 2** (SVM, FP-growth, PCA+K-Means): Feature engineering and clustering techniques
- **Assignment 3** (Human Activity Recognition): GroupKFold validation, XGBoost/LightGBM tuning, and ensemble methods — directly applicable to this final project

---

## 2. Problem Description

### 2.1 Task
Predict weekly severity scores (0-5) for 2,248 regions based on 91-day meteorological windows.

### 2.2 Data Overview

| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| Train | 12,319,040 | 17 | Meteorological features + severity scores |
| Test | 204,568 | 16 | Meteorological features (no scores) |
| Submission | 2,248 | 6 | Region ID + 5 weekly predictions |

**Features (14 meteorological):**
- `prec` (precipitation), `surf_pre` (surface pressure), `humidity`
- `tmp`, `dp_tmp`, `wb_tmp` (temperature variants)
- `tmp_max`, `tmp_min`, `tmp_range` (temperature extremes)
- `surf_tmp` (surface temperature)
- `wind`, `wind_max`, `wind_min`, `wind_range`

**Target:** `score` (0-5, discrete severity levels)

### 2.3 Evaluation
- Metric: Mean Absolute Error (MAE)
- Range: 0-5 (lower is better)
- Budget: 10 submissions/day

---

## 3. Methodology

### 3.1 Feature Engineering

#### Base Features (168 features)
From 91-day windows for each of 14 meteorological features:
- Statistics: mean, std, min, max, q25, q50, q75
- Temporal: last7_mean, last30_mean, trend
- Higher-order: skewness, kurtosis

*This approach mirrors Assignment 3's 42-feature aggregation (6 channels × 7 stats), extended to 14 meteorological channels.*

#### Temporal Features (294 features total)
Block-based temporal features:
- Full 91-day, first 30, middle 30, last 31, last 30, last 14, last 7
- Deltas between blocks (last7-full91, last14-full91, etc.)

*Similar to Assignment 3's "targeted temporal features" which provided the best improvement (+0.0238 over base features).*

#### Same-Season Features (4 features)
Month-of-year severity statistics:
- `season_month`: target month
- `season_mean`: historical mean severity for that month
- `season_median`: historical median severity
- `season_count`: number of historical observations
- `season_zero_frac`: fraction of zero scores

#### Severity History Features (15 features)
- `score_last_known`, `score_mean_all`, `score_median_all`
- `score_mean_last5`, `score_mean_last10`, `score_mean_last20`, `score_mean_last52`
- `score_zero_frac`, `score_high_frac_ge3`
- `score_class_frac_0` to `score_class_frac_5`

### 3.2 Models

#### XGBoost (Primary Model)
- Multi-output regression via `MultiOutputRegressor(XGBRegressor)`
- 5 separate models for each week
- Best params: depth=7, trees=300, lr=0.05, subsample=0.8, colsample=0.9

*Same architecture as Assignment 3's best model (XGBoost with Optuna tuning), adapted for regression.*

#### Ordinal Classification
- Treat scores as ordered classes 0..5
- Train 5 binary classifiers per week (threshold 1,2,3,4,5)
- Convert threshold probabilities to class probabilities
- Expected value = weighted sum of class IDs

*Novel approach for this project, exploiting the ordinal structure of severity scores.*

#### LightGBM
- Alternative gradient boosting implementation
- Best params: num_leaves=70, depth=6, lr=0.05, trees=300

*Used in Assignment 3 as secondary model; here serves as ensemble diversifier.*

#### Neural Networks (CNN)
- Small 1D CNN: Conv1d(32) → Conv1d(64) → AdaptiveAvgPool → Linear(5)
- CNN-GRU: Conv1d → GRU → Linear
- Best standalone: 0.8222 MAE (small CNN)

*Similar to Assignment 3's CNN experiments, though CNN performed poorly there (0.6709 accuracy).*

### 3.3 Ensemble Strategy

**Final Model Composition:**
```
72% XGBoost anchor (tree4 + ordinal1.5%)
+ 18% Season tree (same-month severity correction)
+ 10% LightGBM (alternative gradient boosting)
```

**Blend Weight Search:**
- Season tree: linear improvement from 2% to 20%, degradation at 25%
- LightGBM: optimal at 10%, improvement from 0.8120 to 0.8113

*Conservative blending strategy learned from Assignment 3, where broad feature expansion and complex models consistently overfit.*

---

## 4. Experiments & Results

### 4.1 Submission History (Top 10)

| Rank | Date | File | MAE | Description |
|------|------|------|-----|-------------|
| 1 | May 31 | lgbm_blend_w10 | **0.8113** | 10% LightGBM into anchor |
| 2 | May 31 | gs_blend_w10 | **0.8113** | 10% grid search XGB into anchor |
| 3 | May 31 | lgbm_blend_w05 | 0.8116 | 5% LightGBM |
| 4 | May 31 | gs_blend_w05 | 0.8116 | 5% grid search XGB |
| 5 | May 31 | lgbm_blend_w02 | 0.8118 | 2% LightGBM |
| 6 | May 31 | gs_blend_w02 | 0.8118 | 2% grid search XGB |
| 7 | May 30 | season_w20 | 0.8120 | 20% season tree |
| 8 | May 30 | season_w15 | 0.8122 | 15% season tree |
| 9 | May 30 | season_w25 | 0.8123 | 25% season tree |
| 10 | May 30 | season_w12 | 0.8124 | 12% season tree |

**Total submissions:** 51

### 4.2 Key Milestones

| Date | Milestone | MAE | Delta |
|------|-----------|-----|-------|
| May 18 | XGBoost baseline | 0.8509 | — |
| May 18 | Small CNN | 0.8222 | -0.0287 |
| May 22 | GRU blend (w25) | 0.8167 | -0.0055 |
| May 23 | Tiny tree correction | 0.8159 | -0.0008 |
| May 24 | Ordinal classification | 0.8153 | -0.0006 |
| May 25 | tree3+ordinal1.5% | 0.8149 | -0.0004 |
| May 26 | tree4+ordinal1.5% | 0.8145 | -0.0004 |
| May 28-29 | Season tree (2-20%) | 0.8120 | -0.0025 |
| **May 31** | **LightGBM blend** | **0.8113** | **-0.0007** |

### 4.3 Failed Experiments

| Approach | MAE | Reason |
|----------|-----|--------|
| PatchTST | 1.04-1.21 | Complete failure, poor generalization |
| Meta-stacking Ridge | 1.16 | Complete failure |
| CatBoost | 0.8149-0.8156 | Region categorical signal not helpful |
| V2 CNN (larger) | 0.8901-0.8967 | Overfitting to validation |
| Grid search standalone | 0.849-0.850 | Overfitting to CV |

*Consistent with Assignment 3 findings: "rich temporal features, ExtraTrees, naive OOF ensembles... did not improve public score."*

### 4.4 Grid Search Results

**XGBoost (108 combinations):**
- Best CV MAE: 0.2888
- Best params: depth=7, trees=300, lr=0.05, subsample=0.8, colsample=0.9, reg_alpha=0.1, reg_lambda=1.0
- Standalone public MAE: 0.8491 (overfitting)

**LightGBM (81 combinations):**
- Best CV MAE: 0.3153
- Best params: num_leaves=70, depth=6, lr=0.05, trees=300
- Standalone public MAE: 0.8470 (overfitting)

**Key Finding:** CV MAE is not predictive of public MAE. Standalone grid search models overfit, but blending at 10% improves generalization.

*This mirrors Assignment 3's experience: "Local CV/OOF improvements have not transferred reliably to public score."*

---

## 5. Technical Implementation

### 5.1 Architecture

```
model/
├── train.py              # XGBoost multi-output training
├── utils.py              # Feature aggregation (168 features)
├── experiments.py        # Submission utilities
├── temporal_features.py  # Temporal block features
├── temporal_tree.py      # Tree-based temporal models
├── ordinal_tree.py       # Ordinal classification helpers
├── same_season.py        # Month-of-year severity features
├── severity_history.py   # Score history features
├── cnn_candidate.py      # CNN models
├── patchtst.py           # PatchTST model
└── backtest.py           # Temporal backtest utilities

scripts/
├── generate_season_tree_submission.py
├── generate_ordinal_tree_submission.py
├── generate_lgbm_submission.py
├── grid_search_xgboost.py
├── blend_submissions.py
└── ... (15+ generation scripts)
```

*Architecture follows Assignment 3's pattern: `model/` for core logic, `scripts/` for experiment runners, `output/` for submissions.*

### 5.2 Data Pipeline

1. **Raw data** → 91-day sliding windows (52 windows per region)
2. **Feature extraction** → 294 hybrid temporal features + 4 season features
3. **Model training** → 5 separate week models (GroupKFold by region)
4. **Ensemble blending** → Weighted average of multiple models
5. **Post-processing** → Clip predictions to [0, 5]

### 5.3 Cross-Validation Strategy

- GroupKFold with 5 splits by region
- Prevents data leakage: windows from same region stay together
- Validates temporal generalization

*Same validation strategy as Assignment 3: "GroupKFold partitioning by user... prevents samples from the same user from leaking across train/validation folds."*

---

## 6. Analysis & Insights

### 6.1 What Worked

1. **Ordinal Classification (May 24)**
   - Treating scores as ordered classes via cumulative thresholds
   - Aligns model structure with discrete ordinal target
   - Breakthrough improvement from 0.8153 to 0.8145

2. **Same-Season Features (May 28-29)**
   - Month-of-year historical severity patterns
   - Linear improvement from 2% to 20% blend weight
   - Strongest new signal since ordinal classification

3. **Ensemble Blending (May 31)**
   - Combining diverse models (XGBoost, LightGBM, season tree)
   - 10% blend weight optimal for new models
   - Final improvement from 0.8120 to 0.8113

### 6.2 What Didn't Work

1. **Larger Neural Networks (PatchTST, V2 CNN)**
   - Better validation MAE but worse public MAE
   - Overfitting to local validation distribution

2. **Standalone Grid Search Models**
   - CV MAE not predictive of public MAE
   - Must blend with existing anchor for improvement

3. **CatBoost with Region Categoricals**
   - Ordered target statistics not helpful
   - XGBoost with hand-crafted features superior

### 6.3 Key Insights

1. **CV ≠ Public Score:** Local validation MAE is not reliable for model selection. Distribution checks and safe blending are essential.

2. **Blend Conservatively:** New models should be blended at low weights (2-10%) into proven anchors. Higher weights risk overfitting.

3. **Feature Engineering > Model Complexity:** Hand-crafted temporal and season features outperform complex architectures.

4. **Ordinal Structure Matters:** Treating severity as ordered classes (not regression) aligns with the problem structure.

*These insights align with Assignment 3's conclusions: "Keep feature additions compact and ablated. Broad feature expansion has repeatedly hurt or failed to transfer."*

---

## 7. Future Work

### 7.1 Short-term (Next Submission)

- Submit LightGBM blends at 15%, 20%, 25%, 30% (prepared)
- Test if improvement continues beyond 10% weight
- Explore combining grid search + LightGBM blends

### 7.2 Medium-term

1. **Richer Severity History**
   - Region-specific trend (slope of last 20 scores)
   - Year-over-year same-month comparison
   - Volatility features (std of recent scores)

2. **Stacking with OOF Predictions**
   - Generate OOF predictions from tree/ordinal/season/LightGBM
   - Train ridge meta-model on these predictions

3. **Neural Architecture Search**
   - Try Temporal Convolutional Network (TCN)
   - Multi-scale CNN (parallel convolutions)
   - Attention mechanisms on CNN features

### 7.3 Long-term

1. **Region-specific Models**
   - Cluster regions by severity patterns
   - Train separate models per cluster

2. **Temporal Ensemble**
   - Weight recent data more heavily
   - Adaptive blend weights based on recency

3. **Feature Selection**
   - Identify most important features per week
   - Remove noisy features to reduce overfitting

---

## 8. Conclusion

We achieved **0.8113 MAE** on the public leaderboard, a **4.6% improvement** over the baseline. Our approach combines:

- **Ordinal classification** for discrete severity prediction
- **Same-season features** for historical severity patterns
- **Ensemble blending** of XGBoost, LightGBM, and season corrections

The key insight is that **feature engineering and conservative blending** outperform complex model architectures. Future work will focus on richer severity history features and stacking approaches.

**Lessons from Assignments:**
This project applies lessons from all three course assignments:
- Assignment 1: Regression fundamentals and evaluation metrics
- Assignment 2: Feature engineering and dimensionality considerations
- Assignment 3: GroupKFold validation, gradient boosting tuning, and conservative ensemble strategies

---

## 9. References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
2. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
3. Frank, E., & Hall, M. (2001). A simple approach to ordinal classification. *ECML*.
4. DM2026 Assignment 1: Linear/Logistic Regression (NYCU, Spring 2026).
5. DM2026 Assignment 2: SVM, FP-growth, PCA+K-Means (NYCU, Spring 2026).
6. DM2026 Assignment 3: Human Activity Recognition (NYCU, Spring 2026).

---

## Appendix A: Submission Commands

```bash
# Generate current best submission
python scripts/blend_submissions.py \
  --candidate output/daily_candidates/lgbm_best_20260531_020405.csv \
  --reference output/daily_candidates/season_w20_w20_20260530_122421.csv \
  --candidate-weight 0.10 \
  --name lgbm_blend_w10

# Submit to Kaggle
kaggle competitions submit -c data-mining-2026-final-project \
  -f output/daily_candidates/lgbm_blend_w10_w10_*.csv \
  -m "10% LightGBM blend"
```

## Appendix B: Reproducibility

**Environment:**
- Python 3.10.20
- XGBoost 2.0+
- LightGBM 4.6.0
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.24+

**Random Seeds:** All models use `random_state=42`

**Thread Control:** All models use `n_jobs=2` with environment variables set to 2

**Data Cache:** Features cached as `.npy` files for fast loading

## Appendix C: Assignment Connections

| Assignment | Skills Applied | This Project |
|------------|----------------|--------------|
| Assignment 1 | Regression, Classification metrics | MAE evaluation, score prediction |
| Assignment 2 | Feature engineering, PCA | Temporal features, season features |
| Assignment 3 | GroupKFold, XGBoost/LightGBM, Ensemble | Validation strategy, model tuning, blending |

## Appendix D: Project Timeline

| Date | Phase | Key Activities |
|------|-------|----------------|
| May 16-18 | Setup | Data exploration, baseline XGBoost, first CNN |
| May 19-21 | CNN Exploration | Small CNN, V2 CNN, GRU variants, temporal backtest |
| May 22-23 | GRU Blending | Seed variants, blend weight search, stability ensemble |
| May 24-25 | Ordinal Breakthrough | Ordinal classification, tree refinements |
| May 26-27 | Feature Expansion | History features, blackout temporal, PatchTST (failed) |
| May 28-30 | Season Tree | Same-month severity, weight optimization |
| May 31 | Grid Search + LightGBM | Parameter sweep, ensemble blending, new best |
