# Final Project Submission Tracker

| File | Date | MD5 (first 8) | Kaggle Score | Model | Features | Notes |
|------|------|---------------|-------------|-------|----------|-------|
| submission_xgb.csv | May 16 | d61af8d7 | **0.8509** | XGBoost | 84 base | 52 windows/region, GroupKFold |
| submission_xgb_v1.csv | May 16 | 4edb7ebc | 0.8437 | XGBoost | 168 (skew,kurt,trend,quant) | 52 windows/region |
| submission_xgb_v2.csv | May 17 | ? | Error | XGBoost | 168 (skew,kurt,trend,quant) | Rejected upload / wrong submission |
| submission_lgb_v2.csv | May 17 | ? | Error | LightGBM | ? | Rejected upload / wrong submission |
| submission_xgb_v2_fixed.csv | May 17 | 0f6e11a6 | **0.8434** | XGBoost | 168 (skew,kurt,trend,quant) | 52 windows/region, 200 trees, lr=0.05, md=5, n_jobs=2, sample order fixed |
| submission_xgb_20260518_165131.csv | May 18 | ? | ? | ? | ? | auto-generated |
| cnn_1d_20260518_180837.csv | May 18 | 6d71ce7e | **0.8222** | Small 1D CNN | Raw 91-day sequences | 52 windows/region, 25 epochs, val MAE 0.3614, new best |
| temporal_tree_blocks_20260518_172813.csv | May 18 | 68b81f92 | Not submitted | XGBoost temporal blocks | ~938 block features | CV MAE 0.46336, worse than baseline; do not submit |
| temporal_tree_hybrid_20260518_181718.csv | May 18 | eceeddc4 | 0.8403 | GPU XGBoost hybrid temporal tree | 168 baseline + targeted temporal deltas | 52 windows/region, CV MAE 0.36316, second best |
| cnn_1d_v2_20260519_045002.csv | May 19 | f5556ec4 | 0.8901 | V2 1D CNN | Raw 91-day sequences | 52 windows/region, 100 epochs, scheduler, val MAE 0.24172; local validation badly overestimated public performance |
| cnn_1d_small_20260519_044615.csv | May 19 | 91ea58fa | 0.8282 | Small 1D CNN | Raw 91-day sequences | 52 windows/region, 40 epochs, scheduler, val MAE 0.34218; close to previous best but worse public score |
| cnn_1d_v2_20260519_044515.csv | May 19 | e505f325 | 0.8967 | V2 1D CNN | Raw 91-day sequences | 52 windows/region, 40 epochs, scheduler, val MAE 0.24182; local validation badly overestimated public performance |
| cnn_1d_small_blend_20260520_161232.csv | May 20 | 7e75c32a | 0.8258 | Small 1D CNN blend | Raw 91-day sequences | 70/30 blend of current best and seed 123 candidate; corr 0.99307 vs best, diff 0.06413; close but did not beat 0.8222 |
| cnn_1d_small_20260520_161056.csv | May 20 | fc5d9310 | 0.8512 | Small 1D CNN | Raw 91-day sequences | 52 windows/region, seed 123, 30 epochs, dropout 0.20, weight_decay 0.0005, scheduler, val MAE 0.34531; worse public score |
| cnn_1d_small_20260520_160954.csv | May 20 | 211f3a66 | 0.8812 | Small 1D CNN | Raw 91-day sequences | 52 windows/region, seed 7, 25 epochs, dropout 0.15, weight_decay 0.001, scheduler, val MAE 0.36375; worse public score |
| hist_residual_w0p25_20260604.csv | Jun 4 | 281968ee | **0.8088** (tied) | History residual | Blackout history XGB on prob_blend_recycle8089_ord08 | 0.25% residual; gate corr=1.0 diff=9.7e-05; tied best, no improvement |
| prob_subst_season_900_802_full.csv | Jun 4 | d61cf172 | **0.8088** (tied) | Substitution prob | 90% soft8089 + 8% hybrid_ord + 2% season_scalar_soft | gate safe diff=0.0024; stop season substitution |
| prob_subst_season_890_803_full.csv | Jun 4 | 3c181427 | **0.8088** (tied) | Substitution prob | 89% soft8089 + 8% hybrid_ord + 3% season_scalar_soft | gate safe diff=0.0036; stop season substitution |
| prob_subst_history_920_602_full.csv | Jun 4 | 9a360358 | 0.8118 | Substitution prob | 92% soft8089 + 6% hybrid_ord + 2% history_only_ord | gate safe diff=0.016; LB worse; stop history-only substitution |
| prob_blend_8089_oofcal_ord08_full.csv | Jun 5 | 0f2b73fd | 0.8091 | OOF-cal ordinal prob | 92% soft8089 + 8% ordinal_hybrid_oofcal; gate safe diff=0.0049; **worse than 0.8088** |
| prob_mae_q052_8089_ord08_full.csv | Jun 4 | 40bcfc62 | **0.9206** | MAE decision (quantile) | 92% soft8089 + 8% hybrid_ord; q=0.52 not mean | Offline mean 0.47 vs anchor 0.88; **stop median/quantile decision branch** |
