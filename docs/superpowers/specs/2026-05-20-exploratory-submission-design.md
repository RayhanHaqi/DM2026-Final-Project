# Exploratory Submission Design - 2026-05-20

## Goal

Use the refreshed daily Kaggle budget to test higher-variance ideas while protecting the current best public score. The current best submission is `output/daily_candidates/cnn_1d_20260518_180837.csv` with public MAE `0.8222`.

## Scope

Today's work is limited to generating, validating, and selecting up to three submission candidates. It should not add new model infrastructure unless a candidate cannot be produced with the existing scripts.

## Candidate Mix

1. Small CNN seed variant: run the existing `small` CNN for `25` epochs with `seed=7`, `dropout=0.15`, and the scheduler enabled. This is the safest exploratory slot because it stays close to the current best model family.
2. Small CNN training variant: run the existing `small` CNN with a different seed and modestly changed regularization, such as `dropout=0.20` or adjusted `weight_decay`. This explores training variance without changing architecture.
3. Blend candidate: blend the current best CNN with the best new small-CNN candidate, using a conservative blend such as `70/30` or `50/50` after inspecting prediction distributions.

## Rejected Options

- Do not submit V2 CNN as-is. It produced strong local validation MAE (`0.2417-0.2418`) but poor public MAE (`0.8901-0.8967`).
- Do not submit wide temporal block features. Their CV MAE was `0.46336` and they are already marked as bad.
- Do not spend the exploratory budget on XGBoost parameter-only candidates. The previous daily XGBoost/blend attempt scored `0.9736` publicly.

## Validation Gate

Every candidate must pass these checks before any Kaggle upload:

- CSV columns and row order match `data/sample_submission.csv`.
- Prediction values are clipped to `[0, 5]`.
- Week-level prediction means are not extreme and do not show the V2 downward-drift pattern.
- Correlation with the current best CNN is reasonably high.
- Mean absolute difference from the current best CNN is moderate, not V2-level divergent.

## Submission Policy

Use at most three submissions:

1. Submit the best-looking new small CNN candidate.
2. Submit a second new candidate only if it passes the validation gate and is not redundant with the first.
3. Submit a blend only if it is distributionally safer than the raw candidate or meaningfully different from both raw candidates.

After each Kaggle result, update `AGENTS.md` and `output/SUBMISSIONS.md` with the file path, model settings, validation notes, public MAE, and any rejection reason.

## Testing And Verification

Before generating submissions, run the lightweight test suite and script help checks. After generating candidates, validate submission schema and compare each prediction file against the current best CNN before uploading.
