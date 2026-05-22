# GRU Best-Blend Candidate Design

## Goal

Create conservative blends between the unsafe standalone CNN-GRU submission and the current best public submission. The purpose is to keep most of the proven current-best distribution while injecting a small amount of the CNN-GRU signal that performed better in temporal backtest.

## Context

Current best public submission:

- File: `output/daily_candidates/cnn_1d_20260518_180837.csv`
- Public MAE: `0.8222`
- Role: reference distribution and primary prediction source

CNN-GRU candidate:

- File: `output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv`
- Local validation MAE: `0.243491`
- Temporal backtest MAE: `0.184923`, better than current-best anchor `0.189607`
- Distribution check: unsafe, with correlation `0.643475`, mean absolute diff `0.495751`, max week mean shift `0.194653`

The standalone GRU should not be submitted. A blend may still be useful because small candidate weights can preserve distribution safety while adding some GRU signal.

## Design

Add a small CLI script, `scripts/blend_submissions.py`, that blends two already-generated submission CSVs. It should not retrain any model and should not require GPU.

The blend formula is:

```text
blend = candidate_weight * candidate + (1 - candidate_weight) * reference
```

Only prediction columns are blended. The ID column, column order, row count, and row order must match the reference submission exactly. Predictions are clipped to `[0, 5]` before saving.

## Candidate Weights

Use conservative candidate weights because GRU is far from the current-best distribution:

- `0.10`
- `0.15`
- `0.20`

The rough expected mean absolute differences are:

- `0.10 * 0.495751 = 0.049575`
- `0.15 * 0.495751 = 0.074363`
- `0.20 * 0.495751 = 0.099150`

Weight `0.20` is near the existing `0.10` mean-diff safety threshold, so it is likely the highest candidate weight worth trying first.

## Submission Gate

Each blend must pass distribution comparison against the current best before any Kaggle submission:

- correlation >= `0.98`
- mean absolute difference <= `0.10`
- max week mean shift <= `0.08`
- prediction range within `[0, 5]`

If multiple blends pass, prefer the largest candidate weight that still passes all gates. If no blend passes, do not submit a GRU blend.

## Risks

- The backtest improvement may not transfer to public score, as seen with the 40-epoch CNN.
- A blend can pass distribution checks and still underperform if the GRU signal is directionally wrong for public data.
- Weight `0.20` is borderline because the standalone GRU distribution is very different.

## Decision Rule

Submit at most one GRU-best blend. Use the highest passing weight among `0.10`, `0.15`, and `0.20`; save any remaining submission budget for later follow-up.
