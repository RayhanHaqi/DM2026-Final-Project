# Tiny 40-Epoch CNN Blend Design

## Goal

Test whether adding a very small amount of the older 40-epoch small CNN candidate to the current `w25` best improves public score without disrupting the proven GRU-blend distribution.

## Context

Current best:

- `output/daily_candidates/cnn_gru_blend_w25_20260522_163741.csv`
- Public MAE: `0.8167`

40-epoch small CNN candidate:

- `output/daily_candidates/cnn_1d_small_20260519_044615.csv`
- Public MAE: `0.8282`
- Backtest was very strong (`0.171465`) but public score was worse than the 25-epoch CNN.

The 40-epoch candidate should not be used directly. It may still provide complementary signal if blended at a tiny weight.

## Candidate Weights

Generate:

- 95% `w25` + 5% 40-epoch CNN
- 90% `w25` + 10% 40-epoch CNN

If 10% is clearly unsafe, do not try higher weights.

## Evaluation

Compare each candidate against current best `w25`:

- correlation >= `0.995`
- mean absolute diff <= `0.035`
- max week mean shift <= `0.02`
- predictions in `[0, 5]`

Also compare against old CNN best for context, but the primary safety reference is `w25` because `w25` is the proven public leader.

## Decision Rule

Submit at most one tiny 40-epoch blend. Prefer the 10% blend only if it passes all gates; otherwise use the 5% blend if it passes. If both fail, do not submit.

## Execution Constraint

This plan is CPU-only. Do not run CNN training or Kaggle submission from the assistant session. The user performs the final Kaggle submission manually.
