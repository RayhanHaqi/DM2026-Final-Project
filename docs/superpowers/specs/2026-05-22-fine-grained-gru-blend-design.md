# Fine-Grained GRU Blend Search Design

## Goal

Search narrowly around the current best GRU blend plateau to see whether a fractional blend weight improves over public MAE `0.8167` without jumping to the unsafe `w27` region.

## Context

Known public scores:

- `w20`: `cnn_gru_blend_w20_20260522_160258.csv`, public MAE `0.8171`
- `w25`: `cnn_gru_blend_w25_20260522_163741.csv`, public MAE `0.8167`
- `w26`: `cnn_gru_blend_w26_20260522_170137.csv`, public MAE `0.8167`

Known safety boundary:

- `w25` passes relaxed old-best and incremental gates.
- `w26` is almost identical to `w25` but misses corr-vs-old by about `0.0003`; public tied `w25`.
- `w27` fails corr-vs-old and diff-vs-old.
- `w30` fails all gates.

This suggests the useful region is a plateau around `w25` to `w26`, and the risk increases quickly by `w27`.

## Candidate Weights

Generate only these fractional weights:

- `w25.5`
- `w26.5`

Do not generate or submit `w27+` in this plan. The already-tested `w27` was rejected.

## Evaluation

Each candidate is compared against:

1. Old CNN best: `output/daily_candidates/cnn_1d_20260518_180837.csv`
2. Current best `w25`: `output/daily_candidates/cnn_gru_blend_w25_20260522_163741.csv`
3. Tied current best `w26`: `output/daily_candidates/cnn_gru_blend_w26_20260522_170137.csv`

The old-best comparison measures absolute drift from the original trusted CNN. The `w25`/`w26` comparisons measure whether the candidate is only a tiny perturbation of the known public-best plateau.

## Decision Rule

Submit at most one fractional blend.

Prefer `w26.5` only if all of these hold:

- correlation vs old best >= `0.974`
- mean absolute diff vs old best <= `0.132`
- max week mean shift vs old best <= `0.052`
- correlation vs w25 >= `0.9998`
- mean absolute diff vs w25 <= `0.010`
- prediction range remains within `[0, 5]`

If `w26.5` fails but `w25.5` passes, submit `w25.5`. If both fail, submit nothing and keep `w25`/`w26` as final best.

## Execution Constraint

This plan is CPU-only and should not run any model training. The assistant may generate blend CSVs and run distribution checks, but should not submit to Kaggle. The user performs the final Kaggle submission manually.
