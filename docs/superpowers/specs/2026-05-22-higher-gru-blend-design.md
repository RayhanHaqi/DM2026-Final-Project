# Higher GRU Blend Search Design

## Goal

Search for a higher CNN-GRU blend weight than `w20` while preserving enough distribution safety to justify one final Kaggle submission.

## Context

The `w20` GRU blend is the current best public result:

- File: `output/daily_candidates/cnn_gru_blend_w20_20260522_160258.csv`
- Blend: 80% previous current best + 20% CNN-GRU
- Public MAE: `0.8171`
- Previous best public MAE: `0.8222`

The standalone CNN-GRU is unsafe but useful as a signal:

- File: `output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv`
- Backtest MAE: `0.184923`
- Distribution vs old best: correlation `0.643475`, mean absolute diff `0.495751`, max week mean shift `0.194653`, safe `False`

The successful `w20` result shows the GRU signal transfers to public data when blended conservatively. The next step is to test higher weights offline and spend at most one submission on the best candidate.

## Candidate Weights

Generate these blends first:

- `w22`
- `w24`
- `w25`

Only generate these if `w25` remains plausible after distribution checks:

- `w27`
- `w30`

Approximate expected mean absolute diff vs old best, using standalone GRU diff `0.495751`:

- `w22`: `0.109065`
- `w24`: `0.118980`
- `w25`: `0.123938`
- `w27`: `0.133853`
- `w30`: `0.148725`

These exceed the old `0.10` diff threshold, so they should be judged relative to the successful `w20` as well as the old best.

## Distribution Evaluation

Each candidate must be compared against two references:

1. Old CNN best: `output/daily_candidates/cnn_1d_20260518_180837.csv`
2. New public best `w20`: `output/daily_candidates/cnn_gru_blend_w20_20260522_160258.csv`

The old-best comparison measures absolute drift from the previously trusted model. The `w20` comparison measures incremental risk beyond the known-improved public submission.

## Decision Rule

Submit at most one higher blend.

Prefer the highest blend weight that satisfies all of these:

- correlation vs old best remains at least `0.975`
- mean absolute diff vs old best remains at most `0.13`
- max week mean shift vs old best remains at most `0.055`
- correlation vs `w20` remains at least `0.995`
- mean absolute diff vs `w20` remains at most `0.035`
- prediction range remains within `[0, 5]`

If no higher blend passes, keep `w20` as final best and do not submit again.

## Risk Controls

- Do not train any new model for this search.
- Do not run GPU commands.
- Do not submit more than one higher blend.
- Do not generate `w27` or `w30` if `w25` already looks risky.
- Record every generated path and distribution metric in `AGENTS.md` before submission.

## Expected Outcome

The most likely viable candidates are `w22` or `w24`. Weight `w25` is a useful boundary check. Weights `w27` and `w30` are exploratory only and should be rejected unless their incremental drift from `w20` is surprisingly small.
