# New GRU Seed Blend Design

## Goal

Train one additional CNN-GRU candidate with a different seed, then use conservative blending to test whether a second GRU seed provides complementary signal beyond the current `w25` best.

## Context

Current best:

- `output/daily_candidates/cnn_gru_blend_w25_20260522_163741.csv`
- Public MAE: `0.8167`

Original GRU signal:

- seed `42`
- `output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv`
- unsafe standalone but useful when blended

Regularized GRU variants did not fix standalone distribution mismatch. A different seed may still provide useful ensemble diversity if blended conservatively.

## Candidate Seed

Use seed `21` first. Seed `21` was the second-best small-CNN seed in prior calibration. If seed 21 fails badly, do not train seed 30 in the same execution unless explicitly approved.

## Execution Constraint

The assistant must not run the full training command because the user wants to preserve token/runtime budget and avoid unattended GPU-heavy work. The assistant should provide the exact command, then wait for the user to paste terminal output.

The user-run command is:

```bash
python scripts/generate_cnn_submission.py --model cnn_gru --no-backtest --max-windows-per-region 52 --epochs 25 --seed 21 --dropout 0.15 --weight-decay 0.001 --batch-size 128
```

## Blending Strategy

After the user provides the generated seed-21 GRU path, generate conservative blends:

- 90% `w25` + 10% seed-21 GRU
- 85% `w25` + 15% seed-21 GRU

Only try 20% if 15% is clearly safe.

## Evaluation

Compare every blend against current best `w25`:

- correlation >= `0.995`
- mean absolute diff <= `0.035`
- max week mean shift <= `0.02`
- predictions in `[0, 5]`

Also compare against old CNN best for context.

## Decision Rule

Submit at most one new-seed GRU blend. Prefer the highest seed-21 blend that passes all gates. If no blend passes, submit nothing.

## Risk Controls

- Do not run the full GRU training command from the assistant session.
- Do not run Kaggle submission commands from the assistant session.
- Use `--no-backtest` and `--batch-size 128` for the user-run training command.
- Stop after seed 21 unless the user explicitly approves seed 30.
