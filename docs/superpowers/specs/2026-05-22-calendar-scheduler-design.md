# Calendar-Small-CNN Scheduler Candidate Design

## Goal

Retest the Calendar-Small-CNN direction with scheduler enabled and safer memory settings to see whether convergence stabilizes enough to pass distribution safety.

## Context

Calendar-Small-CNN without scheduler generated successfully but failed distribution safety:

- File: `output/daily_candidates/cnn_1d_small_20260522_143903.csv`
- Validation MAE: `0.361976`
- Distribution correlation: `0.879148`
- Mean absolute diff: `0.238959`
- Max week mean shift: `0.113914`
- Decision: not submitted

The failed distribution suggests the calendar features changed model behavior too much. Scheduler may improve convergence, but it is not expected to fix a large distribution shift by itself. This should be treated as a last-priority experiment after GRU blends and regularized GRU variants.

## Design

Use the existing small CNN architecture with `--calendar` and `--scheduler`. Do not add architecture changes. Use memory-safe settings:

- `--batch-size 128`
- separate backtest and generation commands
- avoid chained commands

Run a lightweight calendar backtest first. Generate a full candidate only if backtest does not crash and is not obviously worse than the anchor.

## Gates

Generation gate:

- lightweight backtest runs without `Killed`
- backtest is near or better than anchor `0.189607`, or at least not much worse than the prior calendar candidate expectation

Submission gate:

- distribution comparison must be `safe True`
- if distribution remains near corr `0.879` or diff `0.239`, reject immediately

## Risk Controls

- Do not run calendar backtest and full training in one chained command.
- Use `--batch-size 128`.
- Use `--no-backtest` during full generation after the standalone backtest.
- Stop after one scheduler run if it fails distribution.

## Decision Rule

Submit only if scheduler makes Calendar-Small-CNN distribution-safe. Otherwise abandon standalone calendar features and keep the current best submission as the reference.
