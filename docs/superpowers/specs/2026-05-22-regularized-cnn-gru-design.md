# Regularized CNN-GRU Candidate Design

## Goal

Test whether stronger regularization can make CNN-GRU predictions closer to the current best submission while preserving the improved temporal backtest signal.

## Context

The first CNN-GRU candidate had strong local metrics but unsafe public-risk signals:

- File: `output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv`
- Local validation MAE: `0.243491`
- Temporal backtest MAE: `0.184923`
- Distribution safety: `safe False`
- Correlation against current best: `0.643475`
- Mean absolute diff: `0.495751`
- Max week mean shift: `0.194653`

This suggests the model is learning a different mapping than the proven small CNN. The next regularization attempts should reduce prediction spread and shift, not chase lower validation MAE alone.

## Design

No new model architecture is required initially. Use existing `--model cnn_gru` and run a small set of conservative training variants with lower learning rate, higher dropout, higher weight decay, and optional scheduler.

The candidates are:

1. GRU dropout 0.25, weight decay 0.001, learning rate 0.001
2. GRU dropout 0.30, weight decay 0.002, learning rate 0.001
3. GRU dropout 0.25, weight decay 0.002, learning rate 0.0005, scheduler enabled

Run generation with `--no-backtest` and `--batch-size 128` to avoid GPU/system memory pressure. Run distribution comparison first. Only run temporal backtest for a generated candidate if distribution is close enough to be plausible.

## Submission Gate

A regularized GRU candidate is submission-eligible only if:

- distribution comparison is `safe True`, or all three distribution metrics are close to thresholds and there is a deliberate decision to risk one submission
- temporal backtest is near or better than the current-best anchor `0.189607`
- generated CSV validates against sample submission

If all three regularized variants fail distribution, stop this direction and do not try more GRU hyperparameters in this session.

## Risk Controls

- Do not chain heavy GPU commands.
- Use `--no-backtest` during generation.
- Use `--batch-size 128`.
- Run at most one lightweight backtest at a time with `--epochs 3`.
- Stop after three variants if distribution still fails.

## Decision Rule

Prefer a distribution-safe GRU blend over a standalone regularized GRU unless the standalone candidate becomes both distribution-safe and backtest-strong. Do not submit any standalone GRU candidate with correlation below `0.98`.
