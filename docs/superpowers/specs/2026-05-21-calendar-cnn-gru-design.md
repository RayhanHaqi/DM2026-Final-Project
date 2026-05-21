# Calendar CNN And CNN-GRU Candidate Design

## Goal

Build two independent next-step Kaggle candidates from the current best small 1D CNN baseline:

- **Calendar-Small-CNN**: same model family as the current best, with seasonal calendar signals added to each 91-day input sequence.
- **CNN-GRU Hybrid**: a small sequence-aware neural model that combines Conv1D local pattern extraction with one GRU layer.

Each candidate is allowed at most one Kaggle submission. The remaining daily submission budget should be saved for a later blend or follow-up candidate after public feedback.

## Current Baseline

The current best public result is `output/daily_candidates/cnn_1d_20260518_180837.csv` with public MAE `0.8222`. It uses raw 91-day meteorological sequences, 52 windows per region, seed 42, 25 epochs, dropout 0.15, weight decay 0.001, and the existing small 1D CNN.

Recent calibration showed that temporal backtest is useful as a rejection filter, but not enough by itself. The 40-epoch CNN had a better backtest score but worse public MAE, and the dropout 0.20 variant improved backtest but failed distribution safety. Future candidates must pass both temporal backtest and distribution checks before submission.

## Candidate A: Calendar-Small-CNN

Calendar-Small-CNN keeps the existing small CNN architecture and training defaults, but appends deterministic calendar features to every daily row in the 91-day sequence:

- `calendar__doy_sin`
- `calendar__doy_cos`
- `calendar__month_sin`
- `calendar__month_cos`
- `calendar__week_sin`
- `calendar__week_cos`

These features should be generated from the existing `date` column for both training and test data. They are deterministic, require no label information, and do not change the target definition. The purpose is to help the baseline model learn seasonal patterns without changing the architecture.

Calendar features are enabled by an explicit CLI flag, not by default. This preserves the current baseline path and keeps historical commands reproducible.

## Candidate B: CNN-GRU Hybrid

CNN-GRU Hybrid adds one new `model_name` option to the existing CNN pipeline. It should use the same input format, training function, standardization, temporal backtest, submission builder, and distribution checks as the current CNN candidates.

The model shape is:

- Input: `(batch, 91, n_features)`.
- Transpose to `(batch, n_features, 91)` for Conv1D.
- Conv1D 1: `n_features -> 32`, kernel size 5, padding 2, ReLU.
- Conv1D 2: `32 -> 64`, kernel size 5, padding 2, ReLU.
- Transpose back to `(batch, 91, 64)` for GRU.
- GRU: input size 64, hidden size 64, one layer, batch-first.
- Dropout on final hidden state.
- Linear head: `64 -> 5` weekly severity predictions.

Use GRU before LSTM because it is smaller, faster, and less likely to overfit with the local 52-window training cap. Plain RNN should not be implemented because it is weaker and less stable for this task.

## Data Flow

The existing data flow remains the source of truth:

1. Read `data/train.csv` and `data/test.csv`.
2. Optionally append calendar columns to copies of both frames.
3. Build sliding 91-day training windows and 5-week labels.
4. Build 91-day test windows using the same feature columns.
5. Standardize using train means and standard deviations.
6. Train the selected model.
7. Clip predictions to `[0, 5]`.
8. Build and validate a submission against `data/sample_submission.csv`.
9. Save timestamped CSV and summary files under `output/daily_candidates/`.

Temporal backtest must use the same candidate settings as final training, including `model_name` and whether calendar features are enabled.

## Submission Gates

No candidate should be submitted solely because it improves local validation MAE. Submit only if all required gates pass:

- Temporal backtest is better than the current anchor `0.189607`, or close enough that distribution safety makes it plausible.
- Distribution comparison against `output/daily_candidates/cnn_1d_20260518_180837.csv` passes or is explicitly judged safe:
  - correlation >= `0.98`
  - mean absolute difference <= `0.10`
  - max week mean shift <= `0.08`
  - predictions are within `[0, 5]`
- The generated CSV matches sample-submission rows and columns.

If both candidates pass the gates, submit Calendar-Small-CNN first unless CNN-GRU has a clearly stronger backtest improvement while still passing distribution safety. Save the final daily submission for a blend or a combined calendar-plus-GRU follow-up after seeing public feedback.

## Testing Requirements

Tests must cover:

- Calendar feature generation produces deterministic sin/cos columns with expected values.
- Sequence train/test builders include calendar columns only when requested.
- Backtest passes the calendar flag into its sample builder.
- `build_torch_model("cnn_gru", ...)` returns a model that accepts `(batch, 91, n_features)` and returns `(batch, 5)`.
- CLI help exposes `--calendar` and `cnn_gru` as a model option.

The full verification command before claiming implementation complete is:

```bash
python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py tests/test_temporal_candidates.py tests/test_scripts.py tests.test_backtest tests.test_candidate_distribution tests.test_cnn_candidate -v
python -m compileall model scripts tests
```

## Open Decisions Resolved

- Calendar features and CNN-GRU are separate candidates first, not one combined model.
- Calendar features are opt-in via CLI flag.
- GRU is implemented before LSTM.
- No plain RNN is implemented.
- No Kaggle submission happens unless backtest and distribution gates pass.
