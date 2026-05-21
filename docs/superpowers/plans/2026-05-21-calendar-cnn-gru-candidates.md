# Calendar CNN And CNN-GRU Candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two independent next candidate pipelines: Calendar-Small-CNN and CNN-GRU Hybrid, both using temporal backtest and distribution gates before submission.

**Architecture:** Extend the existing CNN candidate pipeline rather than creating new runners. Calendar features are deterministic opt-in sequence features generated from `date`; CNN-GRU is one new `model_name` in `model/cnn_candidate.py` that reuses the existing training, backtest, prediction, and submission code.

**Tech Stack:** Python, NumPy, pandas, PyTorch, unittest, existing `model/backtest.py`, existing `scripts/generate_cnn_submission.py`, existing distribution comparison script.

---

## File Structure

- Modify `model/cnn_candidate.py`: add calendar feature helper, add optional calendar support to train/test sequence builders, and add `cnn_gru` model architecture.
- Modify `model/backtest.py`: add optional calendar support to CNN backtest sample building.
- Modify `scripts/generate_cnn_submission.py`: expose `--calendar`, allow `--model cnn_gru`, pass calendar flag through backtest and final data building, and record it in the summary CSV.
- Modify `scripts/run_temporal_backtest.py`: expose `--calendar` for CNN mode and pass it into backtest.
- Modify `tests/test_scripts.py`: assert CLI help includes `--calendar` and `cnn_gru`.
- Create `tests/test_cnn_candidate.py`: cover calendar feature generation, sequence builder behavior, and CNN-GRU output shape.
- Modify `tests/test_backtest.py`: cover CNN backtest calendar plumbing using a patched sample builder.
- Modify `AGENTS.md`: document the new candidate commands and the gating rules after implementation is complete.

## Task 1: Calendar Feature Generation

**Files:**
- Modify: `model/cnn_candidate.py`
- Create: `tests/test_cnn_candidate.py`

- [ ] **Step 1: Write failing calendar feature tests**

Add this test file:

```python
import unittest

import numpy as np
import pandas as pd


class CnnCandidateTests(unittest.TestCase):
    def test_add_calendar_features_adds_deterministic_cyclical_columns(self):
        from model import cnn_candidate

        df = pd.DataFrame({
            "region_id": [1, 1],
            "date": ["2020-01-01", "2020-07-01"],
            "score": [0.0, 1.0],
            "rain": [2.0, 3.0],
        })

        out = cnn_candidate.add_calendar_features(df)

        self.assertIsNot(out, df)
        self.assertIn("calendar__doy_sin", out.columns)
        self.assertIn("calendar__doy_cos", out.columns)
        self.assertIn("calendar__month_sin", out.columns)
        self.assertIn("calendar__month_cos", out.columns)
        self.assertIn("calendar__week_sin", out.columns)
        self.assertIn("calendar__week_cos", out.columns)
        self.assertTrue(np.all(out["calendar__doy_sin"].between(-1.0, 1.0)))
        self.assertTrue(np.all(out["calendar__doy_cos"].between(-1.0, 1.0)))
        self.assertAlmostEqual(out.loc[0, "calendar__month_sin"], 0.5, places=6)
        self.assertAlmostEqual(out.loc[0, "calendar__month_cos"], np.sqrt(3) / 2, places=6)
        self.assertEqual(df.columns.tolist(), ["region_id", "date", "score", "rain"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_cnn_candidate -v
```

Expected: FAIL with an `AttributeError` because `add_calendar_features` does not exist yet.

- [ ] **Step 3: Implement calendar feature helper**

Add this function near the top of `model/cnn_candidate.py`, after imports:

```python
def add_calendar_features(df):
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    day_of_year = dates.dt.dayofyear.astype("float32")
    month = dates.dt.month.astype("float32")
    week = dates.dt.isocalendar().week.astype("float32")

    out["calendar__doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 366.0).astype("float32")
    out["calendar__doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 366.0).astype("float32")
    out["calendar__month_sin"] = np.sin(2.0 * np.pi * month / 12.0).astype("float32")
    out["calendar__month_cos"] = np.cos(2.0 * np.pi * month / 12.0).astype("float32")
    out["calendar__week_sin"] = np.sin(2.0 * np.pi * week / 53.0).astype("float32")
    out["calendar__week_cos"] = np.cos(2.0 * np.pi * week / 53.0).astype("float32")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest tests.test_cnn_candidate -v
```

Expected: PASS for the calendar feature test.

- [ ] **Step 5: Commit**

Run:

```bash
git add model/cnn_candidate.py tests/test_cnn_candidate.py
git commit -m "feat: add cnn calendar features"
```

## Task 2: Opt-In Calendar Features In Sequence Builders

**Files:**
- Modify: `model/cnn_candidate.py`
- Modify: `tests/test_cnn_candidate.py`

- [ ] **Step 1: Write failing sequence builder tests**

Append these tests to `CnnCandidateTests` in `tests/test_cnn_candidate.py`:

```python
    def test_sequence_train_builder_includes_calendar_features_only_when_requested(self):
        from model import cnn_candidate

        rows = []
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        for idx, date in enumerate(dates):
            rows.append({
                "region_id": 1,
                "date": date.strftime("%Y-%m-%d"),
                "score": float(idx) if idx >= 91 else np.nan,
                "rain": float(idx),
            })
        df = pd.DataFrame(rows)

        X_plain, y_plain, regions_plain, plain_cols = cnn_candidate.build_sequence_train_data_from_frame(
            df,
            max_windows_per_region=2,
        )
        X_calendar, y_calendar, regions_calendar, calendar_cols = cnn_candidate.build_sequence_train_data_from_frame(
            df,
            max_windows_per_region=2,
            include_calendar=True,
        )

        self.assertEqual(X_plain.shape[0], X_calendar.shape[0])
        self.assertEqual(y_plain.shape, y_calendar.shape)
        self.assertEqual(regions_plain, regions_calendar)
        self.assertEqual(plain_cols, ["rain"])
        self.assertIn("calendar__doy_sin", calendar_cols)
        self.assertEqual(X_calendar.shape[2], X_plain.shape[2] + 6)

    def test_sequence_test_builder_uses_calendar_feature_columns(self):
        from model import cnn_candidate

        rows = []
        dates = pd.date_range("2020-01-01", periods=91, freq="D")
        for idx, date in enumerate(dates):
            rows.append({
                "region_id": 1,
                "date": date.strftime("%Y-%m-%d"),
                "rain": float(idx),
            })
        df = pd.DataFrame(rows)
        df_with_calendar = cnn_candidate.add_calendar_features(df)
        feat_cols = ["rain", "calendar__doy_sin", "calendar__doy_cos"]

        X_test, regions = cnn_candidate.build_sequence_test_data_from_frame(df_with_calendar, feat_cols)

        self.assertEqual(regions, [1])
        self.assertEqual(X_test.shape, (1, 91, 3))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest tests.test_cnn_candidate -v
```

Expected: FAIL because `build_sequence_train_data_from_frame` does not accept `include_calendar`.

- [ ] **Step 3: Implement opt-in calendar support**

Change the function signature and first lines of `build_sequence_train_data_from_frame` in `model/cnn_candidate.py` to:

```python
def build_sequence_train_data_from_frame(df, max_windows_per_region=52, include_calendar=False):
    if include_calendar:
        df = add_calendar_features(df)
    meta_cols = ["region_id", "date", "score"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
```

No change is needed in `build_sequence_test_data_from_frame`; final test data should receive the same `feat_cols` returned by the training builder after the caller applies `add_calendar_features()` to the test frame.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m unittest tests.test_cnn_candidate -v
```

Expected: PASS for all tests in `tests.test_cnn_candidate`.

- [ ] **Step 5: Commit**

Run:

```bash
git add model/cnn_candidate.py tests/test_cnn_candidate.py
git commit -m "feat: support calendar cnn inputs"
```

## Task 3: Calendar Support In Backtest And Submission CLI

**Files:**
- Modify: `model/backtest.py`
- Modify: `scripts/generate_cnn_submission.py`
- Modify: `scripts/run_temporal_backtest.py`
- Modify: `tests/test_backtest.py`
- Modify: `tests/test_scripts.py`

- [ ] **Step 1: Write failing backtest plumbing test**

Append this test to `BacktestTests` in `tests/test_backtest.py`:

```python
    def test_evaluate_cnn_backtest_passes_calendar_flag_to_sample_builder(self):
        from unittest.mock import patch

        from model import backtest

        df = make_backtest_frame(n_regions=2, n_days=98, n_labels=7)
        calls = []
        real_builder = backtest.build_window_samples_from_frame

        def wrapped_builder(frame, window_days=91, include_calendar=False):
            calls.append(include_calendar)
            return real_builder(frame, window_days=window_days, include_calendar=include_calendar)

        with patch.object(backtest, "build_window_samples_from_frame", side_effect=wrapped_builder):
            with patch.object(backtest, "_fit_predict_cnn_split", return_value=np.zeros((2, 5))):
                summary = backtest.evaluate_cnn_backtest_from_frame(
                    df,
                    n_recent_cutoffs=1,
                    max_train_windows_per_region=2,
                    include_calendar=True,
                )

        self.assertEqual(calls, [True])
        self.assertEqual(summary["n_validation_rows"], 2)
```

- [ ] **Step 2: Write failing CLI help tests**

Update `test_cnn_script_help_imports_local_model_package` in `tests/test_scripts.py` by adding:

```python
        self.assertIn("--calendar", result.stdout)
        self.assertIn("cnn_gru", result.stdout)
```

Update `test_temporal_backtest_script_help_imports_local_model_package` by adding:

```python
        self.assertIn("--calendar", result.stdout)
```

- [ ] **Step 3: Run focused tests to verify they fail**

Run:

```bash
python -m unittest tests.test_backtest tests.test_scripts -v
```

Expected: FAIL because calendar arguments and `cnn_gru` are not wired yet.

- [ ] **Step 4: Add calendar support to backtest sample builder**

Change `build_window_samples_from_frame` in `model/backtest.py` to:

```python
def build_window_samples_from_frame(df, window_days=91, include_calendar=False):
    if include_calendar:
        df = cnn_candidate.add_calendar_features(df)
    meta_cols = ["region_id", "date", "score"]
    feat_cols = [col for col in df.columns if col not in meta_cols]
```

Change the `evaluate_cnn_backtest_from_frame` signature to include `include_calendar=False`, then change its sample builder call to:

```python
    samples, _ = build_window_samples_from_frame(df, include_calendar=include_calendar)
```

Do not pass calendar features into tree backtest; this plan only gates CNN candidates.

- [ ] **Step 5: Add calendar and cnn_gru CLI options**

In `scripts/generate_cnn_submission.py`, change model choices and add calendar:

```python
    parser.add_argument("--model", choices=["small", "v2", "cnn_gru"], default="small")
    parser.add_argument("--calendar", action="store_true")
```

Pass the flag into backtest:

```python
            include_calendar=args.calendar,
```

Pass the flag into final data building:

```python
    X_train, y_train, train_regions, feat_cols = cnn_candidate.build_sequence_train_data_from_frame(
        train_df,
        max_windows_per_region=args.max_windows_per_region,
        include_calendar=args.calendar,
    )
    if args.calendar:
        test_df = cnn_candidate.add_calendar_features(test_df)
    X_test, test_regions = cnn_candidate.build_sequence_test_data_from_frame(test_df, feat_cols)
```

Record it in the summary row:

```python
        "calendar": args.calendar,
```

In `scripts/run_temporal_backtest.py`, add:

```python
    parser.add_argument("--calendar", action="store_true")
```

Pass it into CNN mode:

```python
            include_calendar=args.calendar,
```

- [ ] **Step 6: Run focused tests to verify they pass**

Run:

```bash
python -m unittest tests.test_backtest tests.test_scripts -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add model/backtest.py scripts/generate_cnn_submission.py scripts/run_temporal_backtest.py tests/test_backtest.py tests/test_scripts.py
git commit -m "feat: add calendar cnn candidate flag"
```

## Task 4: CNN-GRU Model Option

**Files:**
- Modify: `model/cnn_candidate.py`
- Modify: `tests/test_cnn_candidate.py`

- [ ] **Step 1: Write failing CNN-GRU shape test**

Append this test to `CnnCandidateTests` in `tests/test_cnn_candidate.py`:

```python
    def test_cnn_gru_model_returns_five_week_predictions(self):
        from model import cnn_candidate

        if cnn_candidate.require_deep_learning_backend() != "torch":
            self.skipTest("PyTorch is required for CNN-GRU")

        import torch

        model = cnn_candidate.build_torch_model("cnn_gru", n_features=4, dropout=0.15)
        x = torch.zeros((3, 91, 4), dtype=torch.float32)

        with torch.no_grad():
            out = model(x)

        self.assertEqual(tuple(out.shape), (3, 5))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest tests.test_cnn_candidate -v
```

Expected: FAIL because `build_torch_model` rejects `cnn_gru`.

- [ ] **Step 3: Implement CNN-GRU architecture**

Add this class inside `build_torch_model` in `model/cnn_candidate.py`, after `V2Cnn`:

```python
    class CnnGru(torch.nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(32, 64, kernel_size=5, padding=2),
                torch.nn.ReLU(),
            )
            self.gru = torch.nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
            self.dropout = torch.nn.Dropout(dropout)
            self.head = torch.nn.Linear(64, 5)

        def forward(self, x):
            features = self.conv(x.transpose(1, 2)).transpose(1, 2)
            _, hidden = self.gru(features)
            return self.head(self.dropout(hidden[-1]))
```

Add the new model selection branch:

```python
    if model_name == "cnn_gru":
        return CnnGru(n_features)
```

Change the error message to:

```python
    raise ValueError("model_name must be 'small', 'v2', or 'cnn_gru'")
```

- [ ] **Step 4: Run focused tests to verify they pass**

Run:

```bash
python -m unittest tests.test_cnn_candidate tests.test_scripts -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add model/cnn_candidate.py tests/test_cnn_candidate.py
git commit -m "feat: add cnn gru candidate model"
```

## Task 5: Documentation And Candidate Commands

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add candidate commands to AGENTS.md**

Add a new subsection under `## Current State` or near the existing CNN guidance with this content:

### Next Candidate Commands

Calendar-Small-CNN candidate:

```bash
python scripts/generate_cnn_submission.py --model small --calendar --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.001
```

CNN-GRU candidate:

```bash
python scripts/generate_cnn_submission.py --model cnn_gru --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.001
```

Distribution gate after generation uses the timestamped candidate path printed by `generate_cnn_submission.py`. For example, after a Calendar-Small-CNN run:

```bash
python scripts/compare_candidate_distribution.py output/daily_candidates/cnn_1d_small_YYYYMMDD_HHMMSS.csv output/daily_candidates/cnn_1d_20260518_180837.csv
```

After a CNN-GRU run:

```bash
python scripts/compare_candidate_distribution.py output/daily_candidates/cnn_1d_cnn_gru_YYYYMMDD_HHMMSS.csv output/daily_candidates/cnn_1d_20260518_180837.csv
```

Submit at most one Calendar-Small-CNN and at most one CNN-GRU candidate. Save the remaining daily submission for a blend or combined follow-up after public feedback. Do not submit if temporal backtest and distribution checks disagree.

- [ ] **Step 2: Commit**

Run:

```bash
git add AGENTS.md
git commit -m "docs: add calendar and cnn gru candidate commands"
```

## Task 6: Verification Before Any Submission

**Files:**
- No source changes expected.

- [ ] **Step 1: Run full unit tests**

Run:

```bash
python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py tests/test_temporal_candidates.py tests/test_scripts.py tests.test_backtest tests.test_candidate_distribution tests.test_cnn_candidate -v
```

Expected: all tests PASS.

- [ ] **Step 2: Compile Python files**

Run:

```bash
python -m compileall model scripts tests
```

Expected: command exits 0 with no syntax errors.

- [ ] **Step 3: Run Calendar-Small-CNN candidate**

Run manually because this is a heavy GPU command:

```bash
python scripts/generate_cnn_submission.py --model small --calendar --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.001
```

Expected: command prints `backtest_mae`, `Saved submission: output/daily_candidates/cnn_1d_small_YYYYMMDD_HHMMSS.csv`, and `Saved summary: ...`. Record the path and scores in `AGENTS.md`.

- [ ] **Step 4: Run Calendar-Small-CNN distribution check**

Run with the generated candidate path from Step 3:

```bash
python scripts/compare_candidate_distribution.py output/daily_candidates/cnn_1d_small_YYYYMMDD_HHMMSS.csv output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: `safe True` before considering a submission. If `safe False`, do not submit Calendar-Small-CNN.

- [ ] **Step 5: Run CNN-GRU candidate**

Run manually because this is a heavy GPU command:

```bash
python scripts/generate_cnn_submission.py --model cnn_gru --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.001
```

Expected: command prints `backtest_mae`, `Saved submission: output/daily_candidates/cnn_1d_cnn_gru_YYYYMMDD_HHMMSS.csv`, and `Saved summary: ...`. Record the path and scores in `AGENTS.md`.

- [ ] **Step 6: Run CNN-GRU distribution check**

Run with the generated candidate path from Step 5:

```bash
python scripts/compare_candidate_distribution.py output/daily_candidates/cnn_1d_cnn_gru_YYYYMMDD_HHMMSS.csv output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: `safe True` before considering a submission. If `safe False`, do not submit CNN-GRU.

- [ ] **Step 7: Submission decision**

Use these rules:

- If neither candidate passes distribution safety, submit nothing.
- If only one candidate passes distribution safety and has acceptable backtest, submit that one.
- If both pass, submit Calendar-Small-CNN first unless CNN-GRU has a clearly stronger backtest improvement and still passes distribution safety.
- Save the final daily submission for a blend or combined calendar-plus-GRU follow-up after seeing public feedback.

- [ ] **Step 8: Commit experiment log updates**

After recording generated candidate paths, backtest scores, distribution metrics, and submission decision in `AGENTS.md`, run:

```bash
git add AGENTS.md
git commit -m "docs: log calendar and cnn gru candidate results"
```

## Self-Review Notes

- Spec coverage: Calendar features, CNN-GRU architecture, CLI flags, backtest parity, distribution gates, docs, and verification are each covered by tasks.
- Placeholder scan: Commands use timestamp examples for generated output paths because the exact filename is created at runtime; implementation files should not keep angle-bracket placeholders.
- Type consistency: `include_calendar` is used consistently in data builders and backtest, while the CLI flag is named `--calendar` and maps to `args.calendar`.
