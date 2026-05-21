# Seed-42 Small-CNN Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test conservative seed-42 small-CNN improvements using temporal backtest and prediction-distribution checks before spending any Kaggle submission slot.

**Architecture:** Keep the current small 1D CNN as the anchor model and vary only low-risk training controls: dropout, weight decay, and scheduler. Add a standalone candidate-distribution comparison script so each generated candidate is evaluated against the current best submission before upload. Use temporal backtest as a rejection/filter metric and distribution similarity as the final safety check.

**Tech Stack:** Python, NumPy, pandas, PyTorch through the existing CNN utilities, `unittest`, existing `model/cnn_candidate.py`, existing `model/backtest.py`, existing submission utilities in `model/experiments.py`.

---

## File Structure

- Create: `scripts/compare_candidate_distribution.py`
  - Loads a candidate submission CSV and a reference submission CSV.
  - Validates identical columns, shape, and region order.
  - Computes all-prediction Pearson correlation, mean absolute difference, per-week means, per-week standard deviations, per-week min/max, and a conservative upload-safety flag.
- Create: `tests/test_candidate_distribution.py`
  - Unit tests for identical predictions, shifted predictions, ID-order mismatch, and column mismatch.
- Modify: `AGENTS.md`
  - Log seed calibration and any candidate backtest/distribution outcomes.
- No planned changes: `model/cnn_candidate.py`
  - Existing small CNN already supports seed, dropout, weight decay, scheduler, and 25-epoch training.
- No planned changes: `scripts/generate_cnn_submission.py`
  - Existing script already runs temporal backtest before final submission generation and supports `--seed`, `--dropout`, `--weight-decay`, `--scheduler`, and `--backtest-cutoffs`.

## Known Anchors

Use these known results to interpret new candidates. Lower MAE is better.

| Candidate | Public MAE | Temporal Backtest MAE | Notes |
|---|---:|---:|---|
| small CNN seed 42, 25 epochs | 0.8222 | 0.189607 | Current best public anchor. |
| small CNN seed 42, 40 epochs, scheduler | 0.8282 | 0.171465 | Backtest likes it, public score got worse. Treat lower backtest alone as insufficient. |
| small CNN seed 123, 30 epochs | 0.8512 | 0.199184 | Backtest correctly weaker. |
| tree/XGBoost-ish | 0.8434-ish | 0.300622 | Backtest correctly much weaker than CNN. |
| V2 CNN 5-epoch smoke | V2 public 0.8901-0.8967 | 0.205499 | Backtest not attractive; V2 remains unsafe. |

## Candidate Set

Keep seed fixed at `42`. Test only these conservative variants first.

| Candidate | Epochs | Dropout | Weight Decay | Scheduler | Purpose |
|---|---:|---:|---:|---|---|
| anchor | 25 | 0.15 | 0.001 | off | Current best setup. |
| reg_dropout20 | 25 | 0.20 | 0.001 | off | Slightly stronger dropout. |
| reg_wd0005 | 25 | 0.15 | 0.0005 | off | Slightly weaker weight decay. |
| scheduler25 | 25 | 0.15 | 0.001 | on | Scheduler without changing model/seed. |
| conservative_combo | 25 | 0.20 | 0.0005 | on | Only run if first three look promising. |

Do not add V2, GRU, 30 seeds, or feature changes in this plan.

### Task 1: Run Seed-42 Backtest Variants

**Files:**
- No file changes.

- [ ] **Step 1: Run dropout 0.20 variant backtest**

Run:

```bash
python scripts/run_temporal_backtest.py --mode cnn --model small --recent-cutoffs 2 --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.20 --weight-decay 0.001
```

Expected: terminal prints `cnn overall_mae <value>` and `saved output/backtests/temporal_backtest_<timestamp>.csv`.

- [ ] **Step 2: Run weight decay 0.0005 variant backtest**

Run:

```bash
python scripts/run_temporal_backtest.py --mode cnn --model small --recent-cutoffs 2 --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.0005
```

Expected: terminal prints `cnn overall_mae <value>` and `saved output/backtests/temporal_backtest_<timestamp>.csv`.

- [ ] **Step 3: Run scheduler variant backtest**

Run:

```bash
python scripts/run_temporal_backtest.py --mode cnn --model small --recent-cutoffs 2 --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.001 --scheduler
```

Expected: terminal prints `cnn overall_mae <value>` and `saved output/backtests/temporal_backtest_<timestamp>.csv`.

- [ ] **Step 4: Decide which variants deserve submission generation**

Use this rule:

```text
Reject immediately if backtest MAE is worse than 0.199184.
Treat 0.189607 to 0.199184 as acceptable but not exciting.
Treat lower than 0.189607 as promising, but remember 40 epochs overfit despite 0.171465.
```

Expected: choose zero, one, or two variants for Task 2. Do not generate submissions for all variants if backtest is weak.

### Task 2: Generate Full Submissions For Promising Variants

**Files:**
- Output only: `output/daily_candidates/*.csv`

- [ ] **Step 1: Generate dropout 0.20 submission if selected**

Run only if Task 1 selected `reg_dropout20`:

```bash
python scripts/generate_cnn_submission.py --model small --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.20 --weight-decay 0.001 --backtest-cutoffs 2
```

Expected: terminal prints `backtest_mae <value>`, `Saved submission: output/daily_candidates/cnn_1d_small_<timestamp>.csv`, and `Saved summary: output/daily_candidates/cnn_1d_small_<timestamp>_summary.csv`.

- [ ] **Step 2: Generate weight decay 0.0005 submission if selected**

Run only if Task 1 selected `reg_wd0005`:

```bash
python scripts/generate_cnn_submission.py --model small --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.0005 --backtest-cutoffs 2
```

Expected: terminal prints `backtest_mae <value>`, `Saved submission: output/daily_candidates/cnn_1d_small_<timestamp>.csv`, and `Saved summary: output/daily_candidates/cnn_1d_small_<timestamp>_summary.csv`.

- [ ] **Step 3: Generate scheduler submission if selected**

Run only if Task 1 selected `scheduler25`:

```bash
python scripts/generate_cnn_submission.py --model small --max-windows-per-region 52 --epochs 25 --seed 42 --dropout 0.15 --weight-decay 0.001 --scheduler --backtest-cutoffs 2
```

Expected: terminal prints `backtest_mae <value>`, `Saved submission: output/daily_candidates/cnn_1d_small_<timestamp>.csv`, and `Saved summary: output/daily_candidates/cnn_1d_small_<timestamp>_summary.csv`.

### Task 3: Add Candidate Distribution Comparison Script

**Files:**
- Create: `scripts/compare_candidate_distribution.py`
- Test: `tests/test_candidate_distribution.py`

- [ ] **Step 1: Write failing tests for distribution metrics and validation**

Create `tests/test_candidate_distribution.py` with this content:

```python
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts import compare_candidate_distribution as compare


def make_submission(path, region_ids=("R1", "R2"), value_offset=0.0):
    df = pd.DataFrame({
        "region_id": list(region_ids),
        "pred_week1": [1.0 + value_offset, 2.0 + value_offset],
        "pred_week2": [1.5 + value_offset, 2.5 + value_offset],
        "pred_week3": [2.0 + value_offset, 3.0 + value_offset],
        "pred_week4": [2.5 + value_offset, 3.5 + value_offset],
        "pred_week5": [3.0 + value_offset, 4.0 + value_offset],
    })
    df.to_csv(path, index=False)
    return df


class CandidateDistributionTests(unittest.TestCase):
    def test_identical_predictions_have_perfect_correlation_and_zero_diff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = f"{tmpdir}/reference.csv"
            candidate_path = f"{tmpdir}/candidate.csv"
            make_submission(reference_path)
            make_submission(candidate_path)

            result = compare.compare_candidate(candidate_path, reference_path)

        self.assertAlmostEqual(result["correlation"], 1.0)
        self.assertAlmostEqual(result["mean_abs_diff"], 0.0)
        self.assertTrue(result["safe"])

    def test_shifted_predictions_have_nonzero_diff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = f"{tmpdir}/reference.csv"
            candidate_path = f"{tmpdir}/candidate.csv"
            make_submission(reference_path)
            make_submission(candidate_path, value_offset=0.2)

            result = compare.compare_candidate(candidate_path, reference_path)

        self.assertAlmostEqual(result["mean_abs_diff"], 0.2)
        self.assertFalse(result["safe"])

    def test_mismatched_id_order_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = f"{tmpdir}/reference.csv"
            candidate_path = f"{tmpdir}/candidate.csv"
            make_submission(reference_path, region_ids=("R1", "R2"))
            make_submission(candidate_path, region_ids=("R2", "R1"))

            with self.assertRaisesRegex(ValueError, "ID order"):
                compare.compare_candidate(candidate_path, reference_path)

    def test_mismatched_columns_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = f"{tmpdir}/reference.csv"
            candidate_path = f"{tmpdir}/candidate.csv"
            reference = make_submission(reference_path)
            candidate = reference.drop(columns=["pred_week5"])
            candidate.to_csv(candidate_path, index=False)

            with self.assertRaisesRegex(ValueError, "Columns"):
                compare.compare_candidate(candidate_path, reference_path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail before implementation**

Run:

```bash
python -m unittest tests/test_candidate_distribution.py -v
```

Expected: FAIL with import error because `scripts/compare_candidate_distribution.py` does not exist yet.

- [ ] **Step 3: Implement distribution comparison script**

Create `scripts/compare_candidate_distribution.py` with this content:

```python
import argparse

import numpy as np
import pandas as pd


def _load_pair(candidate_path, reference_path):
    candidate = pd.read_csv(candidate_path)
    reference = pd.read_csv(reference_path)

    if list(candidate.columns) != list(reference.columns):
        raise ValueError("Columns do not match reference submission")
    if candidate.shape != reference.shape:
        raise ValueError("Shape does not match reference submission")
    if candidate.iloc[:, 0].tolist() != reference.iloc[:, 0].tolist():
        raise ValueError("ID order does not match reference submission")

    return candidate, reference


def _safe_corr(candidate_values, reference_values):
    cand_flat = candidate_values.reshape(-1)
    ref_flat = reference_values.reshape(-1)
    if np.std(cand_flat) == 0.0 or np.std(ref_flat) == 0.0:
        return 1.0 if np.allclose(cand_flat, ref_flat) else 0.0
    return float(np.corrcoef(cand_flat, ref_flat)[0, 1])


def compare_candidate(
    candidate_path,
    reference_path,
    min_corr=0.98,
    max_mean_abs_diff=0.10,
    max_week_mean_shift=0.08,
):
    candidate, reference = _load_pair(candidate_path, reference_path)
    target_cols = list(candidate.columns[1:])
    candidate_values = candidate[target_cols].to_numpy(dtype=float)
    reference_values = reference[target_cols].to_numpy(dtype=float)

    week_mean_shift = np.abs(candidate_values.mean(axis=0) - reference_values.mean(axis=0))
    result = {
        "correlation": _safe_corr(candidate_values, reference_values),
        "mean_abs_diff": float(np.mean(np.abs(candidate_values - reference_values))),
        "candidate_week_means": candidate_values.mean(axis=0).tolist(),
        "reference_week_means": reference_values.mean(axis=0).tolist(),
        "candidate_week_stds": candidate_values.std(axis=0).tolist(),
        "candidate_week_mins": candidate_values.min(axis=0).tolist(),
        "candidate_week_maxs": candidate_values.max(axis=0).tolist(),
        "max_week_mean_shift": float(week_mean_shift.max()),
        "min_prediction": float(candidate_values.min()),
        "max_prediction": float(candidate_values.max()),
    }
    result["safe"] = (
        result["correlation"] >= min_corr
        and result["mean_abs_diff"] <= max_mean_abs_diff
        and result["max_week_mean_shift"] <= max_week_mean_shift
        and result["min_prediction"] >= 0.0
        and result["max_prediction"] <= 5.0
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Compare a candidate submission against a reference submission.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--min-corr", type=float, default=0.98)
    parser.add_argument("--max-mean-abs-diff", type=float, default=0.10)
    parser.add_argument("--max-week-mean-shift", type=float, default=0.08)
    return parser.parse_args()


def main():
    args = parse_args()
    result = compare_candidate(
        args.candidate,
        args.reference,
        min_corr=args.min_corr,
        max_mean_abs_diff=args.max_mean_abs_diff,
        max_week_mean_shift=args.max_week_mean_shift,
    )

    print("correlation", round(result["correlation"], 6))
    print("mean_abs_diff", round(result["mean_abs_diff"], 6))
    print("max_week_mean_shift", round(result["max_week_mean_shift"], 6))
    print("candidate_week_means", [round(v, 6) for v in result["candidate_week_means"]])
    print("reference_week_means", [round(v, 6) for v in result["reference_week_means"]])
    print("candidate_week_stds", [round(v, 6) for v in result["candidate_week_stds"]])
    print("prediction_range", round(result["min_prediction"], 6), round(result["max_prediction"], 6))
    print("safe", result["safe"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run candidate distribution tests and verify they pass**

Run:

```bash
python -m unittest tests/test_candidate_distribution.py -v
```

Expected: `Ran 4 tests` and `OK`.

- [ ] **Step 5: Commit distribution comparison script**

Run:

```bash
git add scripts/compare_candidate_distribution.py tests/test_candidate_distribution.py
git commit -m "feat: add candidate distribution comparison"
```

Expected: commit succeeds.

### Task 4: Compare Promising Candidates Against Current Best

**Files:**
- Output only.

- [ ] **Step 1: Compare selected candidate against current best**

Replace `<candidate_path>` with the CSV generated in Task 2.

Run:

```bash
python scripts/compare_candidate_distribution.py --candidate <candidate_path> --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: output includes `correlation`, `mean_abs_diff`, `max_week_mean_shift`, `candidate_week_means`, `reference_week_means`, `candidate_week_stds`, `prediction_range`, and `safe`.

- [ ] **Step 2: Apply upload safety rule**

Use this rule:

```text
Uploadable only if safe=True, backtest MAE is competitive, and prediction range stays within [0, 5].
If safe=False, do not submit the raw candidate.
If backtest is strong but safe=False, consider a conservative blend in a separate plan.
```

Expected: one candidate is selected for possible Kaggle upload, or no candidate is submitted.

### Task 5: Log Results In AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Add seed-42 improvement experiment log**

Append a concise section under `## Temporal Backtest` or near `## Current State`:

```markdown
Seed-42 conservative CNN calibration:

| Candidate | Backtest MAE | Distribution check | Output | Decision |
|---|---:|---|---|---|
| dropout 0.20 | <value> | <safe/unsafe + key metrics> | `<path or not generated>` | <submit/reject/hold> |
| weight decay 0.0005 | <value> | <safe/unsafe + key metrics> | `<path or not generated>` | <submit/reject/hold> |
| scheduler 25 | <value> | <safe/unsafe + key metrics> | `<path or not generated>` | <submit/reject/hold> |
```

Replace each `<...>` with actual values from the run. If a candidate was not generated because backtest was weak, write `not generated`.

- [ ] **Step 2: Verify AGENTS.md diff is only tracking information**

Run:

```bash
git diff -- AGENTS.md
```

Expected: diff shows only the new seed-42 calibration log.

- [ ] **Step 3: Commit tracking update**

Run:

```bash
git add AGENTS.md
git commit -m "docs: log seed42 cnn calibration"
```

Expected: commit succeeds.

### Task 6: Full Verification

**Files:**
- All changed files.

- [ ] **Step 1: Run full test suite**

Run:

```bash
python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py tests/test_temporal_candidates.py tests/test_scripts.py tests.test_backtest tests/test_candidate_distribution.py -v
```

Expected: all tests pass with `OK`.

- [ ] **Step 2: Run compile check**

Run:

```bash
python -m compileall model scripts tests
```

Expected: command exits successfully with no syntax errors.

- [ ] **Step 3: Check repository status**

Run:

```bash
git status --short
```

Expected: no output, unless generated ignored files exist under `output/`.

### Task 7: Kaggle Submission Decision

**Files:**
- Modify only after submission: `AGENTS.md`
- Modify only after submission: `output/SUBMISSIONS.md`

- [ ] **Step 1: Decide whether to submit**

Use this final decision rule:

```text
Submit if one candidate has safe=True, backtest is competitive with seed42 anchor, and it is not a known overfit pattern.
Do not submit if all candidates are unsafe or backtest is worse than seed123's 0.199184.
```

- [ ] **Step 2: If submitted, update trackers after Kaggle returns public score**

Update `AGENTS.md` and `output/SUBMISSIONS.md` with:

```text
file path
candidate settings
backtest MAE
Kaggle public MAE
decision note
```

Expected: trackers contain enough information to explain why the candidate was submitted and how it performed.

## Self-Review Notes

- Spec coverage: The plan covers the approved seed-42-only improvement direction, conservative hyperparameter candidates, temporal backtest filtering, distribution comparison, tracking, and submission decision criteria.
- Placeholder scan: The only angle-bracket values appear in runtime commands or log templates where actual run output must be inserted after execution; no implementation step is unspecified.
- Type consistency: `compare_candidate(candidate_path, reference_path, ...)` returns a dictionary used consistently by tests and CLI output.
- Scope check: This plan excludes GRU, V2, 30-seed ensembles, and feature engineering so it remains a single focused experiment.
