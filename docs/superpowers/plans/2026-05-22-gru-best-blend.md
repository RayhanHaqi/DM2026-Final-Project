# GRU Best-Blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate conservative blends between the CNN-GRU candidate and the current best submission.

**Architecture:** Add one focused CPU-only blending script that validates two submission CSVs, blends prediction columns by a user-supplied candidate weight, clips to `[0, 5]`, and writes a timestamped CSV. Existing distribution-comparison tooling remains the gate before any submission.

**Tech Stack:** Python, NumPy, pandas, unittest, existing `model.experiments` utilities, existing `scripts/compare_candidate_distribution.py`.

---

## File Structure

- Create `scripts/blend_submissions.py`: CLI for blending two submission CSVs.
- Create `tests/test_blend_submissions.py`: unit tests for validation, weighted blend, clipping, and output path behavior.
- Modify `tests/test_scripts.py`: assert the new script imports and exposes help.
- Modify `AGENTS.md`: log blend candidate commands and final experiment results.

## Task 1: Blend Script Tests

**Files:**
- Create: `tests/test_blend_submissions.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_blend_submissions.py`:

```python
import tempfile
import unittest

import pandas as pd

from scripts import blend_submissions


def make_submission(path, region_ids=("R1", "R2"), offset=0.0):
    df = pd.DataFrame({
        "region_id": list(region_ids),
        "pred_week1": [1.0 + offset, 3.0 + offset],
        "pred_week2": [2.0 + offset, 4.0 + offset],
    })
    df.to_csv(path, index=False)
    return df


class BlendSubmissionTests(unittest.TestCase):
    def test_blend_submissions_uses_candidate_weight_and_reference_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = f"{tmpdir}/candidate.csv"
            reference_path = f"{tmpdir}/reference.csv"
            make_submission(candidate_path, offset=2.0)
            reference = make_submission(reference_path, offset=0.0)

            blended = blend_submissions.blend_submission_frames(candidate_path, reference_path, candidate_weight=0.25)

        self.assertEqual(blended["region_id"].tolist(), reference["region_id"].tolist())
        self.assertAlmostEqual(blended.loc[0, "pred_week1"], 1.5)
        self.assertAlmostEqual(blended.loc[1, "pred_week2"], 4.5)

    def test_blend_submissions_clips_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = f"{tmpdir}/candidate.csv"
            reference_path = f"{tmpdir}/reference.csv"
            pd.DataFrame({"region_id": ["R1"], "pred_week1": [9.0], "pred_week2": [-3.0]}).to_csv(candidate_path, index=False)
            pd.DataFrame({"region_id": ["R1"], "pred_week1": [9.0], "pred_week2": [-3.0]}).to_csv(reference_path, index=False)

            blended = blend_submissions.blend_submission_frames(candidate_path, reference_path, candidate_weight=0.5)

        self.assertEqual(blended.loc[0, "pred_week1"], 5.0)
        self.assertEqual(blended.loc[0, "pred_week2"], 0.0)

    def test_blend_submissions_rejects_mismatched_id_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = f"{tmpdir}/candidate.csv"
            reference_path = f"{tmpdir}/reference.csv"
            make_submission(candidate_path, region_ids=("R2", "R1"))
            make_submission(reference_path, region_ids=("R1", "R2"))

            with self.assertRaisesRegex(ValueError, "ID order"):
                blend_submissions.blend_submission_frames(candidate_path, reference_path, candidate_weight=0.25)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
python -m unittest tests.test_blend_submissions -v
```

Expected: FAIL because `scripts.blend_submissions` does not exist.

## Task 2: Blend Script Implementation

**Files:**
- Create: `scripts/blend_submissions.py`

- [ ] **Step 1: Implement script**

Create `scripts/blend_submissions.py`:

```python
import argparse
import os
from datetime import datetime

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


def blend_submission_frames(candidate_path, reference_path, candidate_weight):
    if not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("candidate_weight must be between 0 and 1")
    candidate, reference = _load_pair(candidate_path, reference_path)
    target_cols = list(reference.columns[1:])
    candidate_values = candidate[target_cols].to_numpy(dtype=float)
    reference_values = reference[target_cols].to_numpy(dtype=float)
    blended_values = candidate_weight * candidate_values + (1.0 - candidate_weight) * reference_values
    blended = reference.copy()
    blended[target_cols] = np.clip(blended_values, 0.0, 5.0)
    return blended


def parse_args():
    parser = argparse.ArgumentParser(description="Blend two submission CSVs with a candidate weight.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate-weight", type=float, required=True)
    parser.add_argument("--output-dir", default="output/daily_candidates")
    parser.add_argument("--name", default="cnn_gru_blend")
    return parser.parse_args()


def main():
    args = parse_args()
    blended = blend_submission_frames(args.candidate, args.reference, args.candidate_weight)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_tag = f"w{int(round(args.candidate_weight * 100)):02d}"
    out_path = os.path.join(args.output_dir, f"{args.name}_{weight_tag}_{timestamp}.csv")
    blended.to_csv(out_path, index=False)
    print(f"Saved blend: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify unit tests pass**

Run:

```bash
python -m unittest tests.test_blend_submissions -v
```

Expected: PASS.

## Task 3: Script Help Test

**Files:**
- Modify: `tests/test_scripts.py`

- [ ] **Step 1: Add failing help test**

Add this test method to `ScriptTests`:

```python
    def test_blend_script_help_imports(self):
        result = subprocess.run(
            [sys.executable, "scripts/blend_submissions.py", "--help"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Blend two submission CSVs", result.stdout)
        self.assertIn("--candidate-weight", result.stdout)
```

- [ ] **Step 2: Verify focused tests pass**

Run:

```bash
python -m unittest tests.test_scripts tests.test_blend_submissions -v
```

Expected: PASS.

- [ ] **Step 3: Commit script and tests**

Run:

```bash
git add scripts/blend_submissions.py tests/test_blend_submissions.py tests/test_scripts.py
git commit -m "feat: add submission blend script"
```

## Task 4: Generate And Gate GRU Blends

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Generate three blends**

Run:

```bash
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv --candidate-weight 0.10
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv --candidate-weight 0.15
python scripts/blend_submissions.py --candidate output/daily_candidates/cnn_1d_cnn_gru_20260522_154103.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv --candidate-weight 0.20
```

Expected: three `output/daily_candidates/cnn_gru_blend_wXX_YYYYMMDD_HHMMSS.csv` files.

- [ ] **Step 2: Run distribution checks**

Run once per generated path:

```bash
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w10_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w15_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
python scripts/compare_candidate_distribution.py --candidate output/daily_candidates/cnn_gru_blend_w20_YYYYMMDD_HHMMSS.csv --reference output/daily_candidates/cnn_1d_20260518_180837.csv
```

Expected: use the timestamped paths from Step 1. Submit only if `safe True`.

- [ ] **Step 3: Record decision in AGENTS.md**

Add a row or subsection with each blend path, candidate weight, distribution metrics, and decision. Commit:

```bash
git add AGENTS.md
git commit -m "docs: log gru blend candidate results"
```

## Task 5: Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Run full tests**

Run:

```bash
python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py tests/test_temporal_candidates.py tests/test_scripts.py tests.test_backtest tests.test_candidate_distribution tests.test_cnn_candidate tests.test_blend_submissions -v
```

Expected: all tests PASS.

- [ ] **Step 2: Compile Python files**

Run:

```bash
python -m compileall model scripts tests
```

Expected: no syntax errors.

## Self-Review Notes

- Spec coverage: blend script, validation, clipping, candidate weights, distribution gate, and logging are covered.
- Placeholder scan: timestamp examples use `YYYYMMDD_HHMMSS` because filenames are created at runtime; exact generated paths must be copied from script output.
- Type consistency: CLI uses `--candidate-weight`; Python function uses `candidate_weight`.
