# Temporal Backtest Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validation-only temporal backtest for both tree and CNN pipelines that ranks candidates more like Kaggle public MAE than the current region-based validation.

**Architecture:** Add one shared `model/backtest.py` module that builds terminal-style historical horizons and evaluates either aggregate-feature tree models or raw-sequence CNN models against recent cutoffs. Expose the evaluator through one CLI script and cover it with small synthetic `unittest` fixtures so the logic can be verified without large dataset runs.

**Tech Stack:** Python, NumPy, pandas, scikit-learn, PyTorch via the existing CNN utilities, XGBoost via the existing tree trainer, `unittest`.

---

## File Structure

- Create: `model/backtest.py`
  - Shared temporal sample builder.
  - Recent-cutoff split builder.
  - Tree backtest evaluator.
  - CNN backtest evaluator with train-only normalization.
  - Summary helpers.
- Create: `tests/test_backtest.py`
  - Synthetic fixtures for sample building, split selection, tree summary shape, and train-only CNN normalization.
- Create: `scripts/run_temporal_backtest.py`
  - CLI entry point for tree, CNN, or both validation modes.
- Modify: `tests/test_scripts.py:7-40`
  - Add `--help` import test for the new backtest script.

### Task 1: Build Shared Temporal Samples And Cutoff Splits

**Files:**
- Create: `model/backtest.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing split-builder tests**

Create `tests/test_backtest.py` with this initial content:

```python
import unittest

import numpy as np
import pandas as pd


def make_backtest_frame(n_regions=2, n_days=98, n_labels=7):
    rows = []
    for region_idx in range(n_regions):
        region_id = f"R{region_idx + 1}"
        for day in range(n_days):
            score = float(day) if day >= n_days - n_labels else np.nan
            rows.append({
                "region_id": region_id,
                "date": f"2026-01-{(day % 28) + 1:02d}",
                "score": score,
                "rain": float(day + region_idx),
                "temp": float(day * 2 + region_idx),
            })
    return pd.DataFrame(rows)


class BacktestSplitTests(unittest.TestCase):
    def test_build_window_samples_extracts_targets_and_score_index(self):
        from model import backtest

        df = make_backtest_frame(n_regions=1, n_days=98, n_labels=7)
        samples, feat_cols = backtest.build_window_samples_from_frame(df, window_days=91)

        self.assertEqual(feat_cols, ["rain", "temp"])
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]["score_idx_start"], 0)
        np.testing.assert_allclose(samples[0]["target"], [91.0, 92.0, 93.0, 94.0, 95.0])
        np.testing.assert_allclose(samples[-1]["target"], [93.0, 94.0, 95.0, 96.0, 97.0])
        self.assertEqual(samples[0]["window"].shape, (91, 2))

    def test_build_recent_backtest_splits_uses_terminal_horizons(self):
        from model import backtest

        df = make_backtest_frame(n_regions=2, n_days=98, n_labels=7)
        samples, _ = backtest.build_window_samples_from_frame(df, window_days=91)
        splits = backtest.build_recent_backtest_splits(samples, n_recent_cutoffs=2)

        self.assertEqual(len(splits), 2)
        self.assertEqual({sample["score_idx_start"] for sample in splits[0]["val_samples"]}, {2})
        self.assertEqual({sample["score_idx_start"] for sample in splits[1]["val_samples"]}, {1})
        self.assertEqual({sample["score_idx_start"] for sample in splits[0]["train_samples"]}, {0, 1})
        self.assertEqual({sample["score_idx_start"] for sample in splits[1]["train_samples"]}, {0})

    def test_build_recent_backtest_splits_respects_max_train_windows_per_region(self):
        from model import backtest

        df = make_backtest_frame(n_regions=2, n_days=99, n_labels=8)
        samples, _ = backtest.build_window_samples_from_frame(df, window_days=91)
        splits = backtest.build_recent_backtest_splits(
            samples,
            n_recent_cutoffs=2,
            max_train_windows_per_region=1,
        )

        self.assertEqual(len(splits[0]["train_samples"]), 2)
        self.assertEqual({sample["score_idx_start"] for sample in splits[0]["train_samples"]}, {2})
        self.assertEqual(len(splits[1]["train_samples"]), 2)
        self.assertEqual({sample["score_idx_start"] for sample in splits[1]["train_samples"]}, {1})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new split tests and confirm they fail**

Run:

```bash
python -m unittest tests.test_backtest.BacktestSplitTests -v
```

Expected: FAIL with `ImportError` because `model/backtest.py` does not exist yet.

- [ ] **Step 3: Write the minimal shared sample and split builder**

Create `model/backtest.py` with this initial implementation:

```python
from collections import defaultdict

import numpy as np
import pandas as pd


def build_window_samples_from_frame(df, window_days=91):
    meta_cols = ["region_id", "date", "score"]
    feat_cols = [col for col in df.columns if col not in meta_cols]
    score_vals = df["score"].to_numpy()
    samples = []

    for region_id, grp in df.groupby("region_id", sort=False):
        indices = grp.index.to_numpy()
        score_positions = np.where(pd.notna(score_vals[indices]))[0]

        for start in range(0, len(score_positions) - 4):
            label_pos = score_positions[start:start + 5]
            first_label_pos = label_pos[0]
            if first_label_pos < window_days:
                continue

            window_indices = indices[first_label_pos - window_days:first_label_pos]
            samples.append({
                "region_id": region_id,
                "score_idx_start": start,
                "window": df.iloc[window_indices][feat_cols].to_numpy(dtype=float),
                "target": score_vals[indices[label_pos]].astype(float),
            })

    return samples, feat_cols


def build_recent_backtest_splits(samples, n_recent_cutoffs=3, max_train_windows_per_region=None):
    by_region = defaultdict(list)
    for sample in samples:
        by_region[sample["region_id"]].append(sample)

    ordered = {}
    for region_id, region_samples in by_region.items():
        ordered[region_id] = sorted(region_samples, key=lambda sample: sample["score_idx_start"])

    min_windows = min(len(region_samples) for region_samples in ordered.values())
    if n_recent_cutoffs >= min_windows:
        raise ValueError("Not enough horizons for requested recent cutoffs")

    splits = []
    for offset in range(1, n_recent_cutoffs + 1):
        train_samples = []
        val_samples = []

        for region_samples in ordered.values():
            val_index = len(region_samples) - offset
            val_samples.append(region_samples[val_index])

            region_train = region_samples[:val_index]
            if max_train_windows_per_region is not None:
                region_train = region_train[-max_train_windows_per_region:]
            train_samples.extend(region_train)

        splits.append({
            "cutoff_offset": offset,
            "train_samples": train_samples,
            "val_samples": val_samples,
        })

    return splits
```

- [ ] **Step 4: Run the split tests and confirm they pass**

Run:

```bash
python -m unittest tests.test_backtest.BacktestSplitTests -v
```

Expected: `Ran 3 tests` and `OK`.

- [ ] **Step 5: Commit the split builder**

Run:

```bash
git add model/backtest.py tests/test_backtest.py
git commit -m "feat: add temporal backtest split builder"
```

Expected: commit succeeds.

### Task 2: Add Tree Backtest Evaluation And Summary Reporting

**Files:**
- Modify: `model/backtest.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Add failing tree-evaluator tests**

Append this to `tests/test_backtest.py`:

```python
from unittest.mock import patch


class FakeTreeModel:
    def __init__(self, prediction):
        self.prediction = np.array(prediction, dtype=float)

    def predict(self, X):
        return np.repeat(self.prediction[None, :], len(X), axis=0)


class BacktestTreeTests(unittest.TestCase):
    def test_evaluate_tree_backtest_returns_expected_summary_shapes(self):
        from model import backtest

        df = make_backtest_frame(n_regions=2, n_days=98, n_labels=7)

        with patch.object(backtest.train, "train_xgboost", return_value=FakeTreeModel([0, 0, 0, 0, 0])):
            summary = backtest.evaluate_tree_backtest_from_frame(
                df,
                n_recent_cutoffs=2,
                max_train_windows_per_region=2,
                params_override={"n_estimators": 10},
            )

        self.assertIn("overall_mae", summary)
        self.assertEqual(len(summary["per_week_mae"]), 5)
        self.assertEqual(len(summary["per_cutoff_mae"]), 2)
        self.assertEqual(summary["n_validation_rows"], 4)
```

- [ ] **Step 2: Run the tree test and confirm it fails**

Run:

```bash
python -m unittest tests.test_backtest.BacktestTreeTests -v
```

Expected: FAIL with `AttributeError` because `evaluate_tree_backtest_from_frame` is not implemented yet.

- [ ] **Step 3: Implement tree feature aggregation and summary helpers**

Update `model/backtest.py` to include these additions:

```python
from sklearn.metrics import mean_absolute_error

from model import train, utils


def summarize_predictions(split_rows):
    y_true = np.vstack([row["y_true"] for row in split_rows])
    preds = np.vstack([row["preds"] for row in split_rows])
    return {
        "overall_mae": float(mean_absolute_error(y_true, preds)),
        "per_week_mae": [float(mean_absolute_error(y_true[:, idx], preds[:, idx])) for idx in range(y_true.shape[1])],
        "per_cutoff_mae": [
            {
                "cutoff_offset": cutoff_offset,
                "mae": float(mean_absolute_error(
                    np.vstack([row["y_true"] for row in split_rows if row["cutoff_offset"] == cutoff_offset]),
                    np.vstack([row["preds"] for row in split_rows if row["cutoff_offset"] == cutoff_offset]),
                )),
            }
            for cutoff_offset in sorted({row["cutoff_offset"] for row in split_rows})
        ],
        "prediction_means": preds.mean(axis=0).tolist(),
        "target_means": y_true.mean(axis=0).tolist(),
        "n_validation_rows": int(len(split_rows)),
    }


def _aggregate_tree_matrix(samples, feat_cols):
    rows = [utils._aggregate_array(sample["window"], feat_cols) for sample in samples]
    return pd.DataFrame(rows)


def evaluate_tree_backtest_from_frame(
    df,
    n_recent_cutoffs=3,
    max_train_windows_per_region=52,
    params_override=None,
):
    samples, feat_cols = build_window_samples_from_frame(df)
    splits = build_recent_backtest_splits(
        samples,
        n_recent_cutoffs=n_recent_cutoffs,
        max_train_windows_per_region=max_train_windows_per_region,
    )
    split_rows = []

    for split in splits:
        X_tr = _aggregate_tree_matrix(split["train_samples"], feat_cols)
        y_tr = np.vstack([sample["target"] for sample in split["train_samples"]])
        X_val = _aggregate_tree_matrix(split["val_samples"], feat_cols)
        y_val = np.vstack([sample["target"] for sample in split["val_samples"]])

        model = train.train_xgboost(X_tr, y_tr, params_override=params_override)
        preds = np.clip(model.predict(X_val), 0.0, 5.0)

        for idx, sample in enumerate(split["val_samples"]):
            split_rows.append({
                "region_id": sample["region_id"],
                "cutoff_offset": split["cutoff_offset"],
                "y_true": y_val[idx],
                "preds": preds[idx],
            })

    return summarize_predictions(split_rows)
```

- [ ] **Step 4: Run the tree test and confirm it passes**

Run:

```bash
python -m unittest tests.test_backtest.BacktestTreeTests -v
```

Expected: `Ran 1 test` and `OK`.

- [ ] **Step 5: Commit the tree evaluator**

Run:

```bash
git add model/backtest.py tests/test_backtest.py
git commit -m "feat: add tree temporal backtest"
```

Expected: commit succeeds.

### Task 3: Add CNN Backtest Evaluation With Train-Only Normalization

**Files:**
- Modify: `model/backtest.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Add failing CNN normalization and evaluator tests**

Append this to `tests/test_backtest.py`:

```python
class BacktestCnnTests(unittest.TestCase):
    def test_standardize_from_train_uses_train_statistics_only(self):
        from model import backtest

        X_train = np.array([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=float)
        X_val = np.array([[[101.0], [103.0]]], dtype=float)

        X_tr_std, X_val_std, mean, std = backtest.standardize_from_train(X_train, X_val)

        self.assertAlmostEqual(float(mean.reshape(-1)[0]), 4.0)
        self.assertAlmostEqual(float(std.reshape(-1)[0]), np.std(X_train), places=6)
        self.assertAlmostEqual(float(X_tr_std.mean()), 0.0, places=6)
        self.assertGreater(float(X_val_std.mean()), 10.0)

    def test_evaluate_cnn_backtest_returns_expected_summary_shapes(self):
        from model import backtest

        df = make_backtest_frame(n_regions=2, n_days=98, n_labels=7)

        with patch.object(backtest, "_fit_predict_cnn_split", return_value=np.zeros((2, 5))):
            summary = backtest.evaluate_cnn_backtest_from_frame(
                df,
                n_recent_cutoffs=2,
                max_train_windows_per_region=2,
                model_name="small",
                epochs=2,
                batch_size=8,
            )

        self.assertIn("overall_mae", summary)
        self.assertEqual(len(summary["per_week_mae"]), 5)
        self.assertEqual(len(summary["per_cutoff_mae"]), 2)
        self.assertEqual(summary["n_validation_rows"], 4)
```

- [ ] **Step 2: Run the CNN tests and confirm they fail**

Run:

```bash
python -m unittest tests.test_backtest.BacktestCnnTests -v
```

Expected: FAIL with `AttributeError` because `standardize_from_train` and `evaluate_cnn_backtest_from_frame` are not implemented yet.

- [ ] **Step 3: Implement train-only normalization and CNN split evaluation**

Update `model/backtest.py` with these additions:

```python
from model import cnn_candidate


def standardize_from_train(X_train, X_val):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (X_train - mean) / std, (X_val - mean) / std, mean, std


def _fit_predict_cnn_split(
    X_train,
    y_train,
    X_val,
    model_name="small",
    epochs=10,
    batch_size=256,
    lr=1e-3,
    seed=42,
    dropout=0.15,
    weight_decay=1e-3,
    scheduler=False,
):
    backend = cnn_candidate.require_deep_learning_backend()
    if backend != "torch":
        raise RuntimeError("This CNN backtest currently supports PyTorch only.")

    import torch

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn_candidate.build_torch_model(model_name, X_train.shape[2], dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2) if scheduler else None
    loss_fn = torch.nn.L1Loss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_va = torch.tensor(X_val, dtype=torch.float32).to(device)

    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X_tr))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            xb = X_tr[batch_idx].to(device)
            yb = y_tr[batch_idx].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        if lr_scheduler is not None:
            lr_scheduler.step(float(np.mean(epoch_losses)))

    model.eval()
    with torch.no_grad():
        preds = model(X_va).cpu().numpy()
    return np.clip(preds, 0.0, 5.0)


def evaluate_cnn_backtest_from_frame(
    df,
    n_recent_cutoffs=3,
    max_train_windows_per_region=52,
    model_name="small",
    epochs=10,
    batch_size=256,
    lr=1e-3,
    seed=42,
    dropout=0.15,
    weight_decay=1e-3,
    scheduler=False,
):
    samples, _ = build_window_samples_from_frame(df)
    splits = build_recent_backtest_splits(
        samples,
        n_recent_cutoffs=n_recent_cutoffs,
        max_train_windows_per_region=max_train_windows_per_region,
    )
    split_rows = []

    for split in splits:
        X_tr = np.stack([sample["window"] for sample in split["train_samples"]]).astype("float32")
        y_tr = np.stack([sample["target"] for sample in split["train_samples"]]).astype("float32")
        X_val = np.stack([sample["window"] for sample in split["val_samples"]]).astype("float32")
        y_val = np.stack([sample["target"] for sample in split["val_samples"]]).astype("float32")

        X_tr_std, X_val_std, _, _ = standardize_from_train(X_tr, X_val)
        preds = _fit_predict_cnn_split(
            X_tr_std,
            y_tr,
            X_val_std,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            dropout=dropout,
            weight_decay=weight_decay,
            scheduler=scheduler,
        )

        for idx, sample in enumerate(split["val_samples"]):
            split_rows.append({
                "region_id": sample["region_id"],
                "cutoff_offset": split["cutoff_offset"],
                "y_true": y_val[idx],
                "preds": preds[idx],
            })

    return summarize_predictions(split_rows)
```

- [ ] **Step 4: Run the CNN tests and confirm they pass**

Run:

```bash
python -m unittest tests.test_backtest.BacktestCnnTests -v
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 5: Commit the CNN evaluator**

Run:

```bash
git add model/backtest.py tests/test_backtest.py
git commit -m "feat: add cnn temporal backtest"
```

Expected: commit succeeds.

### Task 4: Add The CLI Entry Point And Script Help Coverage

**Files:**
- Create: `scripts/run_temporal_backtest.py`
- Modify: `tests/test_scripts.py:7-40`

- [ ] **Step 1: Add the failing script help test**

Insert this test into `tests/test_scripts.py` above the `if __name__ == "__main__":` block:

```python
    def test_temporal_backtest_script_help_imports_local_model_package(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_temporal_backtest.py", "--help"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Run temporal backtest validation", result.stdout)
        self.assertIn("--mode", result.stdout)
        self.assertIn("--recent-cutoffs", result.stdout)
        self.assertIn("--epochs", result.stdout)
```

- [ ] **Step 2: Run the new script test and confirm it fails**

Run:

```bash
python -m unittest tests.test_scripts.ScriptTests.test_temporal_backtest_script_help_imports_local_model_package -v
```

Expected: FAIL because `scripts/run_temporal_backtest.py` does not exist yet.

- [ ] **Step 3: Create the temporal backtest CLI script**

Create `scripts/run_temporal_backtest.py` with this content:

```python
import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import backtest


def parse_args():
    parser = argparse.ArgumentParser(description="Run temporal backtest validation.")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--mode", choices=["tree", "cnn", "both"], default="both")
    parser.add_argument("--recent-cutoffs", type=int, default=3)
    parser.add_argument("--max-windows-per-region", type=int, default=52)
    parser.add_argument("--output-dir", default="output/backtests")
    parser.add_argument("--model", choices=["small", "v2"], default="small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", action="store_true")
    return parser.parse_args()


def flatten_summary(name, summary):
    row = {
        "name": name,
        "overall_mae": summary["overall_mae"],
        "n_validation_rows": summary["n_validation_rows"],
    }
    for idx, value in enumerate(summary["per_week_mae"], start=1):
        row[f"week{idx}_mae"] = value
    for idx, value in enumerate(summary["prediction_means"], start=1):
        row[f"week{idx}_pred_mean"] = value
    for idx, value in enumerate(summary["target_means"], start=1):
        row[f"week{idx}_target_mean"] = value
    return row


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.read_csv(args.train)
    rows = []

    if args.mode in {"tree", "both"}:
        tree_summary = backtest.evaluate_tree_backtest_from_frame(
            train_df,
            n_recent_cutoffs=args.recent_cutoffs,
            max_train_windows_per_region=args.max_windows_per_region,
        )
        rows.append(flatten_summary("tree", tree_summary))
        print("tree overall_mae", round(tree_summary["overall_mae"], 6))

    if args.mode in {"cnn", "both"}:
        cnn_summary = backtest.evaluate_cnn_backtest_from_frame(
            train_df,
            n_recent_cutoffs=args.recent_cutoffs,
            max_train_windows_per_region=args.max_windows_per_region,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
        )
        rows.append(flatten_summary(f"cnn_{args.model}", cnn_summary))
        print("cnn overall_mae", round(cnn_summary["overall_mae"], 6))

    out_path = os.path.join(args.output_dir, f"temporal_backtest_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("saved", out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the new script help test and confirm it passes**

Run:

```bash
python -m unittest tests.test_scripts.ScriptTests.test_temporal_backtest_script_help_imports_local_model_package -v
```

Expected: `Ran 1 test` and `OK`.

- [ ] **Step 5: Commit the CLI and help coverage**

Run:

```bash
git add scripts/run_temporal_backtest.py tests/test_scripts.py
git commit -m "feat: add temporal backtest cli"
```

Expected: commit succeeds.

### Task 5: Full Verification And Implementation Handoff

**Files:**
- Create: `model/backtest.py`
- Create: `tests/test_backtest.py`
- Create: `scripts/run_temporal_backtest.py`
- Modify: `tests/test_scripts.py:7-40`

- [ ] **Step 1: Run focused verification**

Run:

```bash
python -m unittest tests.test_backtest tests.test_scripts -v
```

Expected: all backtest and script tests pass.

- [ ] **Step 2: Run project regression checks**

Run:

```bash
python -m unittest tests/test_progress.py tests/test_submission.py tests/test_experiments.py tests/test_temporal_candidates.py tests/test_scripts.py tests.test_backtest -v
```

Expected: full suite passes with no new failures.

- [ ] **Step 3: Smoke-test the new CLI in tree mode**

Run:

```bash
python scripts/run_temporal_backtest.py --mode tree --recent-cutoffs 2 --max-windows-per-region 52
```

Expected: prints `tree overall_mae ...` and `saved output/backtests/temporal_backtest_<timestamp>.csv`.

- [ ] **Step 4: Smoke-test the new CLI in CNN mode with low-cost settings**

Run:

```bash
python scripts/run_temporal_backtest.py --mode cnn --model small --recent-cutoffs 1 --max-windows-per-region 52 --epochs 2 --batch-size 256
```

Expected: prints `cnn overall_mae ...` and `saved output/backtests/temporal_backtest_<timestamp>.csv`.

- [ ] **Step 5: Commit the verified implementation**

Run:

```bash
git add model/backtest.py tests/test_backtest.py scripts/run_temporal_backtest.py tests/test_scripts.py
git commit -m "feat: add temporal backtest validation"
```

Expected: commit succeeds.
