# Grid Search + LightGBM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a staged grid search over XGBoost parameters (108 combinations) and train LightGBM for comparison, generating top 3 submissions for Kaggle.

**Architecture:** Staged grid search over XGBoost tree parameters (depth, trees, learning rate, subsample, colsample, regularization), then LightGBM training with best params, then blend weight optimization. All using 2 CPU threads.

**Tech Stack:** XGBoost, LightGBM, scikit-learn (GroupKFold), pandas, numpy

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/grid_search_xgboost.py` | Main grid search script: load data, run staged CV, log results, generate submissions |
| `scripts/generate_lgbm_submission.py` | LightGBM training + submission generation |
| `tests/test_grid_search.py` | Unit tests for grid search logic |
| `output/grid_search/grid_search_log.csv` | All CV results (auto-created) |
| `output/grid_search/best_params.json` | Best parameters (auto-created) |

---

## Task 1: Create Grid Search Test File

**Files:**
- Create: `tests/test_grid_search.py`

- [ ] **Step 1: Write test file with helper imports**

```python
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import temporal_tree
```

- [ ] **Step 2: Write test for param grid generation**

```python
def test_generate_param_grid_stage_a():
    """Stage A should produce 36 combinations (4 depth × 3 trees × 3 lr)."""
    from scripts.grid_search_xgboost import generate_param_grid_stage_a
    grid = generate_param_grid_stage_a()
    assert len(grid) == 36
    # Verify all combos have fixed subsample/colsample/reg
    for params in grid:
        assert params["subsample"] == 0.85
        assert params["colsample_bytree"] == 0.85
        assert params["reg_alpha"] == 0.2
        assert params["reg_lambda"] == 1.5
```

- [ ] **Step 3: Write test for single CV run**

```python
def test_run_single_cv_returns_valid_mae():
    """CV should return mean MAE > 0 and finite."""
    from scripts.grid_search_xgboost import run_single_cv
    # Create tiny synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = np.random.rand(n_samples, 5) * 5
    regions = [f"r{i//10}" for i in range(n_samples)]
    
    params = {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": 2,
    }
    
    mae = run_single_cv(X, y, regions, params, n_splits=2)
    assert isinstance(mae, float)
    assert mae > 0
    assert np.isfinite(mae)
```

- [ ] **Step 4: Write test for stage runner**

```python
def test_run_stage_returns_sorted_results():
    """Stage runner should return list of (params, mae) sorted by mae."""
    from scripts.grid_search_xgboost import run_stage
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = np.random.rand(n_samples, 5) * 5
    regions = [f"r{i//10}" for i in range(n_samples)]
    
    fixed = {
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": 2,
    }
    
    grid = [
        {"max_depth": 3, "n_estimators": 10, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 10, "learning_rate": 0.1},
    ]
    
    results = run_stage(X, y, regions, grid, fixed, n_splits=2)
    assert len(results) == 2
    # Verify sorted by MAE (ascending)
    assert results[0][1] <= results[1][1]
    # Verify each result has (params, mae)
    for params, mae in results:
        assert isinstance(mae, float)
        assert "max_depth" in params
```

- [ ] **Step 5: Write test for submission generation**

```python
def test_generate_submission_validates():
    """Generated submission should pass validation."""
    from scripts.grid_search_xgboost import generate_submission_from_params
    from model import experiments
    
    np.random.seed(42)
    n_train = 200
    n_test = 50
    n_features = 10
    
    X_train = pd.DataFrame(np.random.randn(n_train, n_features))
    y_train = np.random.rand(n_train, 5) * 5
    X_test = pd.DataFrame(np.random.randn(n_test, n_features))
    test_regions = [f"r{i}" for i in range(n_test)]
    
    sample = pd.DataFrame({
        "region_id": test_regions,
        "week1": [0.0] * n_test,
        "week2": [0.0] * n_test,
        "week3": [0.0] * n_test,
        "week4": [0.0] * n_test,
        "week5": [0.0] * n_test,
    })
    
    params = {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": 2,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_sub.csv")
        generate_submission_from_params(X_train, y_train, X_test, test_regions, params, sample, out_path)
        
        assert os.path.exists(out_path)
        sub = pd.read_csv(out_path)
        ok, messages = experiments.validate_submission(sub, sample)
        assert ok, f"Validation failed: {messages}"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_grid_search.py -v`
Expected: All 4 tests FAIL with import errors (scripts.grid_search_xgboost not found)

- [ ] **Step 7: Commit test file**

```bash
git add tests/test_grid_search.py
git commit -m "test: add grid search test stubs"
```

---

## Task 2: Create Grid Search Script

**Files:**
- Create: `scripts/grid_search_xgboost.py`

- [ ] **Step 1: Write script skeleton with imports and constants**

```python
import json
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import experiments, temporal_features, temporal_tree
from model.same_season import build_train_same_season_features, build_test_same_season_features
from xgboost import XGBRegressor

THREADS = 2
MAX_WINDOWS = 52
N_SPLITS = 5
OUTPUT_DIR = "output/grid_search"
```

- [ ] **Step 2: Write param grid generators**

```python
def generate_param_grid_stage_a():
    """Stage A: Sweep depth, trees, lr with fixed regularization."""
    grid = []
    for max_depth, n_estimators, learning_rate in product(
        [3, 4, 5, 6],
        [200, 250, 300],
        [0.03, 0.04, 0.05],
    ):
        grid.append({
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "random_state": 42,
            "n_jobs": THREADS,
        })
    return grid


def generate_param_grid_stage_b(best_depth, best_trees, best_lr):
    """Stage B: Sweep subsample and colsample with best from A."""
    grid = []
    for subsample, colsample in product([0.8, 0.85, 0.9], [0.8, 0.85, 0.9]):
        grid.append({
            "max_depth": best_depth,
            "n_estimators": best_trees,
            "learning_rate": best_lr,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "random_state": 42,
            "n_jobs": THREADS,
        })
    return grid


def generate_param_grid_stage_c(best_params):
    """Stage C: Sweep regularization with best from A+B."""
    grid = []
    for reg_alpha, reg_lambda in product([0.1, 0.2, 0.3], [1.0, 1.5, 2.0]):
        params = dict(best_params)
        params["reg_alpha"] = reg_alpha
        params["reg_lambda"] = reg_lambda
        grid.append(params)
    return grid


def generate_param_grid_stage_d(best_params):
    """Stage D: Refinement around best from A+B+C."""
    variations = {
        "max_depth": [-1, 0, 1],
        "n_estimators": [-50, 0, 50],
        "learning_rate": [-0.01, 0, 0.01],
        "subsample": [-0.05, 0, 0.05],
        "colsample_bytree": [-0.05, 0, 0.05],
        "reg_alpha": [-0.1, 0, 0.1],
        "reg_lambda": [-0.5, 0, 0.5],
    }
    
    grid = []
    for key, deltas in variations.items():
        for delta in deltas:
            if delta == 0:
                continue
            params = dict(best_params)
            if key in ("max_depth", "n_estimators"):
                params[key] = max(1, best_params[key] + delta)
            else:
                params[key] = max(0.01, best_params[key] + delta)
            grid.append(params)
    
    # Also add the best params itself
    grid.append(dict(best_params))
    return grid
```

- [ ] **Step 3: Write CV runner functions**

```python
def run_single_cv(X, y, regions, params, n_splits=N_SPLITS):
    """Run GroupKFold CV, return mean MAE."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error
    
    kf = GroupKFold(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in kf.split(X, y, regions):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        
        models = temporal_tree.train_week_models(X_tr, y[train_idx], params_override=params)
        preds = temporal_tree.predict_week_models(models, X_val)
        scores.append(mean_absolute_error(y[val_idx], preds))
    
    return float(np.mean(scores))


def run_stage(X, y, regions, param_grid, fixed_params, n_splits=N_SPLITS, desc="Stage"):
    """Run all combinations in a stage, return sorted (params, mae) list."""
    results = []
    
    for params in tqdm(param_grid, desc=desc):
        full_params = dict(fixed_params)
        full_params.update(params)
        mae = run_single_cv(X, y, regions, full_params, n_splits=n_splits)
        results.append((full_params, mae))
    
    results.sort(key=lambda x: x[1])
    return results
```

- [ ] **Step 4: Write submission generator**

```python
def generate_submission_from_params(X_train, y_train, X_test, test_regions, params, sample, out_path):
    """Train full model with params, generate submission CSV."""
    models = temporal_tree.train_week_models(X_train, y_train, params_override=params)
    preds = temporal_tree.predict_week_models(models, X_test)
    preds = np.clip(preds, 0.0, 5.0)
    
    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise ValueError(f"Submission validation failed: {'; '.join(messages)}")
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub.to_csv(out_path, index=False)
    return out_path
```

- [ ] **Step 5: Write main function**

```python
def main():
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    sample_path = "data/sample_submission.csv"
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    
    print("Loading hybrid features...")
    X_train, y_train, train_regions = temporal_features.load_temporal_train_data(
        train_path, max_windows_per_region=MAX_WINDOWS, feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(test_path, feature_set="hybrid")
    
    print("Building same-season features...")
    season_train = build_train_same_season_features(train_df, max_windows_per_region=MAX_WINDOWS)
    season_test = build_test_same_season_features(train_df, test_df)
    
    X_train_full = pd.concat([X_train.reset_index(drop=True), season_train], axis=1)
    X_test_full = pd.concat([X_test.reset_index(drop=True), season_test], axis=1)
    
    all_results = []
    
    # Stage A: depth, trees, lr
    print("\n=== Stage A: depth, trees, lr (36 combos) ===")
    grid_a = generate_param_grid_stage_a()
    results_a = run_stage(X_train_full, y_train, train_regions, grid_a, {}, desc="Stage A")
    all_results.extend(results_a)
    best_a = results_a[0][0]
    print(f"Best Stage A: MAE={results_a[0][1]:.6f}, params={best_a}")
    
    # Stage B: subsample, colsample
    print("\n=== Stage B: subsample, colsample (9 combos) ===")
    grid_b = generate_param_grid_stage_b(best_a["max_depth"], best_a["n_estimators"], best_a["learning_rate"])
    results_b = run_stage(X_train_full, y_train, train_regions, grid_b, {}, desc="Stage B")
    all_results.extend(results_b)
    best_b = results_b[0][0]
    print(f"Best Stage B: MAE={results_b[0][1]:.6f}, params={best_b}")
    
    # Stage C: regularization
    print("\n=== Stage C: reg_alpha, reg_lambda (9 combos) ===")
    grid_c = generate_param_grid_stage_c(best_b)
    results_c = run_stage(X_train_full, y_train, train_regions, grid_c, {}, desc="Stage C")
    all_results.extend(results_c)
    best_c = results_c[0][0]
    print(f"Best Stage C: MAE={results_c[0][1]:.6f}, params={best_c}")
    
    # Stage D: refinement
    print("\n=== Stage D: refinement (~22 combos) ===")
    grid_d = generate_param_grid_stage_d(best_c)
    results_d = run_stage(X_train_full, y_train, train_regions, grid_d, {}, desc="Stage D")
    all_results.extend(results_d)
    
    # Sort all results
    all_results.sort(key=lambda x: x[1])
    
    # Log all results
    log_path = os.path.join(OUTPUT_DIR, "grid_search_log.csv")
    log_rows = []
    for params, mae in all_results:
        row = {"mae": mae}
        row.update(params)
        log_rows.append(row)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"\nSaved log: {log_path}")
    
    # Save best params
    best_params = all_results[0][0]
    best_path = os.path.join(OUTPUT_DIR, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best params: {best_path}")
    
    # Generate top 3 submissions
    print("\nGenerating top 3 submissions...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, (params, mae) in enumerate(all_results[:3]):
        out_path = os.path.join("output/daily_candidates", f"grid_search_top{i+1}_{timestamp}.csv")
        generate_submission_from_params(X_train_full, y_train, X_test_full, test_regions, params, sample, out_path)
        print(f"Top {i+1}: MAE={mae:.6f} -> {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_grid_search.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/grid_search_xgboost.py tests/test_grid_search.py
git commit -m "feat: add staged XGBoost grid search script"
```

---

## Task 3: Create LightGBM Script

**Files:**
- Create: `scripts/generate_lgbm_submission.py`
- Create: `tests/test_lgbm.py`

- [ ] **Step 1: Write LightGBM test file**

```python
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_lgbm_cv_returns_valid_mae():
    """LightGBM CV should return valid MAE."""
    from scripts.generate_lgbm_submission import run_lgbm_cv
    
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y = np.random.rand(n_samples, 5) * 5
    regions = [f"r{i//10}" for i in range(n_samples)]
    
    params = {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "n_jobs": 2,
        "verbose": -1,
    }
    
    mae = run_lgbm_cv(X, y, regions, params, n_splits=2)
    assert isinstance(mae, float)
    assert mae > 0
    assert np.isfinite(mae)


def test_lgbm_submission_validates():
    """LightGBM submission should pass validation."""
    from scripts.generate_lgbm_submission import generate_lgbm_submission
    from model import experiments
    
    np.random.seed(42)
    n_train = 200
    n_test = 50
    n_features = 10
    
    X_train = pd.DataFrame(np.random.randn(n_train, n_features))
    y_train = np.random.rand(n_train, 5) * 5
    X_test = pd.DataFrame(np.random.randn(n_test, n_features))
    test_regions = [f"r{i}" for i in range(n_test)]
    
    sample = pd.DataFrame({
        "region_id": test_regions,
        "week1": [0.0] * n_test,
        "week2": [0.0] * n_test,
        "week3": [0.0] * n_test,
        "week4": [0.0] * n_test,
        "week5": [0.0] * n_test,
    })
    
    params = {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "n_jobs": 2,
        "verbose": -1,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_lgbm.csv")
        generate_lgbm_submission(X_train, y_train, X_test, test_regions, params, sample, out_path)
        
        assert os.path.exists(out_path)
        sub = pd.read_csv(out_path)
        ok, messages = experiments.validate_submission(sub, sample)
        assert ok, f"Validation failed: {messages}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_lgbm.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Write LightGBM script**

```python
import json
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import experiments, temporal_features
from model.same_season import build_train_same_season_features, build_test_same_season_features
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

THREADS = 2
MAX_WINDOWS = 52
N_SPLITS = 5


def run_lgbm_cv(X, y, regions, params, n_splits=N_SPLITS):
    """Run GroupKFold CV with LightGBM, return mean MAE."""
    kf = GroupKFold(n_splits=n_splits)
    scores = []
    
    for train_idx, val_idx in kf.split(X, y, regions):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        
        preds = np.zeros((len(val_idx), y.shape[1]))
        for week_idx in range(y.shape[1]):
            model = LGBMRegressor(**params)
            model.fit(X_tr, y[train_idx, week_idx])
            preds[:, week_idx] = model.predict(X_val)
        
        scores.append(mean_absolute_error(y[val_idx], preds))
    
    return float(np.mean(scores))


def generate_lgbm_submission(X_train, y_train, X_test, test_regions, params, sample, out_path):
    """Train LightGBM, generate submission CSV."""
    preds = np.zeros((len(X_test), y_train.shape[1]))
    for week_idx in range(y_train.shape[1]):
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train[:, week_idx])
        preds[:, week_idx] = model.predict(X_test)
    
    preds = np.clip(preds, 0.0, 5.0)
    
    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise ValueError(f"Validation failed: {'; '.join(messages)}")
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub.to_csv(out_path, index=False)
    return out_path


def main():
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)
    
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    sample_path = "data/sample_submission.csv"
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    
    print("Loading hybrid features...")
    X_train, y_train, train_regions = temporal_features.load_temporal_train_data(
        train_path, max_windows_per_region=MAX_WINDOWS, feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(test_path, feature_set="hybrid")
    
    print("Building same-season features...")
    season_train = build_train_same_season_features(train_df, max_windows_per_region=MAX_WINDOWS)
    season_test = build_test_same_season_features(train_df, test_df)
    
    X_train_full = pd.concat([X_train.reset_index(drop=True), season_train], axis=1)
    X_test_full = pd.concat([X_test.reset_index(drop=True), season_test], axis=1)
    
    # LightGBM param grid
    param_grid = []
    for num_leaves, max_depth, learning_rate, n_estimators in product(
        [31, 50, 70],
        [4, 5, 6],
        [0.03, 0.04, 0.05],
        [200, 250, 300],
    ):
        param_grid.append({
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "n_jobs": THREADS,
            "verbose": -1,
            "random_state": 42,
        })
    
    print(f"\nRunning LightGBM grid search ({len(param_grid)} combos)...")
    results = []
    for params in tqdm(param_grid, desc="LightGBM CV"):
        mae = run_lgbm_cv(X_train_full, y_train, train_regions, params)
        results.append((params, mae))
    
    results.sort(key=lambda x: x[1])
    
    # Log results
    os.makedirs("output/grid_search", exist_ok=True)
    log_rows = []
    for params, mae in results:
        row = {"mae": mae}
        row.update(params)
        log_rows.append(row)
    pd.DataFrame(log_rows).to_csv("output/grid_search/lgbm_grid_search_log.csv", index=False)
    
    # Generate best submission
    best_params = results[0][0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"output/daily_candidates/lgbm_best_{timestamp}.csv"
    generate_lgbm_submission(X_train_full, y_train, X_test_full, test_regions, best_params, sample, out_path)
    
    print(f"\nBest LightGBM: MAE={results[0][1]:.6f}")
    print(f"Best params: {best_params}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_lgbm.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_lgbm_submission.py tests/test_lgbm.py
git commit -m "feat: add LightGBM grid search script"
```

---

## Task 4: Run Grid Search Overnight

**Files:**
- Execute: `scripts/grid_search_xgboost.py`
- Execute: `scripts/generate_lgbm_submission.py`

- [ ] **Step 1: Run XGBoost grid search**

Run: `python scripts/grid_search_xgboost.py`
Expected: Runs for ~8-13 hours, produces `output/grid_search/grid_search_log.csv` and top 3 submissions

- [ ] **Step 2: Verify grid search output**

Run: `head -20 output/grid_search/grid_search_log.csv`
Expected: CSV with columns for all params + mae, sorted by mae ascending

Run: `cat output/grid_search/best_params.json`
Expected: JSON with best XGBoost parameters

Run: `ls -la output/daily_candidates/grid_search_top*.csv`
Expected: 3 submission files

- [ ] **Step 3: Run LightGBM grid search**

Run: `python scripts/generate_lgbm_submission.py`
Expected: Runs for ~1-2 hours, produces `output/grid_search/lgbm_grid_search_log.csv` and best submission

- [ ] **Step 4: Verify LightGBM output**

Run: `cat output/grid_search/lgbm_grid_search_log.csv | head -5`
Expected: CSV with LightGBM params + mae

Run: `ls -la output/daily_candidates/lgbm_best_*.csv`
Expected: 1 submission file

---

## Task 5: Optimize Blend Weights

**Files:**
- Execute: `scripts/blend_submissions.py` (existing)

- [ ] **Step 1: Run blend weight optimization**

After grid search completes, optimize blend weights for top 3 XGBoost + LightGBM:

```bash
# For each top submission, blend with ordinal and season at various weights
for i in 1 2 3; do
  for ord_w in 0.005 0.01 0.015 0.02 0.025 0.03; do
    for season_w in 0.10 0.15 0.18 0.20 0.22 0.25; do
      python scripts/blend_submissions.py \
        --candidate output/daily_candidates/grid_search_top${i}_*.csv \
        --reference output/daily_candidates/tree4_ordinal_w015_20260525.csv \
        --candidate-weight ${ord_w} \
        --name "gs_top${i}_ord${ord_w}"
    done
  done
done
```

- [ ] **Step 2: Submit top 3 within daily budget**

After blend optimization, submit the best 3 combinations to Kaggle.

---

## Task 6: Update AGENTS.md

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update submission history with grid search results**

Add new rows to the submission history table with grid search results.

- [ ] **Step 2: Update current state if new best found**

If grid search produces a new best, update the Current State section.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: log grid search results"
```

---

## Verification Checklist

- [ ] All tests pass: `PYTHONPATH=. pytest tests/test_grid_search.py tests/test_lgbm.py -v`
- [ ] Grid search log has 100+ rows
- [ ] Top 3 submissions pass `experiments.validate_submission()`
- [ ] LightGBM submission passes validation
- [ ] All scripts use `n_jobs=2` and thread env vars = 2
- [ ] AGENTS.md updated with results
