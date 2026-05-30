import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import temporal_tree


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
