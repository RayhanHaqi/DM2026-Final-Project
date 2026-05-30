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
