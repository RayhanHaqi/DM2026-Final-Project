import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import model.train as train
import model.utils as utils


class FakeModel:
    def predict(self, X):
        return np.zeros((len(X), 5))


class FakeMultiOutputRegressor:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        return self


class ProgressTests(unittest.TestCase):
    def test_train_xgboost_uses_safe_default_resource_settings(self):
        captured_params = []

        class FakeXGBRegressor:
            def __init__(self, **params):
                captured_params.append(params)

        X = pd.DataFrame({"feature": [0, 1]})
        y = np.zeros((2, 5))

        with patch.object(train, "XGBRegressor", FakeXGBRegressor), patch.object(train, "MultiOutputRegressor", FakeMultiOutputRegressor):
            train.train_xgboost(X, y)

        self.assertEqual(captured_params[0]["n_estimators"], 200)
        self.assertEqual(captured_params[0]["n_jobs"], 2)

    def test_cv_evaluate_wraps_folds_with_tqdm(self):
        calls = []

        def fake_tqdm(iterable, **kwargs):
            calls.append(kwargs)
            return iterable

        X = pd.DataFrame({"feature": [0, 1, 2, 3]})
        y = np.zeros((4, 5))
        groups = ["a", "a", "b", "b"]

        with patch.object(train, "tqdm", fake_tqdm), patch.object(train, "train_xgboost", return_value=FakeModel()):
            train.cv_evaluate(X, y, groups, n_splits=2)

        self.assertEqual(calls, [{"total": 2, "desc": "Cross-validation"}])

    def test_cv_evaluate_passes_params_override_to_train_xgboost(self):
        params_seen = []

        def fake_train_xgboost(X, y, params_override=None):
            params_seen.append(params_override)
            return FakeModel()

        X = pd.DataFrame({"feature": [0, 1, 2, 3]})
        y = np.zeros((4, 5))
        groups = ["a", "a", "b", "b"]
        params = {"max_depth": 4}

        with patch.object(train, "tqdm", lambda iterable, **kwargs: iterable), patch.object(train, "train_xgboost", fake_train_xgboost):
            train.cv_evaluate(X, y, groups, n_splits=2, params_override=params)

        self.assertEqual(params_seen, [params, params])

    def test_build_train_features_wraps_regions_with_tqdm(self):
        calls = []

        def fake_tqdm(iterable, **kwargs):
            calls.append(kwargs)
            return iterable

        rows = []
        for region_id in ["a", "b"]:
            for day in range(96):
                rows.append({
                    "region_id": region_id,
                    "date": f"2026-01-{(day % 28) + 1:02d}",
                    "score": 0.0,
                    "feature": float(day),
                })
        df = pd.DataFrame(rows)

        with patch.object(utils, "tqdm", fake_tqdm), patch.object(pd, "read_csv", return_value=df), patch.object(utils, "_aggregate_array", return_value=pd.Series({"feature__mean": 0.0})):
            X, y, regions = utils._build_train_features("unused.csv", max_windows_per_region=None)

        self.assertEqual(calls, [{"total": 2, "desc": "Building train windows"}])
        self.assertEqual(X.shape, (2, 1))
        self.assertEqual(y.shape, (2, 5))
        self.assertEqual(regions, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
