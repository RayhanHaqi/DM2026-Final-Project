import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class TemporalFeatureTests(unittest.TestCase):
    def test_hybrid_temporal_features_keep_baseline_and_targeted_deltas(self):
        from model import temporal_features

        window = np.arange(91, dtype=float).reshape(91, 1)
        features = temporal_features.build_hybrid_temporal_features_from_window(window, ["rain"])

        self.assertAlmostEqual(features["rain__trend"], 1.0)
        self.assertIn("rain__skew", features)
        self.assertIn("rain__kurt", features)
        self.assertAlmostEqual(features["temporal__rain__first30_mean"], 14.5)
        self.assertAlmostEqual(features["temporal__rain__last7_minus_full91_mean"], 42.0)
        self.assertAlmostEqual(features["temporal__rain__last_value_minus_first_value"], 90.0)

    def test_hybrid_temporal_features_are_smaller_than_full_block_features(self):
        from model import temporal_features

        window = np.arange(91, dtype=float).reshape(91, 1)
        hybrid = temporal_features.build_hybrid_temporal_features_from_window(window, ["rain"])
        blocks = temporal_features.build_temporal_features_from_window(window, ["rain"])

        self.assertLess(len(hybrid), len(blocks))

    def test_build_temporal_features_preserves_recent_block_and_delta_signal(self):
        from model import temporal_features

        window = np.arange(91, dtype=float).reshape(91, 1)
        features = temporal_features.build_temporal_features_from_window(window, ["rain"])

        self.assertAlmostEqual(features["full91__rain__mean"], 45.0)
        self.assertAlmostEqual(features["first30__rain__mean"], 14.5)
        self.assertAlmostEqual(features["last30__rain__mean"], 75.5)
        self.assertAlmostEqual(features["last7__rain__mean"], 87.0)
        self.assertAlmostEqual(features["delta_last7_full91__rain__mean"], 42.0)
        self.assertAlmostEqual(features["delta_last30_first30__rain__mean"], 61.0)
        self.assertAlmostEqual(features["full91__rain__last_minus_first"], 90.0)

    def test_build_temporal_train_data_from_frame_uses_91_days_before_five_labels(self):
        from model import temporal_features

        df = pd.DataFrame({
            "region_id": ["R1"] * 96,
            "date": [f"2026-01-{(day % 28) + 1:02d}" for day in range(96)],
            "score": np.arange(96, dtype=float),
            "rain": np.arange(96, dtype=float),
        })

        X, y, regions = temporal_features.build_temporal_train_data_from_frame(df, feature_set="hybrid")

        self.assertEqual(X.shape[0], 1)
        self.assertEqual(regions, ["R1"])
        np.testing.assert_allclose(y[0], [91.0, 92.0, 93.0, 94.0, 95.0])
        self.assertAlmostEqual(X.iloc[0]["rain__mean"], 45.0)

    def test_build_temporal_test_data_from_frame_uses_last_91_days(self):
        from model import temporal_features

        df = pd.DataFrame({
            "region_id": ["R1"] * 100,
            "date": [f"2026-01-{(day % 28) + 1:02d}" for day in range(100)],
            "rain": np.arange(100, dtype=float),
        })

        X, regions = temporal_features.build_temporal_test_data_from_frame(df, feature_set="hybrid")

        self.assertEqual(X.shape[0], 1)
        self.assertEqual(regions, ["R1"])
        self.assertAlmostEqual(X.iloc[0]["rain__mean"], 54.0)


class FakeWeekEstimator:
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class TemporalTreeTests(unittest.TestCase):
    def test_train_week_models_fits_one_model_per_target_week(self):
        from model import temporal_tree

        X = pd.DataFrame({"f": [0.0, 1.0, 2.0]})
        y = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
        ])

        with patch.object(temporal_tree, "XGBRegressor", FakeWeekEstimator):
            models = temporal_tree.train_week_models(X, y, params_override={"max_depth": 2})
            preds = temporal_tree.predict_week_models(models, X)

        self.assertEqual(len(models), 5)
        self.assertEqual(preds.shape, (3, 5))
        np.testing.assert_allclose(preds[0], [2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(models[0].params["max_depth"], 2)


class CnnCandidateTests(unittest.TestCase):
    def test_require_deep_learning_backend_fails_loudly_when_missing(self):
        from model import cnn_candidate

        with patch.object(cnn_candidate.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "PyTorch or TensorFlow"):
                cnn_candidate.require_deep_learning_backend()

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_build_torch_model_v2_has_more_capacity_and_predicts_five_weeks(self):
        import torch
        from model import cnn_candidate

        small = cnn_candidate.build_torch_model("small", n_features=14, dropout=0.15)
        v2 = cnn_candidate.build_torch_model("v2", n_features=14, dropout=0.20)
        x = torch.zeros((2, 91, 14), dtype=torch.float32)

        self.assertEqual(small(x).shape, (2, 5))
        self.assertEqual(v2(x).shape, (2, 5))
        small_params = sum(p.numel() for p in small.parameters())
        v2_params = sum(p.numel() for p in v2.parameters())
        self.assertGreater(v2_params, small_params)


if __name__ == "__main__":
    unittest.main()
