import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
