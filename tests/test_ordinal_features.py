import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from model.ordinal_features import ORDINAL_FEATURE_SETS, load_ordinal_train_test


class OrdinalFeaturesTests(unittest.TestCase):
    def test_invalid_feature_set_raises(self):
        with self.assertRaises(ValueError):
            load_ordinal_train_test("a.csv", "b.csv", feature_set="invalid")

    @patch("model.ordinal_features.temporal_features.load_temporal_test_data")
    @patch("model.ordinal_features.temporal_features.load_temporal_train_data")
    def test_hybrid_returns_train_regions(self, mock_train, mock_test):
        mock_train.return_value = (pd.DataFrame({"a": [1, 2]}), [[0, 1]], ["R1", "R2"])
        mock_test.return_value = (pd.DataFrame({"a": [3, 4]}), ["R1", "R2"])

        X_train, y_train, train_regions, X_test, test_regions = load_ordinal_train_test(
            "train.csv", "test.csv", feature_set="hybrid"
        )

        self.assertEqual(list(X_train.columns), ["a"])
        self.assertEqual(train_regions, ["R1", "R2"])
        self.assertEqual(test_regions, ["R1", "R2"])
        self.assertEqual(y_train, [[0, 1]])

    @patch("model.ordinal_features.pd.read_csv")
    @patch(
        "model.ordinal_features.severity_history.build_test_blackout_history_features_from_frame"
    )
    @patch(
        "model.ordinal_features.severity_history.build_train_blackout_history_features_from_frame"
    )
    @patch("model.ordinal_features.temporal_features.load_temporal_test_data")
    @patch("model.ordinal_features.temporal_features.load_temporal_train_data")
    def test_hybrid_blackout_concatenates_history_columns(
        self,
        mock_train,
        mock_test,
        mock_train_blackout,
        mock_test_blackout,
        mock_read_csv,
    ):
        mock_train.return_value = (pd.DataFrame({"a": [1, 2]}), [[0, 1]], ["R1", "R2"])
        mock_test.return_value = (pd.DataFrame({"a": [3, 4]}), ["R1", "R2"])
        mock_read_csv.return_value = pd.DataFrame({"region_id": ["R1", "R2"], "score": [1.0, 2.0]})
        mock_train_blackout.return_value = pd.DataFrame(
            {"score_history_mean": [0.1, 0.2], "score_history_count": [1.0, 2.0]}
        )
        mock_test_blackout.return_value = pd.DataFrame(
            {"score_history_mean": [0.3, 0.4], "score_history_count": [3.0, 4.0]}
        )

        X_train, _, _, X_test, test_regions = load_ordinal_train_test(
            "train.csv", "test.csv", feature_set="hybrid_blackout"
        )

        self.assertEqual(test_regions, ["R1", "R2"])
        self.assertIn("a", X_train.columns)
        self.assertIn("history__score_history_mean", X_train.columns)
        self.assertIn("history__score_history_count", X_test.columns)
        mock_test_blackout.assert_called_once()
        call_kwargs = mock_test_blackout.call_args.kwargs
        self.assertEqual(call_kwargs["window_days"], 91)
        self.assertEqual(list(mock_test_blackout.call_args.args[1]), ["R1", "R2"])

    @patch("model.ordinal_features.pd.read_csv")
    @patch(
        "model.ordinal_features.severity_history.build_train_blackout_history_features_from_frame"
    )
    @patch("model.ordinal_features.temporal_features.load_temporal_test_data")
    @patch("model.ordinal_features.temporal_features.load_temporal_train_data")
    def test_hybrid_blackout_train_mismatch_raises(
        self,
        mock_train,
        mock_test,
        mock_train_blackout,
        mock_read_csv,
    ):
        mock_train.return_value = (pd.DataFrame({"a": [1, 2]}), [[0, 1]], ["R1", "R2"])
        mock_test.return_value = (pd.DataFrame({"a": [3]}), ["R1"])
        mock_read_csv.return_value = pd.DataFrame({"region_id": ["R1"], "score": [1.0]})
        mock_train_blackout.return_value = pd.DataFrame({"score_history_mean": [0.1]})

        with self.assertRaisesRegex(ValueError, "blackout row mismatch"):
            load_ordinal_train_test("train.csv", "test.csv", feature_set="hybrid_blackout")

    @patch("model.ordinal_features.pd.read_csv")
    @patch(
        "model.ordinal_features.severity_history.build_test_blackout_history_features_from_frame"
    )
    @patch(
        "model.ordinal_features.severity_history.build_train_blackout_history_features_from_frame"
    )
    @patch("model.ordinal_features.temporal_features.load_temporal_test_data")
    @patch("model.ordinal_features.temporal_features.load_temporal_train_data")
    def test_history_only_uses_history_columns_without_hybrid(
        self,
        mock_train,
        mock_test,
        mock_train_blackout,
        mock_test_blackout,
        mock_read_csv,
    ):
        mock_train.return_value = (
            pd.DataFrame({"a": [1, 2]}),
            np.array([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]]),
            ["R1", "R2"],
        )
        mock_test.return_value = (pd.DataFrame({"a": [3]}), ["R1"])
        mock_read_csv.return_value = pd.DataFrame({"region_id": ["R1"], "score": [1.0]})
        mock_train_blackout.return_value = pd.DataFrame(
            {"score_history_mean": [0.1, 0.2], "score_history_count": [1.0, 2.0]}
        )
        mock_test_blackout.return_value = pd.DataFrame(
            {"score_history_mean": [0.3], "score_history_count": [3.0]}
        )

        X_train, y_train, _, X_test, _ = load_ordinal_train_test(
            "train.csv", "test.csv", feature_set="history_only"
        )

        self.assertNotIn("a", X_train.columns)
        self.assertIn("history__score_history_mean", X_train.columns)
        self.assertEqual(len(X_train), len(y_train))

    def test_feature_sets_tuple_documents_supported_modes(self):
        self.assertIn("hybrid", ORDINAL_FEATURE_SETS)
        self.assertIn("hybrid_season", ORDINAL_FEATURE_SETS)
        self.assertIn("hybrid_blackout", ORDINAL_FEATURE_SETS)
        self.assertIn("history_only", ORDINAL_FEATURE_SETS)


if __name__ == "__main__":
    unittest.main()
