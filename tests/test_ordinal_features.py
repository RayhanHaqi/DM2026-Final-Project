import unittest
from unittest.mock import patch

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

    def test_feature_sets_tuple_documents_supported_modes(self):
        self.assertIn("hybrid", ORDINAL_FEATURE_SETS)
        self.assertIn("hybrid_season", ORDINAL_FEATURE_SETS)


if __name__ == "__main__":
    unittest.main()
