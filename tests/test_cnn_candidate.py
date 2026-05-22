import unittest

import numpy as np
import pandas as pd


class CnnCandidateTests(unittest.TestCase):
    def test_add_calendar_features_adds_deterministic_cyclical_columns(self):
        from model import cnn_candidate

        df = pd.DataFrame({
            "region_id": [1, 1],
            "date": ["2020-01-01", "2020-07-01"],
            "score": [0.0, 1.0],
            "rain": [2.0, 3.0],
        })

        out = cnn_candidate.add_calendar_features(df)

        self.assertIsNot(out, df)
        self.assertIn("calendar__doy_sin", out.columns)
        self.assertIn("calendar__doy_cos", out.columns)
        self.assertIn("calendar__month_sin", out.columns)
        self.assertIn("calendar__month_cos", out.columns)
        self.assertIn("calendar__week_sin", out.columns)
        self.assertIn("calendar__week_cos", out.columns)
        self.assertTrue(np.all(out["calendar__doy_sin"].between(-1.0, 1.0)))
        self.assertTrue(np.all(out["calendar__doy_cos"].between(-1.0, 1.0)))
        self.assertAlmostEqual(out.loc[0, "calendar__month_sin"], 0.5, places=6)
        self.assertAlmostEqual(out.loc[0, "calendar__month_cos"], np.sqrt(3) / 2, places=6)
        self.assertEqual(df.columns.tolist(), ["region_id", "date", "score", "rain"])

    def test_add_calendar_features_handles_out_of_bounds_dates(self):
        from model import cnn_candidate

        df = pd.DataFrame({
            "region_id": [1, 2],
            "date": ["3004-12-31", "2020-07-01"],
            "score": [0.0, 1.0],
            "rain": [2.0, 3.0],
        })

        out = cnn_candidate.add_calendar_features(df)

        self.assertIn("calendar__doy_sin", out.columns)
        self.assertFalse(out.isna().any().any())

    def test_sequence_train_builder_includes_calendar_features_only_when_requested(self):
        from model import cnn_candidate

        rows = []
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        for idx, date in enumerate(dates):
            rows.append({
                "region_id": 1,
                "date": date.strftime("%Y-%m-%d"),
                "score": float(idx) if idx >= 91 else np.nan,
                "rain": float(idx),
            })
        df = pd.DataFrame(rows)

        X_plain, y_plain, regions_plain, plain_cols = cnn_candidate.build_sequence_train_data_from_frame(
            df,
            max_windows_per_region=2,
        )
        X_calendar, y_calendar, regions_calendar, calendar_cols = cnn_candidate.build_sequence_train_data_from_frame(
            df,
            max_windows_per_region=2,
            include_calendar=True,
        )

        self.assertEqual(X_plain.shape[0], X_calendar.shape[0])
        self.assertEqual(y_plain.shape, y_calendar.shape)
        self.assertEqual(regions_plain, regions_calendar)
        self.assertEqual(plain_cols, ["rain"])
        self.assertIn("calendar__doy_sin", calendar_cols)
        self.assertEqual(X_calendar.shape[2], X_plain.shape[2] + 6)

    def test_sequence_test_builder_uses_calendar_feature_columns(self):
        from model import cnn_candidate

        rows = []
        dates = pd.date_range("2020-01-01", periods=91, freq="D")
        for idx, date in enumerate(dates):
            rows.append({
                "region_id": 1,
                "date": date.strftime("%Y-%m-%d"),
                "rain": float(idx),
            })
        df = pd.DataFrame(rows)
        df_with_calendar = cnn_candidate.add_calendar_features(df)
        feat_cols = ["rain", "calendar__doy_sin", "calendar__doy_cos"]

        X_test, regions = cnn_candidate.build_sequence_test_data_from_frame(df_with_calendar, feat_cols)

        self.assertEqual(regions, [1])
        self.assertEqual(X_test.shape, (1, 91, 3))

    def test_cnn_gru_model_returns_five_week_predictions(self):
        from model import cnn_candidate

        if cnn_candidate.require_deep_learning_backend() != "torch":
            self.skipTest("PyTorch is required for CNN-GRU")

        import torch

        model = cnn_candidate.build_torch_model("cnn_gru", n_features=4, dropout=0.15)
        x = torch.zeros((3, 91, 4), dtype=torch.float32)

        with torch.no_grad():
            out = model(x)

        self.assertEqual(tuple(out.shape), (3, 5))


if __name__ == "__main__":
    unittest.main()
