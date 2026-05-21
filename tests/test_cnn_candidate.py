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


if __name__ == "__main__":
    unittest.main()
