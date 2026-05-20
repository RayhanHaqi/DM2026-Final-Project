import unittest

import numpy as np
import pandas as pd

from model import experiments


class ExperimentTests(unittest.TestCase):
    def test_clip_predictions_limits_values_to_target_range(self):
        preds = np.array([[-1.0, 2.0, 6.0]])
        result = experiments.clip_predictions(preds)
        np.testing.assert_allclose(result, [[0.0, 2.0, 5.0]])

    def test_blend_predictions_combines_two_arrays(self):
        current = np.array([[1.0, 3.0]])
        candidate = np.array([[3.0, 5.0]])
        result = experiments.blend_predictions(candidate, current, candidate_weight=0.75)
        np.testing.assert_allclose(result, [[2.5, 4.5]])

    def test_build_submission_matches_sample_order_and_columns(self):
        sample = pd.DataFrame({
            "region_id": ["R1", "R2", "R3"],
            "pred_week1": [0, 0, 0],
            "pred_week2": [0, 0, 0],
        })
        region_ids = ["R3", "R1", "R2"]
        preds = np.array([
            [3.1, 3.2],
            [1.1, 1.2],
            [2.1, 2.2],
        ])

        sub = experiments.build_submission(region_ids, preds, sample)

        self.assertEqual(list(sub.columns), list(sample.columns))
        self.assertEqual(sub["region_id"].tolist(), ["R1", "R2", "R3"])
        self.assertEqual(sub["pred_week1"].tolist(), [1.1, 2.1, 3.1])
        self.assertEqual(sub["pred_week2"].tolist(), [1.2, 2.2, 3.2])

    def test_validate_submission_requires_sample_columns_and_order(self):
        sample = pd.DataFrame({"region_id": ["R1", "R2"], "pred_week1": [0, 0]})
        sub = pd.DataFrame({"region_id": ["R2", "R1"], "pred_week1": [0.2, 0.1]})

        ok, messages = experiments.validate_submission(sub, sample)

        self.assertFalse(ok)
        self.assertIn("ID order does not match sample submission", messages)

if __name__ == "__main__":
    unittest.main()
