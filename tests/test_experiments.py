import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from model import experiments


class FakeModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full((len(X), 5), self.value)


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

    def test_select_top_candidates_sorts_by_cv_mae(self):
        results = [
            {"name": "b", "cv_mae": 0.4},
            {"name": "a", "cv_mae": 0.3},
            {"name": "c", "cv_mae": 0.5},
        ]

        selected = experiments.select_top_candidates(results, limit=2)

        self.assertEqual([r["name"] for r in selected], ["a", "b"])

    def test_run_daily_candidates_writes_top_two_and_blend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = os.path.join(tmpdir, "sample_submission.csv")
            current_best_path = os.path.join(tmpdir, "current_best.csv")
            output_dir = os.path.join(tmpdir, "out")
            sample = pd.DataFrame({
                "region_id": ["R1", "R2"],
                "pred_week1": [0, 0],
                "pred_week2": [0, 0],
                "pred_week3": [0, 0],
                "pred_week4": [0, 0],
                "pred_week5": [0, 0],
            })
            sample.to_csv(sample_path, index=False)
            current_best = sample.copy()
            current_best.iloc[:, 1:] = 1.0
            current_best.to_csv(current_best_path, index=False)

            candidates = [
                {"name": "candidate_a", "params": {"score": 0.3, "pred": 3.0}},
                {"name": "candidate_b", "params": {"score": 0.2, "pred": 2.0}},
                {"name": "candidate_c", "params": {"score": 0.4, "pred": 4.0}},
            ]

            def fake_cv_evaluate(X, y, groups, n_splits=5, params_override=None):
                return [params_override["score"]], params_override["score"], 0.0

            def fake_train_xgboost(X, y, params_override=None):
                return FakeModel(params_override["pred"])

            with patch.object(experiments.train, "cv_evaluate", fake_cv_evaluate), patch.object(experiments.train, "train_xgboost", fake_train_xgboost):
                results = experiments.run_daily_candidates(
                    X_train=pd.DataFrame({"f": [0, 1]}),
                    y_train=np.zeros((2, 5)),
                    train_regions=["R1", "R2"],
                    X_test=pd.DataFrame({"f": [0, 1]}),
                    test_regions=["R2", "R1"],
                    sample_path=sample_path,
                    current_best_path=current_best_path,
                    output_dir=output_dir,
                    candidates=candidates,
                    limit=3,
                    timestamp="20260518_143012",
                )

            self.assertEqual([r["name"] for r in results], ["candidate_b", "candidate_a", "blend_candidate_b_070"])
            for result in results:
                self.assertIn("20260518_143012", os.path.basename(result["path"]))
                sub = pd.read_csv(result["path"])
                self.assertEqual(sub["region_id"].tolist(), ["R1", "R2"])
                self.assertEqual(list(sub.columns), list(sample.columns))

            self.assertTrue(os.path.exists(os.path.join(output_dir, "summary_20260518_143012.csv")))


if __name__ == "__main__":
    unittest.main()
