import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts import compare_candidate_distribution as compare


def make_submission(path, region_ids=("R1", "R2"), value_offset=0.0):
    df = pd.DataFrame({
        "region_id": list(region_ids),
        "pred_week1": [1.0 + value_offset, 2.0 + value_offset],
        "pred_week2": [1.5 + value_offset, 2.5 + value_offset],
        "pred_week3": [2.0 + value_offset, 3.0 + value_offset],
        "pred_week4": [2.5 + value_offset, 3.5 + value_offset],
        "pred_week5": [3.0 + value_offset, 4.0 + value_offset],
    })
    df.to_csv(path, index=False)
    return df


class CandidateDistributionTests(unittest.TestCase):
    def test_identical_predictions_have_perfect_correlation_and_zero_diff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = f"{tmpdir}/reference.csv"
            candidate_path = f"{tmpdir}/candidate.csv"
            make_submission(reference_path)
            make_submission(candidate_path)

            result = compare.compare_candidate(candidate_path, reference_path)

        self.assertAlmostEqual(result["correlation"], 1.0)
        self.assertAlmostEqual(result["mean_abs_diff"], 0.0)
        self.assertTrue(result["safe"])

    def test_shifted_predictions_have_nonzero_diff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = f"{tmpdir}/reference.csv"
            candidate_path = f"{tmpdir}/candidate.csv"
            make_submission(reference_path)
            make_submission(candidate_path, value_offset=0.2)

            result = compare.compare_candidate(candidate_path, reference_path)

        self.assertAlmostEqual(result["mean_abs_diff"], 0.2)
        self.assertFalse(result["safe"])

    def test_mismatched_id_order_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = f"{tmpdir}/reference.csv"
            candidate_path = f"{tmpdir}/candidate.csv"
            make_submission(reference_path, region_ids=("R1", "R2"))
            make_submission(candidate_path, region_ids=("R2", "R1"))

            with self.assertRaisesRegex(ValueError, "ID order"):
                compare.compare_candidate(candidate_path, reference_path)

    def test_mismatched_columns_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = f"{tmpdir}/reference.csv"
            candidate_path = f"{tmpdir}/candidate.csv"
            reference = make_submission(reference_path)
            candidate = reference.drop(columns=["pred_week5"])
            candidate.to_csv(candidate_path, index=False)

            with self.assertRaisesRegex(ValueError, "Columns"):
                compare.compare_candidate(candidate_path, reference_path)


if __name__ == "__main__":
    unittest.main()
