import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from model import utils


class SubmissionTests(unittest.TestCase):
    def test_generate_submission_matches_sample_order_and_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(data_dir)
            os.makedirs(output_dir)

            pd.DataFrame({
                "region_id": ["R1", "R2", "R3"],
                "pred_week1": [0, 0, 0],
                "pred_week2": [0, 0, 0],
            }).to_csv(os.path.join(data_dir, "sample_submission.csv"), index=False)

            region_ids = ["R3", "R1", "R2"]
            preds = np.array([
                [3.1, 3.2],
                [1.1, 1.2],
                [2.1, 2.2],
            ])

            utils.generate_submission(region_ids, preds, os.path.join(output_dir, "submission.csv"), timestamp="20260518_143012")

            sub = pd.read_csv(os.path.join(output_dir, "submission_20260518_143012.csv"))
            self.assertEqual(list(sub.columns), ["region_id", "pred_week1", "pred_week2"])
            self.assertEqual(sub["region_id"].tolist(), ["R1", "R2", "R3"])
            self.assertEqual(sub["pred_week1"].tolist(), [1.1, 2.1, 3.1])
            self.assertEqual(sub["pred_week2"].tolist(), [1.2, 2.2, 3.2])


if __name__ == "__main__":
    unittest.main()
