import tempfile
import unittest

import pandas as pd

from scripts import blend_submissions


def make_submission(path, region_ids=("R1", "R2"), offset=0.0):
    df = pd.DataFrame({
        "region_id": list(region_ids),
        "pred_week1": [1.0 + offset, 3.0 + offset],
        "pred_week2": [2.0 + offset, 4.0 + offset],
    })
    df.to_csv(path, index=False)
    return df


class BlendSubmissionTests(unittest.TestCase):
    def test_blend_submissions_uses_candidate_weight_and_reference_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = f"{tmpdir}/candidate.csv"
            reference_path = f"{tmpdir}/reference.csv"
            make_submission(candidate_path, offset=2.0)
            reference = make_submission(reference_path, offset=0.0)

            blended = blend_submissions.blend_submission_frames(candidate_path, reference_path, candidate_weight=0.25)

        self.assertEqual(blended["region_id"].tolist(), reference["region_id"].tolist())
        self.assertAlmostEqual(blended.loc[0, "pred_week1"], 1.5)
        self.assertAlmostEqual(blended.loc[1, "pred_week2"], 4.5)

    def test_blend_submissions_clips_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = f"{tmpdir}/candidate.csv"
            reference_path = f"{tmpdir}/reference.csv"
            pd.DataFrame({"region_id": ["R1"], "pred_week1": [9.0], "pred_week2": [-3.0]}).to_csv(candidate_path, index=False)
            pd.DataFrame({"region_id": ["R1"], "pred_week1": [9.0], "pred_week2": [-3.0]}).to_csv(reference_path, index=False)

            blended = blend_submissions.blend_submission_frames(candidate_path, reference_path, candidate_weight=0.5)

        self.assertEqual(blended.loc[0, "pred_week1"], 5.0)
        self.assertEqual(blended.loc[0, "pred_week2"], 0.0)

    def test_blend_submissions_rejects_mismatched_id_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = f"{tmpdir}/candidate.csv"
            reference_path = f"{tmpdir}/reference.csv"
            make_submission(candidate_path, region_ids=("R2", "R1"))
            make_submission(reference_path, region_ids=("R1", "R2"))

            with self.assertRaisesRegex(ValueError, "ID order"):
                blend_submissions.blend_submission_frames(candidate_path, reference_path, candidate_weight=0.25)

    def test_write_blend_submission_uses_explicit_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = f"{tmpdir}/candidate.csv"
            reference_path = f"{tmpdir}/reference.csv"
            output_path = f"{tmpdir}/explicit.csv"
            make_submission(candidate_path, offset=2.0)
            make_submission(reference_path, offset=0.0)

            written_path = blend_submissions.write_blend_submission(
                candidate_path,
                reference_path,
                candidate_weight=0.25,
                output_path=output_path,
            )

            saved = pd.read_csv(output_path)

        self.assertEqual(written_path, output_path)
        self.assertAlmostEqual(saved.loc[0, "pred_week1"], 1.5)
        self.assertAlmostEqual(saved.loc[1, "pred_week2"], 4.5)


if __name__ == "__main__":
    unittest.main()
