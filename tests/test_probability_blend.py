import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from model.probability_blend import (
    blend_class_probs,
    class_probs_to_predictions,
    load_prob_cache,
    parse_weighted_cache_specs,
    reorder_class_probs,
    save_prob_cache,
    soft_probs_from_regression,
)
from scripts import blend_prob_submissions


class ProbabilityBlendTests(unittest.TestCase):
  def test_soft_probs_from_regression_preserves_expected_value(self):
    probs = soft_probs_from_regression(np.array([2.3]))[0, 0]
    expected = np.dot(probs, np.arange(6))
    self.assertAlmostEqual(probs.sum(), 1.0)
    self.assertAlmostEqual(expected, 2.3)
    self.assertAlmostEqual(probs[2], 0.7)
    self.assertAlmostEqual(probs[3], 0.3)

  def test_soft_probs_from_regression_temperature_spreads_mass(self):
    probs = soft_probs_from_regression(np.array([2.0]), temperature=0.5)[0, 0]
    self.assertAlmostEqual(probs.sum(), 1.0)
    self.assertGreater(probs[1], 0.0)
    self.assertGreater(probs[3], 0.0)

  def test_blend_class_probs_normalizes_weights(self):
    probs_a = np.zeros((2, 1, 6), dtype=float)
    probs_a[:, :, 2] = 1.0
    probs_b = np.zeros((2, 1, 6), dtype=float)
    probs_b[:, :, 4] = 1.0

    blended = blend_class_probs([probs_a, probs_b], [1.0, 1.0])

    np.testing.assert_allclose(blended[:, 0, 2], [0.5, 0.5])
    np.testing.assert_allclose(blended[:, 0, 4], [0.5, 0.5])

  def test_class_probs_to_predictions_uses_expected_value(self):
    class_probs = np.zeros((1, 2, 6), dtype=float)
    class_probs[0, 0, 1] = 0.5
    class_probs[0, 0, 2] = 0.5
    class_probs[0, 1, 4] = 1.0

    preds = class_probs_to_predictions(class_probs)

    np.testing.assert_allclose(preds, [[1.5, 4.0]])

  def test_prob_cache_roundtrip(self):
    class_probs = np.zeros((3, 5, 6), dtype=float)
    class_probs[:, :, 0] = 1.0
    region_ids = ["R1", "R2", "R3"]

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, "cache.npz")
      save_prob_cache(path, class_probs, region_ids, source="test")
      loaded = load_prob_cache(path)

    np.testing.assert_allclose(loaded["class_probs"], class_probs)
    self.assertEqual(loaded["region_ids"], region_ids)
    self.assertEqual(loaded["source"], "test")

  def test_parse_weighted_cache_specs(self):
    parsed = parse_weighted_cache_specs(["a.npz:0.2", "b.npz:0.8"])
    self.assertEqual(parsed, [("a.npz", 0.2), ("b.npz", 0.8)])

  def test_reorder_class_probs_follows_target_region_ids(self):
    class_probs = np.zeros((2, 1, 6), dtype=float)
    class_probs[0, 0, 1] = 1.0
    class_probs[1, 0, 3] = 1.0

    reordered = reorder_class_probs(class_probs, ["R1", "R2"], ["R2", "R1"])

    self.assertAlmostEqual(reordered[0, 0, 3], 1.0)
    self.assertAlmostEqual(reordered[1, 0, 1], 1.0)

  def test_blend_prob_caches_aligns_different_region_orders(self):
    probs_a = np.zeros((2, 1, 6), dtype=float)
    probs_a[0, 0, 1] = 1.0
    probs_a[1, 0, 3] = 1.0
    probs_b = np.zeros((2, 1, 6), dtype=float)
    probs_b[0, 0, 2] = 1.0
    probs_b[1, 0, 4] = 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
      cache_a = os.path.join(tmpdir, "a.npz")
      cache_b = os.path.join(tmpdir, "b.npz")
      save_prob_cache(cache_a, probs_a, ["R1", "R2"], source="a")
      save_prob_cache(cache_b, probs_b, ["R2", "R1"], source="b")

      blended, region_ids, _ = blend_prob_submissions.blend_prob_caches(
          [f"{cache_a}:0.5", f"{cache_b}:0.5"],
          ["R1", "R2"],
      )

    self.assertEqual(region_ids, ["R1", "R2"])
    self.assertAlmostEqual(blended[0, 0, 1], 0.5)
    self.assertAlmostEqual(blended[0, 0, 4], 0.5)
    self.assertAlmostEqual(blended[1, 0, 2], 0.5)
    self.assertAlmostEqual(blended[1, 0, 3], 0.5)

  def test_write_prob_blend_submission_matches_sample_order(self):
    probs_a = np.zeros((2, 2, 6), dtype=float)
    probs_a[:, :, 1] = 1.0
    probs_b = np.zeros((2, 2, 6), dtype=float)
    probs_b[:, :, 3] = 1.0
    region_ids = ["R1", "R2"]

    with tempfile.TemporaryDirectory() as tmpdir:
      cache_a = os.path.join(tmpdir, "a.npz")
      cache_b = os.path.join(tmpdir, "b.npz")
      sample_path = os.path.join(tmpdir, "sample.csv")
      output_path = os.path.join(tmpdir, "blend.csv")

      save_prob_cache(cache_a, probs_a, region_ids, source="a")
      save_prob_cache(cache_b, probs_b, region_ids, source="b")
      pd.DataFrame({
        "region_id": ["R2", "R1"],
        "pred_week1": [0.0, 0.0],
        "pred_week2": [0.0, 0.0],
      }).to_csv(sample_path, index=False)

      blend_prob_submissions.write_prob_blend_submission(
          [f"{cache_a}:0.5", f"{cache_b}:0.5"],
          sample_path,
          output_path,
      )
      saved = pd.read_csv(output_path)

    self.assertEqual(saved["region_id"].tolist(), ["R2", "R1"])
    self.assertAlmostEqual(saved.loc[saved["region_id"] == "R1", "pred_week1"].iloc[0], 2.0)
    self.assertAlmostEqual(saved.loc[saved["region_id"] == "R2", "pred_week2"].iloc[0], 2.0)


if __name__ == "__main__":
  unittest.main()
