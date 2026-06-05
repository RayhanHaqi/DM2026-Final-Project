import unittest

import numpy as np

from model.ordinal_calibration import (
    apply_temperature_threshold_probs,
    expected_scores_from_threshold_probs,
    fit_temperature_scalar,
    fit_threshold_temperatures,
    oof_mae_from_threshold_probs,
)
from model.ordinal_tree import threshold_probs_to_class_probs


class OrdinalCalibrationTests(unittest.TestCase):
    def test_fit_temperature_near_one_for_well_calibrated(self):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0.05, 0.95, size=500)
        labels = (rng.uniform(0, 1, size=500) < probs).astype(int)
        temp = fit_temperature_scalar(probs, labels)
        self.assertGreater(temp, 0.2)
        self.assertLess(temp, 4.0)

    def test_high_temperature_softens_overconfident_probs(self):
        probs = np.array([0.95, 0.05, 0.9, 0.1])
        softened = apply_temperature_threshold_probs(probs, 2.0)
        self.assertLess(softened[0], probs[0])
        self.assertGreater(softened[1], probs[1])

    def test_oof_mae_after_calibration_on_synthetic(self):
        n = 400
        rng = np.random.default_rng(1)
        y = rng.integers(0, 6, size=(n, 5)).astype(float)
        thresh = np.zeros((n, 5, 5), dtype=float)
        for week in range(5):
            for ti, t in enumerate([1, 2, 3, 4, 5]):
                thresh[:, week, ti] = np.clip((y[:, week] >= t).mean() + 0.35, 0.05, 0.95)
        before = oof_mae_from_threshold_probs(thresh, y)
        temps = fit_threshold_temperatures(thresh, y)
        after = oof_mae_from_threshold_probs(thresh, y, temps)
        self.assertLessEqual(after, before + 0.05)

    def test_expected_scores_shape(self):
        thresh = np.full((3, 5, 5), 0.5)
        scores = expected_scores_from_threshold_probs(thresh)
        self.assertEqual(scores.shape, (3, 5))
        class_direct = threshold_probs_to_class_probs(thresh[:, 0, :])[0]
        self.assertAlmostEqual(scores[0, 0], class_direct @ np.arange(6), places=6)


if __name__ == "__main__":
    unittest.main()
