import numpy as np

from model.ordinal_tree import threshold_probs_to_class_probs, expected_value_from_class_probs


def test_threshold_probs_to_class_probs_produces_six_classes():
    threshold_probs = np.array([[0.8, 0.6, 0.3, 0.1, 0.0]])

    class_probs = threshold_probs_to_class_probs(threshold_probs)

    np.testing.assert_allclose(class_probs.sum(axis=1), [1.0])
    np.testing.assert_allclose(class_probs[0], [0.2, 0.2, 0.3, 0.2, 0.1, 0.0])


def test_expected_value_from_class_probs_uses_score_ids():
    class_probs = np.array([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0]])

    expected = expected_value_from_class_probs(class_probs)
    np.testing.assert_allclose(expected, [1.5])
