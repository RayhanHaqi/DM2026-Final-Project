import numpy as np


def threshold_probs_to_class_probs(threshold_probs):
    probs = np.asarray(threshold_probs, dtype=float)
    probs = np.minimum.accumulate(np.clip(probs, 0.0, 1.0), axis=1)
    class_probs = np.zeros((probs.shape[0], 6), dtype=float)
    class_probs[:, 0] = 1.0 - probs[:, 0]
    for class_id in range(1, 5):
        class_probs[:, class_id] = probs[:, class_id - 1] - probs[:, class_id]
    class_probs[:, 5] = probs[:, 4]
    return np.clip(class_probs, 0.0, 1.0)


def expected_value_from_class_probs(class_probs):
    score_ids = np.arange(6, dtype=float)
    return np.asarray(class_probs, dtype=float) @ score_ids
