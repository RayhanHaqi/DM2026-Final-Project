import numpy as np
from xgboost import XGBClassifier

DEFAULT_CLASSIFIER_PARAMS = {
    "n_estimators": 250,
    "max_depth": 4,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.2,
    "reg_lambda": 1.5,
    "random_state": 42,
    "n_jobs": 4,
}

ORDINAL_THRESHOLDS = [1, 2, 3, 4, 5]


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


def expected_values_from_week_class_probs(class_probs):
    """Map (n_samples, n_weeks, 6) class probabilities to expected scores."""
    arr = np.asarray(class_probs, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != 6:
        raise ValueError("class_probs must have shape (n_samples, n_weeks, 6)")
    score_ids = np.arange(6, dtype=float)
    return np.clip(arr @ score_ids, 0.0, 5.0)


def fit_predict_ordinal_class_probs(X_train, y_train, X_test, classifier_params=None):
    """Train cumulative-threshold XGB models per week; return (n_test, 5, 6) class probs."""
    params = DEFAULT_CLASSIFIER_PARAMS if classifier_params is None else classifier_params
    week_class_probs = []

    for week_idx in range(5):
        week_preds = np.zeros((len(X_test), len(ORDINAL_THRESHOLDS)), dtype=float)
        for thresh_idx, thresh in enumerate(ORDINAL_THRESHOLDS):
            y_binary = (y_train[:, week_idx] >= thresh).astype(int)
            clf = XGBClassifier(**params)
            clf.fit(X_train, y_binary)
            week_preds[:, thresh_idx] = clf.predict_proba(X_test)[:, 1]
        week_class_probs.append(threshold_probs_to_class_probs(week_preds))

    return np.stack(week_class_probs, axis=1)
