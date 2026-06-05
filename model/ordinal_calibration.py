"""OOF temperature scaling for ordinal cumulative-threshold probabilities."""

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from model.ordinal_tree import (
    DEFAULT_CLASSIFIER_PARAMS,
    ORDINAL_THRESHOLDS,
    expected_value_from_class_probs,
    threshold_probs_to_class_probs,
)

TEMPERATURE_GRID = np.linspace(0.25, 4.0, 76)
LOGIT_EPS = 1e-6


def logit(probs):
    p = np.clip(np.asarray(probs, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.clip(1.0 / (1.0 + np.exp(-x)), LOGIT_EPS, 1.0 - LOGIT_EPS)


def apply_temperature_threshold_probs(probs, temperature):
    """Platt-style temperature on probabilities: sigmoid(logit(p) / T)."""
    temp = max(float(temperature), LOGIT_EPS)
    return sigmoid(logit(probs) / temp)


def fit_temperature_scalar(probs, labels, grid=None):
    """Fit one temperature minimizing binary NLL on (probs, labels)."""
    probs = np.asarray(probs, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    if len(probs) != len(labels):
        raise ValueError("probs and labels length mismatch")
    if len(probs) == 0:
        return 1.0

    grid = TEMPERATURE_GRID if grid is None else np.asarray(grid, dtype=float)
    best_t = 1.0
    best_nll = np.inf
    for temp in grid:
        p_cal = apply_temperature_threshold_probs(probs, temp)
        nll = -np.mean(
            labels * np.log(p_cal) + (1.0 - labels) * np.log(1.0 - p_cal)
        )
        if nll < best_nll:
            best_nll = nll
            best_t = float(temp)
    return best_t


def fit_threshold_temperatures(oof_threshold_probs, y_train):
    """Per (week, threshold) temperatures from OOF cumulative probabilities."""
    oof = np.asarray(oof_threshold_probs, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n_weeks = y.shape[1]
    n_thresh = len(ORDINAL_THRESHOLDS)
    temps = np.ones((n_weeks, n_thresh), dtype=float)

    for week_idx in range(n_weeks):
        for thresh_idx, thresh in enumerate(ORDINAL_THRESHOLDS):
            labels = (y[:, week_idx] >= thresh).astype(int)
            temps[week_idx, thresh_idx] = fit_temperature_scalar(
                oof[:, week_idx, thresh_idx],
                labels,
            )
    return temps


def apply_threshold_temperatures(threshold_probs, temperatures):
    """Apply per-(week,threshold) temperatures to threshold_probs (..., n_weeks, n_thresh)."""
    arr = np.asarray(threshold_probs, dtype=float)
    temps = np.asarray(temperatures, dtype=float)
    out = np.empty_like(arr)
    n_weeks, n_thresh = temps.shape
    for week_idx in range(n_weeks):
        for thresh_idx in range(n_thresh):
            out[..., week_idx, thresh_idx] = apply_temperature_threshold_probs(
                arr[..., week_idx, thresh_idx],
                temps[week_idx, thresh_idx],
            )
    return out


def expected_scores_from_threshold_probs(threshold_probs):
    """(n, n_weeks, n_thresh) -> (n, n_weeks) expected scores."""
    arr = np.asarray(threshold_probs, dtype=float)
    n_samples, n_weeks, _ = arr.shape
    scores = np.zeros((n_samples, n_weeks), dtype=float)
    for week_idx in range(n_weeks):
        class_probs = threshold_probs_to_class_probs(arr[:, week_idx, :])
        scores[:, week_idx] = expected_value_from_class_probs(class_probs)
    return scores


def collect_oof_threshold_probabilities(
    X,
    y,
    groups,
    n_splits=5,
    classifier_params=None,
):
    """GroupKFold OOF cumulative threshold probabilities, shape (n, 5, 5)."""
    params = DEFAULT_CLASSIFIER_PARAMS if classifier_params is None else classifier_params
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    n_samples, n_weeks = y.shape
    n_thresh = len(ORDINAL_THRESHOLDS)
    oof = np.full((n_samples, n_weeks, n_thresh), np.nan, dtype=float)
    kf = GroupKFold(n_splits=n_splits)

    for train_idx, val_idx in kf.split(X, y, groups):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        X_val_arr = X_val.values if hasattr(X_val, "values") else X_val

        for week_idx in range(n_weeks):
            for thresh_idx, thresh in enumerate(ORDINAL_THRESHOLDS):
                y_binary = (y[train_idx, week_idx] >= thresh).astype(int)
                clf = XGBClassifier(**params)
                clf.fit(X_tr, y_binary)
                oof[val_idx, week_idx, thresh_idx] = clf.predict_proba(X_val_arr)[:, 1]

    if np.isnan(oof).any():
        raise ValueError("OOF threshold probabilities contain NaNs")
    return oof


def _predict_test_threshold_probs(X_train, y_train, X_test, classifier_params=None):
    params = DEFAULT_CLASSIFIER_PARAMS if classifier_params is None else classifier_params
    X_test_arr = X_test.values if hasattr(X_test, "values") else X_test
    y = np.asarray(y_train, dtype=float)
    n_test = len(X_test_arr)
    n_weeks = y.shape[1]
    n_thresh = len(ORDINAL_THRESHOLDS)
    test_thresh = np.zeros((n_test, n_weeks, n_thresh), dtype=float)

    for week_idx in range(n_weeks):
        for thresh_idx, thresh in enumerate(ORDINAL_THRESHOLDS):
            y_binary = (y[:, week_idx] >= thresh).astype(int)
            clf = XGBClassifier(**params)
            clf.fit(X_train, y_binary)
            test_thresh[:, week_idx, thresh_idx] = clf.predict_proba(X_test_arr)[:, 1]
    return test_thresh


def threshold_probs_to_week_class_probs(threshold_probs):
    """(n, n_weeks, n_thresh) -> (n, n_weeks, 6)."""
    arr = np.asarray(threshold_probs, dtype=float)
    n_samples, n_weeks, _ = arr.shape
    out = np.zeros((n_samples, n_weeks, 6), dtype=float)
    for week_idx in range(n_weeks):
        out[:, week_idx, :] = threshold_probs_to_class_probs(arr[:, week_idx, :])
    return out


def oof_mae_from_threshold_probs(threshold_probs, y_train, temperatures=None):
    """MAE of expected scores from threshold probs (optional temperature apply)."""
    thresh = np.asarray(threshold_probs, dtype=float)
    if temperatures is not None:
        thresh = apply_threshold_temperatures(thresh, temperatures)
    preds = expected_scores_from_threshold_probs(thresh)
    return float(mean_absolute_error(y_train, preds))


def fit_predict_ordinal_class_probs_temperature(
    X_train,
    y_train,
    X_test,
    groups,
    n_splits=5,
    classifier_params=None,
):
    """OOF temperature fit on train, full refit on train, calibrated test class probs."""
    oof_thresh = collect_oof_threshold_probabilities(
        X_train,
        y_train,
        groups,
        n_splits=n_splits,
        classifier_params=classifier_params,
    )
    temperatures = fit_threshold_temperatures(oof_thresh, y_train)
    oof_mae_before = oof_mae_from_threshold_probs(oof_thresh, y_train)
    oof_mae_after = oof_mae_from_threshold_probs(oof_thresh, y_train, temperatures)

    test_thresh = _predict_test_threshold_probs(
        X_train, y_train, X_test, classifier_params=classifier_params
    )
    test_thresh = apply_threshold_temperatures(test_thresh, temperatures)
    class_probs = threshold_probs_to_week_class_probs(test_thresh)

    return class_probs, {
        "oof_mae_before": oof_mae_before,
        "oof_mae_after": oof_mae_after,
        "temperatures": temperatures,
        "n_splits": n_splits,
    }
