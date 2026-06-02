import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments, severity_history, temporal_features
from model.ordinal_tree import expected_value_from_class_probs, threshold_probs_to_class_probs


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


def _with_history_features(X, history):
    if len(X) != len(history):
        raise ValueError("history feature rows must align with temporal feature rows")
    X_reset = X.reset_index(drop=True)
    history_reset = history.reset_index(drop=True).add_prefix("history__")
    return pd.concat([X_reset, history_reset], axis=1)


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    train_path = "data/train.csv"
    test_path = "data/test.csv"
    sample_path = "data/sample_submission.csv"
    max_windows_per_region = 52

    train_df = pd.read_csv(train_path)
    X_train, y_train, _ = temporal_features.load_temporal_train_data(
        train_path,
        max_windows_per_region=max_windows_per_region,
        feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(test_path, feature_set="hybrid")

    train_history = severity_history.build_train_history_features_from_frame(
        train_df,
        max_windows_per_region=max_windows_per_region,
    )
    test_history = severity_history.build_test_history_features_from_frame(train_df, test_regions)
    X_train = _with_history_features(X_train, train_history)
    X_test = _with_history_features(X_test, test_history)

    thresholds = [1, 2, 3, 4, 5]
    week_class_probs = []
    for week_idx in range(5):
        week_threshold_preds = np.zeros((len(X_test), 5), dtype=float)
        for thresh_idx, thresh in enumerate(thresholds):
            y_binary = (y_train[:, week_idx] >= thresh).astype(int)
            clf = XGBClassifier(**DEFAULT_CLASSIFIER_PARAMS)
            clf.fit(X_train, y_binary)
            week_threshold_preds[:, thresh_idx] = clf.predict_proba(X_test)[:, 1]
        week_class_probs.append(threshold_probs_to_class_probs(week_threshold_preds))

    preds = np.column_stack([expected_value_from_class_probs(cp) for cp in week_class_probs])
    preds = np.clip(preds, 0.0, 5.0)

    sample = pd.read_csv(sample_path)
    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise SystemExit("; ".join(messages))

    out_path = "output/daily_candidates/history_ordinal_tree_expected_20260525.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"Saved history ordinal tree submission: {out_path}")


if __name__ == "__main__":
    main()
