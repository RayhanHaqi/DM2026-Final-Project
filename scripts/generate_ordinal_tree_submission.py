import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments, temporal_features
from model.ordinal_tree import threshold_probs_to_class_probs, expected_value_from_class_probs
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


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    X_train, y_train, train_regions = temporal_features.load_temporal_train_data(
        "data/train.csv", max_windows_per_region=52, feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data("data/test.csv", feature_set="hybrid")
    sample = pd.read_csv("data/sample_submission.csv")

    thresholds = [1, 2, 3, 4, 5]
    all_threshold_preds = []

    for week_idx in range(5):
        week_preds = np.zeros((len(X_test), 5), dtype=float)
        for thresh_idx, thresh in enumerate(thresholds):
            y_binary = (y_train[:, week_idx] >= thresh).astype(int)
            clf = XGBClassifier(**DEFAULT_CLASSIFIER_PARAMS)
            clf.fit(X_train, y_binary)
            week_preds[:, thresh_idx] = clf.predict_proba(X_test)[:, 1]
        all_threshold_preds.append(week_preds)

    week_class_probs = []
    for week_preds in all_threshold_preds:
        class_probs = threshold_probs_to_class_probs(week_preds)
        week_class_probs.append(class_probs)

    preds = np.column_stack([expected_value_from_class_probs(cp) for cp in week_class_probs])
    preds = np.clip(preds, 0.0, 5.0)

    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise SystemExit("; ".join(messages))

    out_path = "output/daily_candidates/ordinal_tree_expected_20260524.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved ordinal tree submission: {out_path}")


if __name__ == "__main__":
    main()
