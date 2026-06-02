import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments, severity_history, temporal_features, temporal_tree


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
    window_days = 91

    train_df = pd.read_csv(train_path)

    X_train, y_train, train_regions = temporal_features.load_temporal_train_data(
        train_path,
        max_windows_per_region=max_windows_per_region,
        feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(test_path, feature_set="hybrid")

    train_history = severity_history.build_train_blackout_history_features_from_frame(
        train_df,
        max_windows_per_region=max_windows_per_region,
        window_days=window_days,
    )
    test_history = severity_history.build_test_blackout_history_features_from_frame(
        train_df, test_regions, window_days=window_days,
    )

    X_train = _with_history_features(X_train, train_history)
    X_test = _with_history_features(X_test, test_history)

    cv_scores, cv_mean, cv_std = temporal_tree.cv_evaluate_week_models(
        X_train, y_train, train_regions, n_splits=3,
    )
    print(f"cv_mae mean={cv_mean:.6f} std={cv_std:.6f}")

    models = temporal_tree.train_week_models(X_train, y_train)
    preds = np.clip(temporal_tree.predict_week_models(models, X_test), 0.0, 5.0)

    sample = pd.read_csv(sample_path)
    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise SystemExit("; ".join(messages))

    out_path = "output/daily_candidates/history_blackout_temporal_tree_20260526.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"Saved history blackout temporal tree: {out_path}")


if __name__ == "__main__":
    main()
