import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import experiments, temporal_features, temporal_tree
from model.same_season import build_train_same_season_features, build_test_same_season_features


def main():
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"

    train_path = "data/train.csv"
    test_path = "data/test.csv"
    sample_path = "data/sample_submission.csv"
    max_windows = 52

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Loading hybrid features...")
    X_train, y_train, train_regions = temporal_features.load_temporal_train_data(
        train_path, max_windows_per_region=max_windows, feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(test_path, feature_set="hybrid")

    print("Building same-season features...")
    season_train = build_train_same_season_features(train_df, max_windows_per_region=max_windows)
    season_test = build_test_same_season_features(train_df, test_df)

    X_train = pd.concat([X_train.reset_index(drop=True), season_train], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), season_test], axis=1)

    params = {
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.2,
        "reg_lambda": 1.5,
        "random_state": 42,
        "n_jobs": 2,
    }

    print("Cross-validating...")
    scores, cv_mean, _ = temporal_tree.cv_evaluate_week_models(X_train, y_train, train_regions, n_splits=5, params_override=params)
    print(f"CV MAE = {cv_mean:.6f} (folds: {[f'{s:.4f}' for s in scores]})")

    print("Training full model...")
    models = temporal_tree.train_week_models(X_train, y_train, params_override=params)

    print("Predicting test...")
    preds = temporal_tree.predict_week_models(models, X_test)
    preds = np.clip(preds, 0.0, 5.0)

    sample = pd.read_csv(sample_path)
    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise SystemExit("; ".join(messages))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"output/daily_candidates/season_tree_{timestamp}.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  CV={cv_mean:.6f}")


if __name__ == "__main__":
    main()
