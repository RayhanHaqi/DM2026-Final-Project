import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import experiments, temporal_tree
from model.ordinal_features import load_ordinal_train_test


def main():
    experiments.set_thread_limits(2)

    train_path = "data/train.csv"
    test_path = "data/test.csv"
    sample_path = "data/sample_submission.csv"
    max_windows = 52

    print("Loading hybrid + same-season features...")
    X_train, y_train, train_regions, X_test, test_regions = load_ordinal_train_test(
        train_path,
        test_path,
        max_windows_per_region=max_windows,
        feature_set="hybrid_season",
    )

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
    scores, cv_mean, _ = temporal_tree.cv_evaluate_week_models(
        X_train, y_train, train_regions, n_splits=5, params_override=params
    )
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
