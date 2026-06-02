import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments, temporal_features, temporal_tree


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    train_path = "data/train.csv"
    test_path = "data/test.csv"
    sample_path = "data/sample_submission.csv"
    best_path = "output/daily_candidates/tree4_ordinal_w015_20260525.csv"
    max_windows_per_region = 52

    X_train, y_train, train_regions = temporal_features.load_temporal_train_data(
        train_path,
        max_windows_per_region=max_windows_per_region,
        feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(test_path, feature_set="hybrid")

    best_sub = pd.read_csv(best_path)
    id_col = best_sub.columns[0]
    target_cols = list(best_sub.columns[1:])
    best_lookup = {rid: best_sub.loc[best_sub[id_col] == rid, target_cols].values[0].astype(float) for rid in best_sub[id_col]}
    best_pred_train = np.vstack([best_lookup[rid] for rid in train_regions])
    best_pred_test = np.vstack([best_lookup[rid] for rid in test_regions])

    residual_y = y_train - best_pred_train

    cv_scores, cv_mean, cv_std = temporal_tree.cv_evaluate_week_models(
        X_train, residual_y, train_regions, n_splits=3,
    )
    print(f"residual_cv_mae mean={cv_mean:.6f} std={cv_std:.6f}")

    models = temporal_tree.train_week_models(X_train, residual_y)
    residual_preds = temporal_tree.predict_week_models(models, X_test)
    preds = np.clip(best_pred_test + residual_preds, 0.0, 5.0)

    sample = pd.read_csv(sample_path)
    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise SystemExit("; ".join(messages))

    out_path = "output/daily_candidates/residual_correction_20260526.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"Saved residual correction: {out_path}")


if __name__ == "__main__":
    main()
