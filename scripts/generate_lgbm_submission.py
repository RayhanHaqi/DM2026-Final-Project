import json
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import experiments, temporal_features
from model.same_season import build_train_same_season_features, build_test_same_season_features
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

THREADS = 2
MAX_WINDOWS = 52
N_SPLITS = 5


def run_lgbm_cv(X, y, regions, params, n_splits=N_SPLITS):
    """Run GroupKFold CV with LightGBM, return mean MAE."""
    kf = GroupKFold(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in kf.split(X, y, regions):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]

        preds = np.zeros((len(val_idx), y.shape[1]))
        for week_idx in range(y.shape[1]):
            model = LGBMRegressor(**params)
            model.fit(X_tr, y[train_idx, week_idx])
            preds[:, week_idx] = model.predict(X_val)

        scores.append(mean_absolute_error(y[val_idx], preds))

    return float(np.mean(scores))


def generate_lgbm_submission(X_train, y_train, X_test, test_regions, params, sample, out_path):
    """Train LightGBM, generate submission CSV."""
    preds = np.zeros((len(X_test), y_train.shape[1]))
    for week_idx in range(y_train.shape[1]):
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train[:, week_idx])
        preds[:, week_idx] = model.predict(X_test)

    preds = np.clip(preds, 0.0, 5.0)

    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise ValueError(f"Validation failed: {'; '.join(messages)}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub.to_csv(out_path, index=False)
    return out_path


def main():
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)

    train_path = "data/train.csv"
    test_path = "data/test.csv"
    sample_path = "data/sample_submission.csv"

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    print("Loading hybrid features...")
    X_train, y_train, train_regions = temporal_features.load_temporal_train_data(
        train_path, max_windows_per_region=MAX_WINDOWS, feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(test_path, feature_set="hybrid")

    print("Building same-season features...")
    season_train = build_train_same_season_features(train_df, max_windows_per_region=MAX_WINDOWS)
    season_test = build_test_same_season_features(train_df, test_df)

    X_train_full = pd.concat([X_train.reset_index(drop=True), season_train], axis=1)
    X_test_full = pd.concat([X_test.reset_index(drop=True), season_test], axis=1)

    # LightGBM param grid
    param_grid = []
    for num_leaves, max_depth, learning_rate, n_estimators in product(
        [31, 50, 70],
        [4, 5, 6],
        [0.03, 0.04, 0.05],
        [200, 250, 300],
    ):
        param_grid.append({
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "n_jobs": THREADS,
            "verbose": -1,
            "random_state": 42,
        })

    print(f"\nRunning LightGBM grid search ({len(param_grid)} combos)...")
    results = []
    for params in tqdm(param_grid, desc="LightGBM CV"):
        mae = run_lgbm_cv(X_train_full, y_train, train_regions, params)
        results.append((params, mae))

    results.sort(key=lambda x: x[1])

    # Log results
    os.makedirs("output/grid_search", exist_ok=True)
    log_rows = []
    for params, mae in results:
        row = {"mae": mae}
        row.update(params)
        log_rows.append(row)
    pd.DataFrame(log_rows).to_csv("output/grid_search/lgbm_grid_search_log.csv", index=False)

    # Generate best submission
    best_params = results[0][0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"output/daily_candidates/lgbm_best_{timestamp}.csv"
    generate_lgbm_submission(X_train_full, y_train, X_test_full, test_regions, best_params, sample, out_path)

    print(f"\nBest LightGBM: MAE={results[0][1]:.6f}")
    print(f"Best params: {best_params}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
