import json
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import experiments, temporal_features, temporal_tree
from model.same_season import build_train_same_season_features, build_test_same_season_features
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

THREADS = 2
MAX_WINDOWS = 52
N_SPLITS = 5
OUTPUT_DIR = "output/grid_search"


def generate_param_grid_stage_a():
    """Stage A: Sweep depth, trees, lr with fixed regularization."""
    grid = []
    for max_depth, n_estimators, learning_rate in product(
        [3, 4, 5, 6],
        [200, 250, 300],
        [0.03, 0.04, 0.05],
    ):
        grid.append({
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "random_state": 42,
            "n_jobs": THREADS,
        })
    return grid


def generate_param_grid_stage_b(best_depth, best_trees, best_lr):
    """Stage B: Sweep subsample and colsample with best from A."""
    grid = []
    for subsample, colsample in product([0.8, 0.85, 0.9], [0.8, 0.85, 0.9]):
        grid.append({
            "max_depth": best_depth,
            "n_estimators": best_trees,
            "learning_rate": best_lr,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "reg_alpha": 0.2,
            "reg_lambda": 1.5,
            "random_state": 42,
            "n_jobs": THREADS,
        })
    return grid


def generate_param_grid_stage_c(best_params):
    """Stage C: Sweep regularization with best from A+B."""
    grid = []
    for reg_alpha, reg_lambda in product([0.1, 0.2, 0.3], [1.0, 1.5, 2.0]):
        params = dict(best_params)
        params["reg_alpha"] = reg_alpha
        params["reg_lambda"] = reg_lambda
        grid.append(params)
    return grid


def generate_param_grid_stage_d(best_params):
    """Stage D: Refinement around best from A+B+C."""
    variations = {
        "max_depth": [-1, 0, 1],
        "n_estimators": [-50, 0, 50],
        "learning_rate": [-0.01, 0, 0.01],
        "subsample": [-0.05, 0, 0.05],
        "colsample_bytree": [-0.05, 0, 0.05],
        "reg_alpha": [-0.1, 0, 0.1],
        "reg_lambda": [-0.5, 0, 0.5],
    }

    grid = []
    seen = set()
    for key, deltas in variations.items():
        for delta in deltas:
            if delta == 0:
                continue
            params = dict(best_params)
            if key in ("max_depth", "n_estimators"):
                params[key] = max(1, best_params[key] + delta)
            else:
                params[key] = round(max(0.01, best_params[key] + delta), 4)
            # Deduplicate
            sig = tuple(sorted(params.items()))
            if sig not in seen:
                seen.add(sig)
                grid.append(params)

    # Also add the best params itself
    sig = tuple(sorted(best_params.items()))
    if sig not in seen:
        grid.append(dict(best_params))
    return grid


def run_single_cv(X, y, regions, params, n_splits=N_SPLITS):
    """Run GroupKFold CV, return mean MAE."""
    kf = GroupKFold(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in kf.split(X, y, regions):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]

        models = temporal_tree.train_week_models(X_tr, y[train_idx], params_override=params)
        preds = temporal_tree.predict_week_models(models, X_val)
        scores.append(mean_absolute_error(y[val_idx], preds))

    return float(np.mean(scores))


def run_stage(X, y, regions, param_grid, fixed_params, n_splits=N_SPLITS, desc="Stage"):
    """Run all combinations in a stage, return sorted (params, mae) list."""
    results = []

    for params in tqdm(param_grid, desc=desc):
        full_params = dict(fixed_params)
        full_params.update(params)
        mae = run_single_cv(X, y, regions, full_params, n_splits=n_splits)
        results.append((full_params, mae))

    results.sort(key=lambda x: x[1])
    return results


def generate_submission_from_params(X_train, y_train, X_test, test_regions, params, sample, out_path):
    """Train full model with params, generate submission CSV."""
    models = temporal_tree.train_week_models(X_train, y_train, params_override=params)
    preds = temporal_tree.predict_week_models(models, X_test)
    preds = np.clip(preds, 0.0, 5.0)

    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise ValueError(f"Submission validation failed: {'; '.join(messages)}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sub.to_csv(out_path, index=False)
    return out_path


def main():
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    all_results = []

    # Stage A: depth, trees, lr
    print("\n=== Stage A: depth, trees, lr (36 combos) ===")
    grid_a = generate_param_grid_stage_a()
    results_a = run_stage(X_train_full, y_train, train_regions, grid_a, {}, desc="Stage A")
    all_results.extend(results_a)
    best_a = results_a[0][0]
    print(f"Best Stage A: MAE={results_a[0][1]:.6f}, params={best_a}")

    # Stage B: subsample, colsample
    print("\n=== Stage B: subsample, colsample (9 combos) ===")
    grid_b = generate_param_grid_stage_b(best_a["max_depth"], best_a["n_estimators"], best_a["learning_rate"])
    results_b = run_stage(X_train_full, y_train, train_regions, grid_b, {}, desc="Stage B")
    all_results.extend(results_b)
    best_b = results_b[0][0]
    print(f"Best Stage B: MAE={results_b[0][1]:.6f}, params={best_b}")

    # Stage C: regularization
    print("\n=== Stage C: reg_alpha, reg_lambda (9 combos) ===")
    grid_c = generate_param_grid_stage_c(best_b)
    results_c = run_stage(X_train_full, y_train, train_regions, grid_c, {}, desc="Stage C")
    all_results.extend(results_c)
    best_c = results_c[0][0]
    print(f"Best Stage C: MAE={results_c[0][1]:.6f}, params={best_c}")

    # Stage D: refinement
    print("\n=== Stage D: refinement ===")
    grid_d = generate_param_grid_stage_d(best_c)
    print(f"Stage D: {len(grid_d)} combos")
    results_d = run_stage(X_train_full, y_train, train_regions, grid_d, {}, desc="Stage D")
    all_results.extend(results_d)

    # Sort all results
    all_results.sort(key=lambda x: x[1])

    # Log all results
    log_path = os.path.join(OUTPUT_DIR, "grid_search_log.csv")
    log_rows = []
    for params, mae in all_results:
        row = {"mae": mae}
        row.update(params)
        log_rows.append(row)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"\nSaved log: {log_path}")

    # Save best params
    best_params = all_results[0][0]
    best_path = os.path.join(OUTPUT_DIR, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best params: {best_path}")

    # Generate top 3 submissions
    print("\nGenerating top 3 submissions...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, (params, mae) in enumerate(all_results[:3]):
        out_path = os.path.join("output/daily_candidates", f"grid_search_top{i+1}_{timestamp}.csv")
        generate_submission_from_params(X_train_full, y_train, X_test_full, test_regions, params, sample, out_path)
        print(f"Top {i+1}: MAE={mae:.6f} -> {out_path}")

    print("\n=== Grid Search Complete ===")
    print(f"Best MAE: {all_results[0][1]:.6f}")
    print(f"Best params: {all_results[0][0]}")


if __name__ == "__main__":
    main()
