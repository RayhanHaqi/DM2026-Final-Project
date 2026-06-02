import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments, temporal_features, temporal_tree


def parse_args():
    parser = argparse.ArgumentParser(description="Generate quantile/absolute-error tree submissions.")
    parser.add_argument("--variant", choices=["abs", "quantile050", "quantile045", "quantile055"], default="abs")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    parser.add_argument("--output-dir", default="output/daily_candidates")
    parser.add_argument("--max-windows-per-region", type=int, default=52)
    parser.add_argument("--feature-set", choices=["hybrid", "blocks"], default="hybrid")
    return parser.parse_args()


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    args = parse_args()

    params = {
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

    if args.variant == "abs":
        params["objective"] = "reg:absoluteerror"
        params["booster"] = "gbtree"
        out_name = "temporal_tree_absolute_error"
    elif args.variant == "quantile050":
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = 0.5
        out_name = "temporal_tree_quantile_a050"
    elif args.variant == "quantile045":
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = 0.45
        out_name = "temporal_tree_quantile_a045"
    else:
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = 0.55
        out_name = "temporal_tree_quantile_a055"

    X_train, y_train, train_regions = temporal_features.load_temporal_train_data(
        args.train,
        max_windows_per_region=args.max_windows_per_region,
        feature_set=args.feature_set,
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(args.test, feature_set=args.feature_set)

    scores, mean_mae, std_mae = temporal_tree.cv_evaluate_week_models(
        X_train, y_train, train_regions, n_splits=3, params_override=params,
    )
    print(f"cv_mae mean={mean_mae:.6f} std={std_mae:.6f}")

    models = temporal_tree.train_week_models(X_train, y_train, params_override=params)
    preds = np.clip(temporal_tree.predict_week_models(models, X_test), 0.0, 5.0)

    sample = pd.read_csv(args.sample)
    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise SystemExit("; ".join(messages))

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{out_name}_20260524.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved {args.variant} submission: {out_path}")


if __name__ == "__main__":
    main()
