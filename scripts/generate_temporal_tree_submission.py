import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import experiments, temporal_features, temporal_tree


def parse_args():
    parser = argparse.ArgumentParser(description="Generate temporal-block tree Kaggle submission.")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    parser.add_argument("--output-dir", default="output/daily_candidates")
    parser.add_argument("--max-windows-per-region", type=int, default=52)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--feature-set", choices=["hybrid", "blocks"], default="hybrid")
    parser.add_argument("--gpu", action="store_true", help="Use XGBoost GPU training with device=cuda.")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.04,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": 2,
    }
    if args.feature_set == "blocks":
        params.update({
            "n_estimators": 180,
            "max_depth": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.3,
            "reg_lambda": 2.0,
        })
    if args.gpu:
        params.update({"tree_method": "hist", "device": "cuda"})

    X_train, y_train, train_regions = temporal_features.load_temporal_train_data(
        args.train,
        max_windows_per_region=args.max_windows_per_region,
        feature_set=args.feature_set,
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(args.test, feature_set=args.feature_set)

    scores, mean_mae, std_mae = [], None, None
    if not args.no_cv:
        scores, mean_mae, std_mae = temporal_tree.cv_evaluate_week_models(
            X_train,
            y_train,
            train_regions,
            n_splits=args.n_splits,
            params_override=params,
        )

    models = temporal_tree.train_week_models(X_train, y_train, params_override=params)
    preds = np.clip(temporal_tree.predict_week_models(models, X_test), 0.0, 5.0)

    sample = pd.read_csv(args.sample)
    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise SystemExit("; ".join(messages))

    os.makedirs(args.output_dir, exist_ok=True)
    name = f"temporal_tree_{args.feature_set}"
    out_path = os.path.join(args.output_dir, f"{name}_{timestamp}.csv")
    summary_path = os.path.join(args.output_dir, f"{name}_{timestamp}_summary.csv")
    sub.to_csv(out_path, index=False)
    pd.DataFrame([{
        "name": name,
        "feature_set": args.feature_set,
        "gpu": args.gpu,
        "cv_mae": mean_mae,
        "cv_std": std_mae,
        "cv_scores": repr(scores),
        "max_windows_per_region": args.max_windows_per_region,
        "params": repr(params),
        "path": out_path,
    }]).to_csv(summary_path, index=False)

    print(f"Saved submission: {out_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
