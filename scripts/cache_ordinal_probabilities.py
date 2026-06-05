"""Train ordinal threshold classifiers and cache test class probabilities."""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments
from model.ordinal_features import ORDINAL_FEATURE_SETS, load_ordinal_train_test
from model.ordinal_tree import fit_predict_ordinal_class_probs
from model.probability_blend import save_prob_cache

SOURCE_BY_FEATURE_SET = {
    "hybrid": "ordinal_hybrid",
    "hybrid_season": "ordinal_season_hybrid",
    "hybrid_blackout": "ordinal_blackout_hybrid",
    "history_only": "ordinal_history_only",
}

SMOKE_CLASSIFIER_PARAMS = {
    "n_estimators": 15,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cache ordinal tree class probabilities.")
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument(
        "--feature-set",
        choices=ORDINAL_FEATURE_SETS,
        default="hybrid",
        help=(
            "hybrid = temporal only; hybrid_season = temporal + same-season; "
            "hybrid_blackout = temporal + blackout severity history; "
            "history_only = blackout severity history without meteorology."
        ),
    )
    parser.add_argument("--max-windows-per-region", type=int, default=52)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast run: 2 windows/region and fewer trees.",
    )
    parser.add_argument("--output-dir", default="output/prob_cache")
    parser.add_argument("--output-path")
    parser.add_argument("--name", default=None, help="Defaults to ordinal_<feature_set>.")
    return parser.parse_args()


def main():
    experiments.set_thread_limits(4)
    args = parse_args()

    max_windows = 2 if args.smoke else args.max_windows_per_region
    classifier_params = SMOKE_CLASSIFIER_PARAMS if args.smoke else None

    print(
        f"Loading features (feature_set={args.feature_set}, "
        f"max_windows={max_windows}, smoke={args.smoke})..."
    )
    X_train, y_train, _, X_test, test_regions = load_ordinal_train_test(
        args.train_path,
        args.test_path,
        max_windows_per_region=max_windows,
        feature_set=args.feature_set,
    )
    print(f"Feature dims: train={X_train.shape[1]}, test rows={len(X_test)}")

    print("Training ordinal classifiers and predicting test probabilities...")
    class_probs = fit_predict_ordinal_class_probs(
        X_train, y_train, X_test, classifier_params=classifier_params
    )

    if args.output_path:
        out_path = args.output_path
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        name = args.name or SOURCE_BY_FEATURE_SET[args.feature_set]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"{name}_{timestamp}.npz")

    save_prob_cache(
        out_path,
        class_probs,
        test_regions,
        source=SOURCE_BY_FEATURE_SET[args.feature_set],
        metadata={
            "max_windows_per_region": max_windows,
            "feature_set": args.feature_set,
            "smoke": args.smoke,
        },
    )
    print(f"Saved ordinal probability cache: {out_path}")
    print(f"shape={class_probs.shape}")


if __name__ == "__main__":
    main()
