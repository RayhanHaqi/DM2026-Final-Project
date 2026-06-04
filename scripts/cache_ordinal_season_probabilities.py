"""Train ordinal classifiers on hybrid + same-season features; cache test class probabilities."""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import temporal_features
from model.ordinal_tree import fit_predict_ordinal_class_probs
from model.probability_blend import save_prob_cache
from model.same_season import build_train_same_season_features, build_test_same_season_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache ordinal class probabilities with hybrid + same-season features.",
    )
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--max-windows-per-region", type=int, default=52)
    parser.add_argument("--output-dir", default="output/prob_cache")
    parser.add_argument("--output-path")
    parser.add_argument("--name", default="ordinal_season_hybrid")
    return parser.parse_args()


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    args = parse_args()
    max_windows = args.max_windows_per_region

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    print("Loading hybrid temporal features...")
    X_train, y_train, _ = temporal_features.load_temporal_train_data(
        args.train_path,
        max_windows_per_region=max_windows,
        feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(args.test_path, feature_set="hybrid")

    print("Building same-season features...")
    season_train = build_train_same_season_features(train_df, max_windows_per_region=max_windows)
    season_test = build_test_same_season_features(train_df, test_df)

    if len(season_train) != len(X_train):
        raise ValueError(
            f"Train row mismatch: hybrid {len(X_train)} vs season {len(season_train)}"
        )
    if len(season_test) != len(X_test):
        raise ValueError(
            f"Test row mismatch: hybrid {len(X_test)} vs season {len(season_test)}"
        )

    X_train = pd.concat([X_train.reset_index(drop=True), season_train], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), season_test], axis=1)
    print(f"Feature dims: train={X_train.shape[1]}, test={X_test.shape[1]}")

    print("Training ordinal classifiers (hybrid + season)...")
    class_probs = fit_predict_ordinal_class_probs(X_train, y_train, X_test)

    if args.output_path:
        out_path = args.output_path
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"{args.name}_{timestamp}.npz")

    save_prob_cache(
        out_path,
        class_probs,
        test_regions,
        source="ordinal_season_hybrid",
        metadata={
            "max_windows_per_region": max_windows,
            "season_features": 5,
        },
    )
    print(f"Saved ordinal season probability cache: {out_path}")
    print(f"shape={class_probs.shape}")


if __name__ == "__main__":
    main()
