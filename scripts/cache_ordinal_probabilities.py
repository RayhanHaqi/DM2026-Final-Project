"""Train ordinal threshold classifiers and cache test class probabilities."""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import temporal_features
from model.ordinal_tree import fit_predict_ordinal_class_probs
from model.probability_blend import save_prob_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Cache ordinal tree class probabilities.")
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--max-windows-per-region", type=int, default=52)
    parser.add_argument("--output-dir", default="output/prob_cache")
    parser.add_argument("--output-path")
    parser.add_argument("--name", default="ordinal_hybrid")
    return parser.parse_args()


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    args = parse_args()
    X_train, y_train, _ = temporal_features.load_temporal_train_data(
        args.train_path,
        max_windows_per_region=args.max_windows_per_region,
        feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(args.test_path, feature_set="hybrid")

    print("Training ordinal classifiers and predicting test probabilities...")
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
        source="ordinal_hybrid",
        metadata={"max_windows_per_region": args.max_windows_per_region},
    )
    print(f"Saved ordinal probability cache: {out_path}")
    print(f"shape={class_probs.shape}")


if __name__ == "__main__":
    main()
