"""Cache hybrid ordinal class probabilities with OOF temperature scaling."""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments
from model.ordinal_calibration import fit_predict_ordinal_class_probs_temperature
from model.ordinal_features import load_ordinal_train_test
from model.probability_blend import save_prob_cache
from scripts.cache_ordinal_probabilities import SMOKE_CLASSIFIER_PARAMS

DEFAULT_OUTPUT = "output/prob_cache/ordinal_hybrid_oofcal_best.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ordinal hybrid with OOF temperature calibration."
    )
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--feature-set", default="hybrid", choices=("hybrid",))
    parser.add_argument("--max-windows-per-region", type=int, default=52)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    experiments.set_thread_limits(4)
    args = parse_args()
    max_windows = 2 if args.smoke else args.max_windows_per_region
    classifier_params = SMOKE_CLASSIFIER_PARAMS if args.smoke else None

    print(
        f"Loading hybrid ordinal features (max_windows={max_windows}, smoke={args.smoke})..."
    )
    X_train, y_train, train_regions, X_test, test_regions = load_ordinal_train_test(
        args.train_path,
        args.test_path,
        max_windows_per_region=max_windows,
        feature_set=args.feature_set,
    )

    print("OOF temperature calibration + test prediction...")
    class_probs, meta = fit_predict_ordinal_class_probs_temperature(
        X_train,
        y_train,
        X_test,
        train_regions,
        n_splits=args.n_splits,
        classifier_params=classifier_params,
    )

    print(f"OOF MAE before calibration: {meta['oof_mae_before']:.6f}")
    print(f"OOF MAE after calibration:  {meta['oof_mae_after']:.6f}")
    print(f"Temperature grid mean: {meta['temperatures'].mean():.4f}")
    print(f"Temperature range: [{meta['temperatures'].min():.4f}, {meta['temperatures'].max():.4f}]")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_prob_cache(
        args.output_path,
        class_probs,
        test_regions,
        source="ordinal_hybrid_oofcal",
        metadata={
            "max_windows_per_region": max_windows,
            "feature_set": args.feature_set,
            "oof_mae_before": meta["oof_mae_before"],
            "oof_mae_after": meta["oof_mae_after"],
            "n_splits": args.n_splits,
            "temperatures": meta["temperatures"],
        },
    )
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
