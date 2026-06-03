import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments, temporal_features
from model.ordinal_tree import expected_values_from_week_class_probs, fit_predict_ordinal_class_probs


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    X_train, y_train, _train_regions = temporal_features.load_temporal_train_data(
        "data/train.csv", max_windows_per_region=52, feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data("data/test.csv", feature_set="hybrid")
    sample = pd.read_csv("data/sample_submission.csv")

    class_probs = fit_predict_ordinal_class_probs(X_train, y_train, X_test)
    preds = expected_values_from_week_class_probs(class_probs)

    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise SystemExit("; ".join(messages))

    out_path = "output/daily_candidates/ordinal_tree_expected_20260524.csv"
    sub.to_csv(out_path, index=False)
    print(f"Saved ordinal tree submission: {out_path}")


if __name__ == "__main__":
    main()
