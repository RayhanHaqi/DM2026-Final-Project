import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"

    best_path = "output/daily_candidates/tree4_ordinal_w015_20260525.csv"
    sample_path = "data/sample_submission.csv"
    recent_label_means = np.load("data/train_y_temporal_hybrid_52.npy").mean(axis=0)

    best = pd.read_csv(best_path)
    sample = pd.read_csv(sample_path)
    id_col = sample.columns[0]
    target_cols = list(sample.columns[1:])

    best_preds = best[target_cols].to_numpy(dtype=float)
    best_means = best_preds.mean(axis=0)

    scale = np.where(best_means != 0.0, recent_label_means / best_means, 1.0)
    bias = recent_label_means - scale * best_means

    calibrated = scale * best_preds + bias
    calibrated = np.clip(calibrated, 0.0, 5.0)

    sub = best.copy()
    sub[target_cols] = calibrated

    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise SystemExit("; ".join(messages))

    out_path = "output/daily_candidates/affine_calibrated_best_20260526.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"Saved affine calibrated submission: {out_path}")
    print(f"scale={np.round(scale, 6).tolist()} bias={np.round(bias, 6).tolist()}")


if __name__ == "__main__":
    main()
