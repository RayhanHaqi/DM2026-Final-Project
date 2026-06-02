import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd


def _load_pair(candidate_path, reference_path):
    candidate = pd.read_csv(candidate_path)
    reference = pd.read_csv(reference_path)
    if list(candidate.columns) != list(reference.columns):
        raise ValueError("Columns do not match reference submission")
    if candidate.shape != reference.shape:
        raise ValueError("Shape does not match reference submission")
    if candidate.iloc[:, 0].tolist() != reference.iloc[:, 0].tolist():
        raise ValueError("ID order does not match reference submission")
    return candidate, reference


def blend_weekly_submission_frames(candidate_path, reference_path, candidate_weights):
    weights = np.asarray(candidate_weights, dtype=float)
    if weights.shape != (5,):
        raise ValueError("Expected five weekly weights")
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("candidate weights must be between 0 and 1")

    candidate, reference = _load_pair(candidate_path, reference_path)
    target_cols = list(reference.columns[1:])
    if len(target_cols) != 5:
        raise ValueError("Expected five target columns")

    candidate_values = candidate[target_cols].to_numpy(dtype=float)
    reference_values = reference[target_cols].to_numpy(dtype=float)
    blended_values = weights * candidate_values + (1.0 - weights) * reference_values
    blended = reference.copy()
    blended[target_cols] = np.clip(blended_values, 0.0, 5.0)
    return blended


def parse_args():
    parser = argparse.ArgumentParser(description="Blend two submission CSVs with one candidate weight per target week.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate-weights", required=True, help="Comma-separated five weights, e.g. 0.24,0.24,0.25,0.26,0.26")
    parser.add_argument("--output-dir", default="output/daily_candidates")
    parser.add_argument("--name", default="weekly_blend")
    return parser.parse_args()


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    args = parse_args()
    weights = [float(value) for value in args.candidate_weights.split(",")]
    blended = blend_weekly_submission_frames(args.candidate, args.reference, weights)
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_tag = "w" + "_".join(str(int(round(weight * 100))).zfill(2) for weight in weights)
    out_path = os.path.join(args.output_dir, f"{args.name}_{weight_tag}_{timestamp}.csv")
    blended.to_csv(out_path, index=False)
    print(f"Saved weekly blend: {out_path}")


if __name__ == "__main__":
    main()
