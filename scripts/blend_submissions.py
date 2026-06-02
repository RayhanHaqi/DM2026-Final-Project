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


def blend_submission_frames(candidate_path, reference_path, candidate_weight):
    if not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("candidate_weight must be between 0 and 1")
    candidate, reference = _load_pair(candidate_path, reference_path)
    target_cols = list(reference.columns[1:])
    candidate_values = candidate[target_cols].to_numpy(dtype=float)
    reference_values = reference[target_cols].to_numpy(dtype=float)
    blended_values = candidate_weight * candidate_values + (1.0 - candidate_weight) * reference_values
    blended = reference.copy()
    blended[target_cols] = np.clip(blended_values, 0.0, 5.0)
    return blended


def write_blend_submission(candidate_path, reference_path, candidate_weight, output_path):
    blended = blend_submission_frames(candidate_path, reference_path, candidate_weight)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    blended.to_csv(output_path, index=False)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Blend two submission CSVs with a candidate weight.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate-weight", type=float, required=True)
    parser.add_argument("--output-dir", default="output/daily_candidates")
    parser.add_argument("--name", default="cnn_gru_blend")
    parser.add_argument("--output-path")
    return parser.parse_args()


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    args = parse_args()
    if args.output_path:
        out_path = write_blend_submission(
            args.candidate,
            args.reference,
            args.candidate_weight,
            args.output_path,
        )
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weight_tag = f"w{int(round(args.candidate_weight * 100)):02d}"
        out_path = os.path.join(args.output_dir, f"{args.name}_{weight_tag}_{timestamp}.csv")
        write_blend_submission(
            args.candidate,
            args.reference,
            args.candidate_weight,
            out_path,
        )
    print(f"Saved blend: {out_path}")


if __name__ == "__main__":
    main()
