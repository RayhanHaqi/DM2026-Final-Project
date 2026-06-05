import argparse

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


def _safe_corr(candidate_values, reference_values):
    cand_flat = candidate_values.reshape(-1)
    ref_flat = reference_values.reshape(-1)
    if np.std(cand_flat) == 0.0 or np.std(ref_flat) == 0.0:
        return 1.0 if np.allclose(cand_flat, ref_flat) else 0.0
    return float(np.corrcoef(cand_flat, ref_flat)[0, 1])


def compare_candidate(
    candidate_path,
    reference_path,
    min_corr=0.98,
    max_mean_abs_diff=0.10,
    max_week_mean_shift=0.08,
    max_global_mean_shift=None,
):
    candidate, reference = _load_pair(candidate_path, reference_path)
    target_cols = list(candidate.columns[1:])
    candidate_values = candidate[target_cols].to_numpy(dtype=float)
    reference_values = reference[target_cols].to_numpy(dtype=float)

    week_mean_shift = np.abs(candidate_values.mean(axis=0) - reference_values.mean(axis=0))
    global_mean_shift = float(
        abs(candidate_values.mean() - reference_values.mean())
    )
    result = {
        "correlation": _safe_corr(candidate_values, reference_values),
        "mean_abs_diff": float(np.mean(np.abs(candidate_values - reference_values))),
        "global_mean_shift": global_mean_shift,
        "candidate_week_means": candidate_values.mean(axis=0).tolist(),
        "reference_week_means": reference_values.mean(axis=0).tolist(),
        "candidate_week_stds": candidate_values.std(axis=0).tolist(),
        "candidate_week_mins": candidate_values.min(axis=0).tolist(),
        "candidate_week_maxs": candidate_values.max(axis=0).tolist(),
        "max_week_mean_shift": float(week_mean_shift.max()),
        "min_prediction": float(candidate_values.min()),
        "max_prediction": float(candidate_values.max()),
    }
    result["safe"] = (
        result["correlation"] >= min_corr
        and result["mean_abs_diff"] <= max_mean_abs_diff
        and result["max_week_mean_shift"] <= max_week_mean_shift
        and result["min_prediction"] >= 0.0
        and result["max_prediction"] <= 5.0
    )
    if max_global_mean_shift is not None:
        result["safe"] = (
            result["safe"] and result["global_mean_shift"] <= max_global_mean_shift
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Compare a candidate submission against a reference submission.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--min-corr", type=float, default=0.98)
    parser.add_argument("--max-mean-abs-diff", type=float, default=0.10)
    parser.add_argument("--max-week-mean-shift", type=float, default=0.08)
    parser.add_argument(
        "--max-global-mean-shift",
        type=float,
        default=None,
        help="If set, require |mean(cand)-mean(ref)| <= this for safe=True.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = compare_candidate(
        args.candidate,
        args.reference,
        min_corr=args.min_corr,
        max_mean_abs_diff=args.max_mean_abs_diff,
        max_week_mean_shift=args.max_week_mean_shift,
        max_global_mean_shift=args.max_global_mean_shift,
    )

    print("correlation", round(result["correlation"], 6))
    print("mean_abs_diff", round(result["mean_abs_diff"], 6))
    print("global_mean_shift", round(result["global_mean_shift"], 6))
    print("max_week_mean_shift", round(result["max_week_mean_shift"], 6))
    print("candidate_week_means", [round(v, 6) for v in result["candidate_week_means"]])
    print("reference_week_means", [round(v, 6) for v in result["reference_week_means"]])
    print("candidate_week_stds", [round(v, 6) for v in result["candidate_week_stds"]])
    print("prediction_range", round(result["min_prediction"], 6), round(result["max_prediction"], 6))
    print("safe", result["safe"])


if __name__ == "__main__":
    main()
