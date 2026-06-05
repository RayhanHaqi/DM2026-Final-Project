"""Distribution gate and submission format check for candidate CSVs."""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments
from scripts.compare_candidate_distribution import compare_candidate


def main():
    parser = argparse.ArgumentParser(
        description="Gate candidates vs reference and validate submission format."
    )
    parser.add_argument("--reference", required=True, help="Reference submission CSV")
    parser.add_argument("--candidate", action="append", required=True, help="Candidate CSV (repeatable)")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    parser.add_argument("--min-corr", type=float, default=0.995)
    parser.add_argument("--max-mean-abs-diff", type=float, default=0.035)
    parser.add_argument("--max-week-mean-shift", type=float, default=0.02)
    parser.add_argument(
        "--max-global-mean-shift",
        type=float,
        default=0.05,
        help="Reject if |global mean cand - ref| exceeds this (catches invalid decoders).",
    )
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)

    all_ok = True
    for path in args.candidate:
        print(f"=== {path} ===")
        cand = pd.read_csv(path)
        ok, messages = experiments.validate_submission(cand, sample)
        print("validate_submission", ok, messages if messages else "")

        result = compare_candidate(
            path,
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
        print("safe", result["safe"])
        print()
        all_ok = all_ok and ok and result["safe"]

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
