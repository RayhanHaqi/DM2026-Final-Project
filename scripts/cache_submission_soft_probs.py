"""Convert an existing regression submission CSV into a soft class-probability cache."""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.probability_blend import save_prob_cache, soft_probs_from_regression


def parse_args():
    parser = argparse.ArgumentParser(description="Cache soft probabilities from a submission CSV.")
    parser.add_argument("--submission", required=True, help="Input submission CSV with 5 week columns.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional Gaussian spread. Default preserves each prediction's expected value.",
    )
    parser.add_argument("--output-dir", default="output/prob_cache")
    parser.add_argument("--output-path")
    parser.add_argument("--name", default="soft_submission")
    return parser.parse_args()


def main():
    args = parse_args()
    sub = pd.read_csv(args.submission)
    id_col = sub.columns[0]
    target_cols = list(sub.columns[1:])
    region_ids = sub[id_col].tolist()
    values = sub[target_cols].to_numpy(dtype=float)
    class_probs = soft_probs_from_regression(values, temperature=args.temperature)

    if args.output_path:
        out_path = args.output_path
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"{args.name}_{timestamp}.npz")

    save_prob_cache(
        out_path,
        class_probs,
        region_ids,
        source="soft_submission",
        metadata={"temperature": args.temperature, "submission": args.submission},
    )
    print(f"Saved soft probability cache: {out_path}")
    print(f"shape={class_probs.shape}")


if __name__ == "__main__":
    main()
