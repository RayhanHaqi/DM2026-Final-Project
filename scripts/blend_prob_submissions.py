"""Blend cached class-probability tensors and write a submission CSV."""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.probability_blend import write_prob_blend_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Blend cached class probabilities into a submission.")
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        help="Weighted cache spec path.npz:weight (repeatable).",
    )
    parser.add_argument("--sample-path", default="data/sample_submission.csv")
    parser.add_argument("--output-dir", default="output/daily_candidates")
    parser.add_argument("--name", default="prob_blend")
    parser.add_argument("--output-path")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_path:
        out_path = args.output_path
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"{args.name}_{timestamp}.csv")

    out_path, sources = write_prob_blend_submission(args.cache, args.sample_path, out_path)
    print(f"Saved probability blend: {out_path}")
    for item in sources:
        print(f"  - {item['source']}: {item['path']}")


if __name__ == "__main__":
    main()
