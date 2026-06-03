"""Blend cached class-probability tensors and write a submission CSV."""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import experiments
from model.probability_blend import (
    blend_class_probs,
    class_probs_to_predictions,
    load_prob_cache,
    parse_weighted_cache_specs,
    reorder_class_probs,
)


def blend_prob_caches(cache_specs, target_region_ids):
    parsed = parse_weighted_cache_specs(cache_specs)
    loaded = [load_prob_cache(path) for path, _ in parsed]
    target_region_ids = [str(rid) for rid in target_region_ids]

    aligned_probs = []
    for item in loaded:
        aligned_probs.append(
            reorder_class_probs(item["class_probs"], item["region_ids"], target_region_ids)
        )

    weights = [weight for _, weight in parsed]
    blended = blend_class_probs(aligned_probs, weights)
    return blended, target_region_ids, loaded


def write_prob_blend_submission(cache_specs, sample_path, output_path):
    sample = pd.read_csv(sample_path)
    target_region_ids = sample.iloc[:, 0].tolist()
    blended, region_ids, sources = blend_prob_caches(cache_specs, target_region_ids)
    preds = class_probs_to_predictions(blended)
    sub = experiments.build_submission(region_ids, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise ValueError("; ".join(messages))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sub.to_csv(output_path, index=False)
    return output_path, sources


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
