"""Read-only diagnostic: mean vs median/quantile decoders on prob caches."""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.probability_blend import diagnose_prob_blend_decoders

DEFAULT_CACHE_SPECS = [
    "output/prob_cache/soft_prob_best8089.npz:0.92",
    "output/prob_cache/ordinal_hybrid_best.npz:0.08",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report class mass and decoded global means per cache and blend."
    )
    parser.add_argument("--cache", action="append", help="path.npz:weight (repeatable)")
    parser.add_argument("--sample-path", default="data/sample_submission.csv")
    parser.add_argument(
        "--quantile",
        type=float,
        action="append",
        dest="quantiles",
        help="Quantile decoders to report (repeatable). Default: 0.48 0.5 0.52",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args()


def _print_row(row):
    print(f"\n--- {row['layer']}: {row.get('path', '')} (w={row.get('weight')}) source={row.get('source')} ---")
    mass = row["mass"]
    print("  mean_class_mass", mass["mean_class_mass"])
    for k in range(1, 6):
        key = f"p_ge_{k}"
        if key in mass:
            print(f"  {key}", round(mass[key], 4))
    print("  decode_mean", round(row["decode_mean"], 4))
    print("  decode_median", round(row["decode_median"], 4))
    for key, value in sorted(row.items()):
        if key.startswith("decode_q"):
            print(f"  {key}", round(value, 4))
    if row["layer"] == "blend":
        shift = row["decode_mean"] - row["decode_median"]
        print(f"  mean_minus_median", round(shift, 4))
        if abs(shift) > 0.15:
            print(
                "  WARNING: large mean/median gap — non-mean decoders likely invalid "
                "(soft_probs_from_regression is mean-preserving only)."
            )


def main():
    args = parse_args()
    cache_specs = args.cache if args.cache else list(DEFAULT_CACHE_SPECS)
    quantiles = args.quantiles if args.quantiles else [0.48, 0.5, 0.52]

    sample = pd.read_csv(args.sample_path)
    target_region_ids = sample.iloc[:, 0].tolist()
    layers = diagnose_prob_blend_decoders(cache_specs, target_region_ids, quantiles=quantiles)

    if args.json:
        print(json.dumps(layers, indent=2))
        return

    print("Prob cache decoder diagnostic (read-only, no Kaggle)")
    print("cache_specs:", cache_specs)
    for row in layers:
        _print_row(row)

    blend = layers[-1]
    q52_key = "decode_q52" if "decode_q52" in blend else None
    if q52_key:
        collapse = blend["decode_mean"] - blend[q52_key]
        print(f"\nBlend decode_mean - decode_q52 = {collapse:.4f}")
        if collapse > 0.3:
            print(
                "FAIL: q=0.52 collapses predictions vs mean decoder "
                "(matches prob_mae public MAE 0.9206 failure)."
            )


if __name__ == "__main__":
    main()
