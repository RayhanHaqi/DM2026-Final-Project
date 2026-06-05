"""Re-export blended class probs with MAE-oriented decision rules (median / quantile).

WARNING: Non-mean decoders are INVALID on soft_probs_from_regression caches (92% of
the default 92/8 blend). They collapse predictions and scored 0.9206 on Kaggle.
Use scripts/diagnose_prob_cache_decoders.py before any non-mean experiment.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.probability_blend import write_prob_blend_submission

INVALID_DECODER_MSG = (
    "Non-mean decision on soft-wrapped regression caches is invalid. "
    "Pass --allow-invalid-decoder to proceed (not recommended for Kaggle)."
)

DEFAULT_CACHE_SPECS = [
    "output/prob_cache/soft_prob_best8089.npz:0.92",
    "output/prob_cache/ordinal_hybrid_best.npz:0.08",
]
DEFAULT_ANCHOR = "output/daily_candidates/prob_blend_recycle8089_ord08.csv"

DECISION_VARIANTS = (
    ("mean", None, "mean"),
    ("median", None, "median"),
    ("q045", 0.45, "quantile"),
    ("q048", 0.48, "quantile"),
    ("q050", 0.50, "quantile"),
    ("q052", 0.52, "quantile"),
)

SMOKE_VARIANTS = (
    ("median", None, "median"),
    ("q048", 0.48, "quantile"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Same 92/8 prob blend as best; change mean -> median/quantile decision."
    )
    parser.add_argument("--cache", action="append", help="path.npz:weight (repeatable)")
    parser.add_argument("--sample-path", default="data/sample_submission.csv")
    parser.add_argument("--output-dir", default="output/daily_candidates")
    parser.add_argument("--smoke", action="store_true", help="Only median + q048 variants.")
    parser.add_argument(
        "--decision",
        choices=("mean", "median", "quantile"),
        help="Generate only this decision rule (default: all or smoke subset).",
    )
    parser.add_argument("--quantile", type=float, default=0.48, help="For --decision quantile.")
    parser.add_argument("--output-path", help="Single output CSV (requires --decision).")
    parser.add_argument(
        "--allow-invalid-decoder",
        action="store_true",
        help="Required for median/quantile (invalid on default soft+ordinal caches).",
    )
    return parser.parse_args()


def _require_valid_decoder(args, decision):
    if decision == "mean":
        return
    if not args.allow_invalid_decoder:
        print(INVALID_DECODER_MSG, file=sys.stderr)
        raise SystemExit(2)
    print("WARNING:", INVALID_DECODER_MSG, file=sys.stderr)


def main():
    args = parse_args()
    cache_specs = args.cache if args.cache else list(DEFAULT_CACHE_SPECS)

    if args.decision:
        _require_valid_decoder(args, args.decision)
        tag = args.decision
        if args.decision == "quantile":
            q_tag = f"q{int(round(args.quantile * 100)):02d}"
            tag = f"quantile_{q_tag}"
        out = args.output_path or os.path.join(
            args.output_dir,
            f"prob_mae_{tag}_8089_ord08.csv",
        )
        kwargs = {"decision": args.decision}
        if args.decision == "quantile":
            kwargs["quantile"] = args.quantile
        write_prob_blend_submission(cache_specs, args.sample_path, out, **kwargs)
        print(f"Saved: {out}")
        return

    variants = SMOKE_VARIANTS if args.smoke else DECISION_VARIANTS
    os.makedirs(args.output_dir, exist_ok=True)
    tag_suffix = "_smoke" if args.smoke else "_full"

    for label, q_value, decision in variants:
        _require_valid_decoder(args, decision)
        out_path = os.path.join(
            args.output_dir,
            f"prob_mae_{label}_8089_ord08{tag_suffix}.csv",
        )
        kwargs = {"decision": decision}
        if decision == "quantile":
            kwargs["quantile"] = q_value
        write_prob_blend_submission(cache_specs, args.sample_path, out_path, **kwargs)
        print(f"Saved: {out_path} ({decision}" + (f" q={q_value}" if q_value else "") + ")")

    print(f"\nGate vs anchor: {DEFAULT_ANCHOR}")
    print(
        "  PYTHONPATH=. python scripts/gate_and_validate_submissions.py "
        f"--reference {DEFAULT_ANCHOR} --candidate <csv>"
    )


if __name__ == "__main__":
    main()
