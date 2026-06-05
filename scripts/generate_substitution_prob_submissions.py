"""Build Codex substitution probability blends (season-soft + history-only ordinal)."""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.probability_blend import write_prob_blend_submission

DEFAULT_ANCHOR = "output/daily_candidates/prob_blend_recycle8089_ord08.csv"
DEFAULT_SOFT8089 = "output/prob_cache/soft_prob_best8089.npz"
DEFAULT_HYBRID_ORD = "output/prob_cache/ordinal_hybrid_best.npz"
DEFAULT_SEASON_SCALAR = "output/daily_candidates/season_w20_w20_20260530_122421.csv"
DEFAULT_SOFT_SEASON = "output/prob_cache/soft_season_scalar_best.npz"
DEFAULT_HISTORY_ORD = "output/prob_cache/ordinal_history_only_best.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache sources and write substitution prob-blend candidates."
    )
    parser.add_argument("--track", choices=("season", "history", "both"), default="both")
    parser.add_argument("--smoke", action="store_true", help="Fast ordinal cache (2 windows, 15 trees).")
    parser.add_argument("--skip-cache", action="store_true", help="Skip cache rebuild; blend only.")
    parser.add_argument("--anchor-csv", default=DEFAULT_ANCHOR)
    parser.add_argument("--soft8089", default=DEFAULT_SOFT8089)
    parser.add_argument("--hybrid-ord", default=DEFAULT_HYBRID_ORD)
    parser.add_argument("--season-scalar", default=DEFAULT_SEASON_SCALAR)
    parser.add_argument("--soft-season", default=DEFAULT_SOFT_SEASON)
    parser.add_argument("--history-ord", default=DEFAULT_HISTORY_ORD)
    parser.add_argument("--output-dir", default="output/daily_candidates")
    parser.add_argument("--sample-path", default="data/sample_submission.csv")
    return parser.parse_args()


def _run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _cache_season_soft(season_csv, out_path):
    _run(
        [
            sys.executable,
            "scripts/cache_submission_soft_probs.py",
            "--submission",
            season_csv,
            "--output-path",
            out_path,
        ]
    )


def _cache_history_ordinal(out_path, smoke):
    cmd = [
        sys.executable,
        "scripts/cache_ordinal_probabilities.py",
        "--feature-set",
        "history_only",
        "--output-path",
        out_path,
    ]
    if smoke:
        cmd.append("--smoke")
    _run(cmd)


def _blend_three(soft8089, hybrid_ord, third_cache, third_weight, hybrid_weight, out_path, sample_path):
    soft_weight = 1.0 - hybrid_weight - third_weight
    if soft_weight <= 0:
        raise ValueError("weights must leave positive mass for soft8089")
    specs = [
        f"{soft8089}:{soft_weight}",
        f"{hybrid_ord}:{hybrid_weight}",
        f"{third_cache}:{third_weight}",
    ]
    write_prob_blend_submission(specs, sample_path, out_path)
    print(f"Saved: {out_path}")
    print(f"  weights: soft={soft_weight:.4f} hybrid={hybrid_weight:.4f} third={third_weight:.4f}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tag = "smoke" if args.smoke else "full"

    if not args.skip_cache:
        if args.track in ("season", "both"):
            if not os.path.isfile(args.season_scalar):
                raise SystemExit(f"Missing season scalar CSV: {args.season_scalar}")
            _cache_season_soft(args.season_scalar, args.soft_season)
        if args.track in ("history", "both"):
            _cache_history_ordinal(args.history_ord, args.smoke)

    if args.track in ("season", "both"):
        for third_w, hybrid_w, label in (
            (0.03, 0.08, "890_803"),
            (0.02, 0.08, "900_802"),
        ):
            out = os.path.join(
                args.output_dir,
                f"prob_subst_season_{label}_{tag}.csv",
            )
            _blend_three(
                args.soft8089,
                args.hybrid_ord,
                args.soft_season,
                third_w,
                hybrid_w,
                out,
                args.sample_path,
            )

    if args.track in ("history", "both"):
        for third_w, hybrid_w, label in (
            (0.02, 0.06, "920_602"),
            (0.03, 0.06, "910_603"),
        ):
            out = os.path.join(
                args.output_dir,
                f"prob_subst_history_{label}_{tag}.csv",
            )
            _blend_three(
                args.soft8089,
                args.hybrid_ord,
                args.history_ord,
                third_w,
                hybrid_w,
                out,
                args.sample_path,
            )

    print("\nGate candidates:")
    print(
        f"  PYTHONPATH=. python scripts/gate_and_validate_submissions.py "
        f"--reference {args.anchor_csv} --candidate <csv>"
    )


if __name__ == "__main__":
    main()
