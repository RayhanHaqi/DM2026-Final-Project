import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import backtest


def parse_args():
    parser = argparse.ArgumentParser(description="Run temporal backtest validation.")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--mode", choices=["tree", "cnn", "both"], default="both")
    parser.add_argument("--recent-cutoffs", type=int, default=3)
    parser.add_argument("--max-windows-per-region", type=int, default=52)
    parser.add_argument("--output-dir", default="output/backtests")
    parser.add_argument("--model", choices=["small", "v2"], default="small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", action="store_true")
    return parser.parse_args()


def flatten_summary(name, summary):
    row = {
        "name": name,
        "overall_mae": summary["overall_mae"],
        "n_validation_rows": summary["n_validation_rows"],
    }
    for idx, value in enumerate(summary["per_week_mae"], start=1):
        row[f"week{idx}_mae"] = value
    for idx, value in enumerate(summary["prediction_means"], start=1):
        row[f"week{idx}_pred_mean"] = value
    for idx, value in enumerate(summary["target_means"], start=1):
        row[f"week{idx}_target_mean"] = value
    return row


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    train_df = pd.read_csv(args.train)
    rows = []

    if args.mode in {"tree", "both"}:
        tree_summary = backtest.evaluate_tree_backtest_from_frame(
            train_df,
            n_recent_cutoffs=args.recent_cutoffs,
            max_train_windows_per_region=args.max_windows_per_region,
        )
        rows.append(flatten_summary("tree", tree_summary))
        print("tree overall_mae", round(tree_summary["overall_mae"], 6))

    if args.mode in {"cnn", "both"}:
        cnn_summary = backtest.evaluate_cnn_backtest_from_frame(
            train_df,
            n_recent_cutoffs=args.recent_cutoffs,
            max_train_windows_per_region=args.max_windows_per_region,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
        )
        rows.append(flatten_summary(f"cnn_{args.model}", cnn_summary))
        print("cnn overall_mae", round(cnn_summary["overall_mae"], 6))

    out_path = os.path.join(args.output_dir, f"temporal_backtest_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("saved", out_path)


if __name__ == "__main__":
    main()
