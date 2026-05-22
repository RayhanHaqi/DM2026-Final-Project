import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import backtest, cnn_candidate, experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Generate small 1D CNN Kaggle submission.")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    parser.add_argument("--output-dir", default="output/daily_candidates")
    parser.add_argument("--max-windows-per-region", type=int, default=52)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", choices=["small", "v2", "cnn_gru"], default="small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--calendar", action="store_true")
    parser.add_argument("--no-backtest", action="store_true")
    parser.add_argument("--backtest-cutoffs", type=int, default=2)
    return parser.parse_args()


def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "4"
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend = cnn_candidate.require_deep_learning_backend()
    if backend != "torch":
        raise SystemExit("This runner currently supports PyTorch. Install torch before running it.")

    train_df = pd.read_csv(args.train)

    if not args.no_backtest:
        backtest_summary = backtest.evaluate_cnn_backtest_from_frame(
            train_df,
            n_recent_cutoffs=args.backtest_cutoffs,
            max_train_windows_per_region=args.max_windows_per_region,
            model_name=args.model,
            epochs=min(args.epochs, 10),
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            include_calendar=args.calendar,
        )
        print("backtest_mae", round(backtest_summary["overall_mae"], 6))

    test_df = pd.read_csv(args.test)
    X_train, y_train, train_regions, feat_cols = cnn_candidate.build_sequence_train_data_from_frame(
        train_df,
        max_windows_per_region=args.max_windows_per_region,
        include_calendar=args.calendar,
    )
    if args.calendar:
        test_df = cnn_candidate.add_calendar_features(test_df)
    X_test, test_regions = cnn_candidate.build_sequence_test_data_from_frame(test_df, feat_cols)
    X_train, X_test = cnn_candidate.standardize_sequences(X_train, X_test)

    model, val_mae = cnn_candidate.train_torch_cnn(
        X_train,
        y_train,
        train_regions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        model_name=args.model,
        patience=args.patience,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
    )
    preds = np.clip(cnn_candidate.predict_torch_cnn(model, X_test), 0.0, 5.0)

    sample = pd.read_csv(args.sample)
    sub = experiments.build_submission(test_regions, preds, sample)
    ok, messages = experiments.validate_submission(sub, sample)
    if not ok:
        raise SystemExit("; ".join(messages))

    os.makedirs(args.output_dir, exist_ok=True)
    name = f"cnn_1d_{args.model}"
    out_path = os.path.join(args.output_dir, f"{name}_{timestamp}.csv")
    summary_path = os.path.join(args.output_dir, f"{name}_{timestamp}_summary.csv")
    sub.to_csv(out_path, index=False)
    pd.DataFrame([{
        "name": name,
        "model": args.model,
        "val_mae": val_mae,
        "max_windows_per_region": args.max_windows_per_region,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "patience": args.patience,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "calendar": args.calendar,
        "path": out_path,
    }]).to_csv(summary_path, index=False)

    print(f"Saved submission: {out_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
