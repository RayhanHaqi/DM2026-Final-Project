from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from model import train, utils


def build_window_samples_from_frame(df, window_days=91):
    meta_cols = ["region_id", "date", "score"]
    feat_cols = [col for col in df.columns if col not in meta_cols]
    score_vals = df["score"].to_numpy()
    samples = []

    for region_id, grp in df.groupby("region_id", sort=False):
        indices = grp.index.to_numpy()
        score_positions = np.where(pd.notna(score_vals[indices]))[0]

        for start in range(0, len(score_positions) - 4):
            label_pos = score_positions[start:start + 5]
            first_label_pos = label_pos[0]
            if first_label_pos < window_days:
                continue

            window_indices = indices[first_label_pos - window_days:first_label_pos]
            samples.append({
                "region_id": region_id,
                "score_idx_start": start,
                "window": df.iloc[window_indices][feat_cols].to_numpy(dtype=float),
                "target": score_vals[indices[label_pos]].astype(float),
            })

    return samples, feat_cols


def build_recent_backtest_splits(samples, n_recent_cutoffs=3, max_train_windows_per_region=None):
    by_region = defaultdict(list)
    for sample in samples:
        by_region[sample["region_id"]].append(sample)

    ordered = {}
    for region_id, region_samples in by_region.items():
        ordered[region_id] = sorted(region_samples, key=lambda sample: sample["score_idx_start"])

    min_windows = min(len(region_samples) for region_samples in ordered.values())
    if n_recent_cutoffs >= min_windows:
        raise ValueError("Not enough horizons for requested recent cutoffs")

    splits = []
    for offset in range(1, n_recent_cutoffs + 1):
        train_samples = []
        val_samples = []

        for region_samples in ordered.values():
            val_index = len(region_samples) - offset
            val_samples.append(region_samples[val_index])

            region_train = region_samples[:val_index]
            if max_train_windows_per_region is not None:
                region_train = region_train[-max_train_windows_per_region:]
            train_samples.extend(region_train)

        splits.append({
            "cutoff_offset": offset,
            "train_samples": train_samples,
            "val_samples": val_samples,
        })

    return splits


def summarize_predictions(split_rows):
    y_true = np.vstack([row["y_true"] for row in split_rows])
    preds = np.vstack([row["preds"] for row in split_rows])
    cutoff_offsets = sorted({row["cutoff_offset"] for row in split_rows})

    return {
        "overall_mae": float(mean_absolute_error(y_true, preds)),
        "per_week_mae": [
            float(mean_absolute_error(y_true[:, idx], preds[:, idx]))
            for idx in range(y_true.shape[1])
        ],
        "per_cutoff_mae": [
            {
                "cutoff_offset": cutoff_offset,
                "mae": float(mean_absolute_error(
                    np.vstack([row["y_true"] for row in split_rows if row["cutoff_offset"] == cutoff_offset]),
                    np.vstack([row["preds"] for row in split_rows if row["cutoff_offset"] == cutoff_offset]),
                )),
            }
            for cutoff_offset in cutoff_offsets
        ],
        "prediction_means": preds.mean(axis=0).tolist(),
        "target_means": y_true.mean(axis=0).tolist(),
        "n_validation_rows": int(len(split_rows)),
    }


def _aggregate_tree_matrix(samples, feat_cols):
    rows = [utils._aggregate_array(sample["window"], feat_cols) for sample in samples]
    return pd.DataFrame(rows)


def evaluate_tree_backtest_from_frame(
    df,
    n_recent_cutoffs=3,
    max_train_windows_per_region=52,
    params_override=None,
):
    samples, feat_cols = build_window_samples_from_frame(df)
    splits = build_recent_backtest_splits(
        samples,
        n_recent_cutoffs=n_recent_cutoffs,
        max_train_windows_per_region=max_train_windows_per_region,
    )
    split_rows = []

    for split in splits:
        X_tr = _aggregate_tree_matrix(split["train_samples"], feat_cols)
        y_tr = np.vstack([sample["target"] for sample in split["train_samples"]])
        X_val = _aggregate_tree_matrix(split["val_samples"], feat_cols)
        y_val = np.vstack([sample["target"] for sample in split["val_samples"]])

        model = train.train_xgboost(X_tr, y_tr, params_override=params_override)
        preds = np.clip(model.predict(X_val), 0.0, 5.0)

        for idx, sample in enumerate(split["val_samples"]):
            split_rows.append({
                "region_id": sample["region_id"],
                "cutoff_offset": split["cutoff_offset"],
                "y_true": y_val[idx],
                "preds": preds[idx],
            })

    return summarize_predictions(split_rows)
