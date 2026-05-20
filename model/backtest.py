from collections import defaultdict

import numpy as np
import pandas as pd


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
