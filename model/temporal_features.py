import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from model import utils


BLOCKS = [
    ("full91", 0, 91),
    ("first30", 0, 30),
    ("middle30", 30, 60),
    ("last31", 60, 91),
    ("last30", 61, 91),
    ("last14", 77, 91),
    ("last7", 84, 91),
]


def _block_stats(block, feature_name):
    return {
        f"{feature_name}__mean": block.mean(),
        f"{feature_name}__std": block.std(),
        f"{feature_name}__min": block.min(),
        f"{feature_name}__max": block.max(),
        f"{feature_name}__q25": np.quantile(block, 0.25),
        f"{feature_name}__q50": np.quantile(block, 0.50),
        f"{feature_name}__q75": np.quantile(block, 0.75),
        f"{feature_name}__range": block.max() - block.min(),
        f"{feature_name}__last_minus_first": block[-1] - block[0],
    }


def build_temporal_features_from_window(window_array, feat_cols):
    values = {}
    means = {}

    for block_name, start, end in BLOCKS:
        block_array = window_array[start:end]
        for j, feature_name in enumerate(feat_cols):
            series = block_array[:, j]
            stats = _block_stats(series, feature_name)
            for stat_name, stat_value in stats.items():
                values[f"{block_name}__{stat_name}"] = stat_value
            means[(block_name, feature_name)] = stats[f"{feature_name}__mean"]

    for feature_name in feat_cols:
        values[f"delta_last7_full91__{feature_name}__mean"] = means[("last7", feature_name)] - means[("full91", feature_name)]
        values[f"delta_last14_full91__{feature_name}__mean"] = means[("last14", feature_name)] - means[("full91", feature_name)]
        values[f"delta_last30_first30__{feature_name}__mean"] = means[("last30", feature_name)] - means[("first30", feature_name)]
        values[f"delta_last30_middle30__{feature_name}__mean"] = means[("last30", feature_name)] - means[("middle30", feature_name)]

    return pd.Series(values)


def build_hybrid_temporal_features_from_window(window_array, feat_cols):
    values = utils._aggregate_array(window_array, feat_cols).to_dict()

    for j, feature_name in enumerate(feat_cols):
        series = window_array[:, j]
        full_mean = series.mean()
        first30_mean = series[:30].mean()
        middle30_mean = series[30:60].mean()
        last30_mean = series[-30:].mean()
        last14_mean = series[-14:].mean()
        last7_mean = series[-7:].mean()
        prefix = f"temporal__{feature_name}"

        values[f"{prefix}__first30_mean"] = first30_mean
        values[f"{prefix}__middle30_mean"] = middle30_mean
        values[f"{prefix}__last30_mean"] = last30_mean
        values[f"{prefix}__last14_mean"] = last14_mean
        values[f"{prefix}__last7_mean"] = last7_mean
        values[f"{prefix}__last7_minus_full91_mean"] = last7_mean - full_mean
        values[f"{prefix}__last14_minus_full91_mean"] = last14_mean - full_mean
        values[f"{prefix}__last30_minus_first30_mean"] = last30_mean - first30_mean
        values[f"{prefix}__last_value_minus_first_value"] = series[-1] - series[0]

    return pd.Series(values)


def _build_features_from_window(window_array, feat_cols, feature_set):
    if feature_set == "hybrid":
        return build_hybrid_temporal_features_from_window(window_array, feat_cols)
    if feature_set == "blocks":
        return build_temporal_features_from_window(window_array, feat_cols)
    raise ValueError("feature_set must be 'hybrid' or 'blocks'")


def build_temporal_train_data_from_frame(df, max_windows_per_region=None, feature_set="hybrid"):
    meta_cols = ["region_id", "date", "score"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    score_vals = df["score"].values
    X_list, y_list, region_list = [], [], []

    grouped = df.groupby("region_id", sort=False)
    for region_id, grp in tqdm(grouped, total=grouped.ngroups, desc="Building temporal train windows"):
        indices = grp.index.values
        score_mask = pd.notna(score_vals[indices])
        score_positions = np.where(score_mask)[0]

        if max_windows_per_region is not None:
            start_idx = max(0, len(score_positions) - 5 - max_windows_per_region)
            score_positions = score_positions[start_idx:]

        for start in range(0, len(score_positions) - 4):
            label_pos = score_positions[start:start + 5]
            label_values = score_vals[indices[label_pos]]
            first_label_pos = label_pos[0]
            if first_label_pos < 91:
                continue
            window_indices = indices[first_label_pos - 91:first_label_pos]
            window = df.iloc[window_indices][feat_cols].fillna(0).values
            X_list.append(_build_features_from_window(window, feat_cols, feature_set))
            y_list.append(label_values)
            region_list.append(region_id)

    return pd.DataFrame(X_list).reset_index(drop=True), np.array(y_list), region_list


def build_temporal_test_data_from_frame(df, feature_set="hybrid"):
    meta_cols = ["region_id", "date"]
    if "score" in df.columns:
        meta_cols.append("score")
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X_list, region_list = [], []

    grouped = df.groupby("region_id", sort=False)
    for region_id, grp in tqdm(grouped, total=grouped.ngroups, desc="Building temporal test windows"):
        window = grp[feat_cols].fillna(0).values[-91:]
        X_list.append(_build_features_from_window(window, feat_cols, feature_set))
        region_list.append(region_id)

    return pd.DataFrame(X_list).reset_index(drop=True), region_list


def load_temporal_train_data(path, max_windows_per_region=52, feature_set="hybrid"):
    cache_tag = f"temporal_{feature_set}_{max_windows_per_region}"
    cache_x = path.replace(".csv", f"_X_{cache_tag}.npy")
    cache_y = path.replace(".csv", f"_y_{cache_tag}.npy")
    cache_r = path.replace(".csv", f"_regions_{cache_tag}.npy")

    if os.path.exists(cache_x):
        return pd.DataFrame(np.load(cache_x, allow_pickle=True)), np.load(cache_y), list(np.load(cache_r, allow_pickle=True))

    X, y, regions = build_temporal_train_data_from_frame(pd.read_csv(path), max_windows_per_region=max_windows_per_region, feature_set=feature_set)
    np.save(cache_x, X.values)
    np.save(cache_y, y)
    np.save(cache_r, np.array(regions, dtype=object))
    return X, y, regions


def load_temporal_test_data(path, feature_set="hybrid"):
    cache_x = path.replace(".csv", f"_X_temporal_{feature_set}_test.npy")
    cache_r = path.replace(".csv", f"_regions_temporal_{feature_set}_test.npy")

    if os.path.exists(cache_x):
        return pd.DataFrame(np.load(cache_x, allow_pickle=True)), list(np.load(cache_r, allow_pickle=True))

    X, regions = build_temporal_test_data_from_frame(pd.read_csv(path), feature_set=feature_set)
    np.save(cache_x, X.values)
    np.save(cache_r, np.array(regions, dtype=object))
    return X, regions
