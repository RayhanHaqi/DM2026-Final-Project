import numpy as np
import pandas as pd


def _empty_features():
    features = {
        "score_last_known": 0.0,
        "score_mean_all": 0.0,
        "score_median_all": 0.0,
        "score_mean_last5": 0.0,
        "score_mean_last10": 0.0,
        "score_mean_last20": 0.0,
        "score_mean_last52": 0.0,
        "score_zero_frac": 0.0,
        "score_high_frac_ge3": 0.0,
    }
    for class_id in range(6):
        features[f"score_class_frac_{class_id}"] = 0.0
    return features


def summarize_score_history(score_values, cutoff_pos=None):
    values = np.asarray(score_values, dtype=float)
    if cutoff_pos is not None:
        values = values[:cutoff_pos]
    values = values[~np.isnan(values)]

    features = _empty_features()
    if len(values) == 0:
        return features

    features["score_last_known"] = float(values[-1])
    features["score_mean_all"] = float(values.mean())
    features["score_median_all"] = float(np.median(values))
    features["score_mean_last5"] = float(values[-5:].mean())
    features["score_mean_last10"] = float(values[-10:].mean())
    features["score_mean_last20"] = float(values[-20:].mean())
    features["score_mean_last52"] = float(values[-52:].mean())
    features["score_zero_frac"] = float(np.mean(values == 0.0))
    features["score_high_frac_ge3"] = float(np.mean(values >= 3.0))
    for class_id in range(6):
        features[f"score_class_frac_{class_id}"] = float(np.mean(values == float(class_id)))
    return features


def build_region_test_history(train_df):
    rows = {}
    for region_id, grp in train_df.groupby("region_id", sort=False):
        rows[region_id] = summarize_score_history(grp["score"].to_numpy(dtype=float))
    return pd.DataFrame.from_dict(rows, orient="index")


def build_train_history_features_from_frame(df, max_windows_per_region=None):
    score_vals = df["score"].values
    rows = []

    grouped = df.groupby("region_id", sort=False)
    for _, grp in grouped:
        indices = grp.index.values
        score_mask = pd.notna(score_vals[indices])
        score_positions = np.where(score_mask)[0]

        if max_windows_per_region is not None:
            start_idx = max(0, len(score_positions) - 5 - max_windows_per_region)
            score_positions = score_positions[start_idx:]

        for start in range(0, len(score_positions) - 4):
            label_pos = score_positions[start:start + 5]
            first_label_pos = label_pos[0]
            if first_label_pos < 91:
                continue
            rows.append(summarize_score_history(score_vals[indices], cutoff_pos=first_label_pos))

    return pd.DataFrame(rows).reset_index(drop=True)


def build_test_history_features_from_frame(train_df, test_regions):
    region_history = build_region_test_history(train_df)
    empty = _empty_features()
    rows = []
    for region_id in test_regions:
        if region_id in region_history.index:
            rows.append(region_history.loc[region_id].to_dict())
        else:
            rows.append(dict(empty))
    return pd.DataFrame(rows).reset_index(drop=True)


def _summarize_with_count_and_gap(score_values, cutoff_pos=None, first_label_pos=None):
    features = summarize_score_history(score_values, cutoff_pos=cutoff_pos)
    values = np.asarray(score_values, dtype=float)
    if cutoff_pos is not None:
        values = values[:cutoff_pos]
    valid_positions = np.where(~np.isnan(values))[0]
    features["score_history_count"] = float(len(valid_positions))
    if len(valid_positions) == 0 or first_label_pos is None:
        features["score_history_gap_positions"] = 0.0
    else:
        features["score_history_gap_positions"] = float(first_label_pos - valid_positions[-1])
    return features


def build_train_blackout_history_features_from_frame(df, max_windows_per_region=None, window_days=91):
    score_vals = df["score"].values
    rows = []

    grouped = df.groupby("region_id", sort=False)
    for _, grp in grouped:
        indices = grp.index.values
        score_mask = pd.notna(score_vals[indices])
        score_positions = np.where(score_mask)[0]

        if max_windows_per_region is not None:
            start_idx = max(0, len(score_positions) - 5 - max_windows_per_region)
            score_positions = score_positions[start_idx:]

        for start in range(0, len(score_positions) - 4):
            label_pos = score_positions[start:start + 5]
            first_label_pos = label_pos[0]
            if first_label_pos < window_days:
                continue
            cutoff_pos = max(0, first_label_pos - window_days)
            rows.append(_summarize_with_count_and_gap(score_vals[indices], cutoff_pos=cutoff_pos, first_label_pos=first_label_pos))

    return pd.DataFrame(rows).reset_index(drop=True)


def build_test_blackout_history_features_from_frame(train_df, test_regions, window_days=91):
    rows = []
    for region_id in test_regions:
        region_scores = train_df.loc[train_df["region_id"] == region_id, "score"].to_numpy(dtype=float)
        features = _summarize_with_count_and_gap(region_scores, cutoff_pos=None, first_label_pos=None)
        if features["score_history_count"] > 0.0:
            features["score_history_gap_positions"] = float(window_days)
        rows.append(features)
    return pd.DataFrame(rows).reset_index(drop=True)
