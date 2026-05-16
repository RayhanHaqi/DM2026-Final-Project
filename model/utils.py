import numpy as np
import pandas as pd
import os


def _aggregate_features(df_91days):
    """Compute per-feature statistics over 91 days for one region.

    Args:
        df_91days: DataFrame with 91 rows, columns include meteorological features
                   (region_id and date already dropped before calling).

    Returns:
        pd.Series of aggregated features.
    """
    rows = []
    for col in df_91days.columns:
        series = df_91days[col].dropna()
        if len(series) == 0:
            continue
        rows.append({
            f"{col}__mean": series.mean(),
            f"{col}__std": series.std(),
            f"{col}__min": series.min(),
            f"{col}__max": series.max(),
            f"{col}__q25": series.quantile(0.25),
            f"{col}__q50": series.quantile(0.50),
            f"{col}__q75": series.quantile(0.75),
            f"{col}__last7_mean": series.tail(7).mean(),
            f"{col}__last30_mean": series.tail(30).mean(),
            f"{col}__trend": np.polyfit(np.arange(len(series)), series, 1)[0],
        })
    return pd.concat([pd.Series(r) for r in rows])


def _aggregate_array(window_array, feat_cols):
    """Fast feature aggregation over a numpy array window (n_rows, n_feats)."""
    rows = []
    for j in range(window_array.shape[1]):
        series = window_array[:, j]
        rows.append({
            f"{feat_cols[j]}__mean": series.mean(),
            f"{feat_cols[j]}__std": series.std(),
            f"{feat_cols[j]}__min": series.min(),
            f"{feat_cols[j]}__max": series.max(),
            f"{feat_cols[j]}__last7_mean": series[-7:].mean(),
            f"{feat_cols[j]}__last30_mean": series[-30:].mean(),
        })
    return pd.concat([pd.Series(r) for r in rows])


def load_train_data(path, max_windows_per_region=52):
    """Load train.csv. For each region, take the most recent N weekly windows
    (91 days of features -> 5 weekly scores).

    Args:
        path: path to train.csv
        max_windows_per_region: max windows to extract per region

    Returns:
        X, y, region_ids (list of region_id strings per sample, for grouped CV)
    """
    df = pd.read_csv(path)
    meta_cols = ["region_id", "date", "score"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    score_vals = df["score"].values
    region_vals = df["region_id"].values

    X_list, y_list, region_list = [], [], []

    for region_id, grp in df.groupby("region_id", sort=False):
        indices = grp.index.values
        score_mask = pd.notna(score_vals[indices])
        score_positions = np.where(score_mask)[0]

        # Take only the last N windows
        start_idx = max(0, len(score_positions) - 5 - max_windows_per_region)
        score_positions = score_positions[start_idx:]

        for start in range(0, len(score_positions) - 4):
            label_pos = score_positions[start:start + 5]
            label_values = score_vals[indices[label_pos]]
            if len(label_values) != 5:
                continue
            first_label_pos = label_pos[0]
            if first_label_pos < 91:
                continue
            window_indices = indices[first_label_pos - 91:first_label_pos]
            window = df.iloc[window_indices][feat_cols].fillna(0).values
            feats = _aggregate_array(window, feat_cols)
            X_list.append(feats)
            y_list.append(label_values)
            region_list.append(region_id)

    X = pd.DataFrame(X_list).reset_index(drop=True)
    y = np.array(y_list)
    return X, y, region_list


def _aggregate_array(window_array, feat_cols):
    """Vectorized feature aggregation over a numpy array window (91, n_feats)."""
    rows = []
    for j in range(window_array.shape[1]):
        series = window_array[:, j]
        rows.append({
            f"{feat_cols[j]}__mean": series.mean(),
            f"{feat_cols[j]}__std": series.std(),
            f"{feat_cols[j]}__min": series.min(),
            f"{feat_cols[j]}__max": series.max(),
            f"{feat_cols[j]}__q25": np.quantile(series, 0.25),
            f"{feat_cols[j]}__q50": np.quantile(series, 0.50),
            f"{feat_cols[j]}__q75": np.quantile(series, 0.75),
            f"{feat_cols[j]}__last7_mean": series[-7:].mean(),
            f"{feat_cols[j]}__last30_mean": series[-30:].mean(),
        })
    return pd.concat([pd.Series(r) for r in rows])


def load_test_data(path):
    """Load test.csv and extract the last 91 days of features for each region.

    Returns:
        X: DataFrame of feature vectors
        region_ids: list of region_id strings
    """
    df = pd.read_csv(path)
    meta_cols = ["region_id", "date"]
    if "score" in df.columns:
        meta_cols.append("score")
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X_list, region_list = [], []

    for region_id, grp in df.groupby("region_id", sort=False):
        window = grp[feat_cols].fillna(0).values[-91:]
        feats = _aggregate_array(window, feat_cols)
        X_list.append(feats)
        region_list.append(region_id)

    X = pd.DataFrame(X_list).reset_index(drop=True)
    return X, region_list


def generate_submission(region_ids, preds, output_path):
    """Write submission CSV matching sample_submission.csv format.

    preds shape = (n_regions, 5).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sample_path = os.path.join(os.path.dirname(output_path), "..", "data", "sample_submission.csv")
    sample_path = os.path.normpath(sample_path)

    if os.path.exists(sample_path):
        template = pd.read_csv(sample_path)
        id_col = template.columns[0]
        target_cols = list(template.columns[1:])
        rows = []
        for i, rid in enumerate(region_ids):
            row = {id_col: rid}
            for j, col_name in enumerate(target_cols):
                row[col_name] = preds[i, min(j, preds.shape[1] - 1)]
            rows.append(row)
        sub = pd.DataFrame(rows)
    else:
        rows = []
        for i, rid in enumerate(region_ids):
            row = {"region_id": rid}
            for w in range(preds.shape[1]):
                row[f"pred_week{w+1}"] = preds[i, w]
            rows.append(row)
        sub = pd.DataFrame(rows)

    sub.to_csv(output_path, index=False)
    print(f"Saved {len(sub)} rows -> {output_path}")
