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


def load_train_data(path):
    """Load train.csv and construct 91-day sliding window samples.

    For each region, for each consecutive block of 5 weekly scores that has
    91 preceding days of data, extract aggregated features and the 5 scores as label.

    Returns:
        X: DataFrame of feature vectors
        y: np.ndarray of shape (n_samples, 5)
        region_ids: list of region_id strings per sample
    """
    df = pd.read_csv(path, parse_dates=["date"])
    meta_cols = ["region_id", "date", "score"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X_list, y_list, region_list = [], [], []

    for region_id, grp in df.groupby("region_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        score_idx = grp.index[grp["score"].notna()].tolist()

        for start in range(len(score_idx) - 4):
            label_indices = score_idx[start:start + 5]
            label_values = grp.loc[label_indices, "score"].values
            if len(label_values) != 5:
                continue
            first_label_idx = label_indices[0]
            if first_label_idx < 91:
                continue
            window = grp.iloc[first_label_idx - 91:first_label_idx]
            feats = _aggregate_features(window[feat_cols])
            X_list.append(feats)
            y_list.append(label_values)
            region_list.append(region_id)

    X = pd.DataFrame(X_list).reset_index(drop=True)
    y = np.array(y_list)
    return X, y, region_list


def load_test_data(path):
    """Load test.csv and extract the last 91 days of features for each region.

    Returns:
        X: DataFrame of feature vectors
        region_ids: list of region_id strings
    """
    df = pd.read_csv(path, parse_dates=["date"])
    meta_cols = ["region_id", "date"]
    if "score" in df.columns:
        meta_cols.append("score")
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X_list, region_list = [], []

    for region_id, grp in df.groupby("region_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        window = grp.iloc[-91:]
        feats = _aggregate_features(window[feat_cols])
        X_list.append(feats)
        region_list.append(region_id)

    X = pd.DataFrame(X_list).reset_index(drop=True)
    return X, region_list


def generate_submission(region_ids, preds, output_path):
    """Write submission CSV matching sample_submission.csv format.

    Reads sample_submission.csv to get the exact column names,
    then fills in predictions where preds shape = (n_regions, 5).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sample_path = os.path.join(os.path.dirname(output_path), "..", "data", "sample_submission.csv")
    sample_path = os.path.normpath(sample_path)

    if os.path.exists(sample_path):
        template = pd.read_csv(sample_path)
        target_cols = [c for c in template.columns if c != "Id"]
        n_cols = len(target_cols)
        rows = []
        for i, rid in enumerate(region_ids):
            row = {"Id": rid}
            for j, col_name in enumerate(target_cols):
                row[col_name] = preds[i, min(j, preds.shape[1] - 1)]
            rows.append(row)
        sub = pd.DataFrame(rows)
    else:
        rows = []
        for i, rid in enumerate(region_ids):
            row = {"Id": rid}
            for w in range(preds.shape[1]):
                row[f"Week_{w+1}"] = preds[i, w]
            rows.append(row)
        sub = pd.DataFrame(rows)

    sub.to_csv(output_path, index=False)
    print(f"Saved {len(sub)} rows -> {output_path}")
