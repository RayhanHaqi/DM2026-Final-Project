import numpy as np
import pandas as pd
import os


def _aggregate_array(window_array, feat_cols):
    rows = []
    for j in range(window_array.shape[1]):
        series = window_array[:, j]
        name = feat_cols[j]
        mu = series.mean()
        sigma = series.std()
        rows.append({
            f"{name}__mean": mu,
            f"{name}__std": sigma,
            f"{name}__min": series.min(),
            f"{name}__max": series.max(),
            f"{name}__q25": np.quantile(series, 0.25),
            f"{name}__q50": np.quantile(series, 0.50),
            f"{name}__q75": np.quantile(series, 0.75),
            f"{name}__last7_mean": series[-7:].mean(),
            f"{name}__last30_mean": series[-30:].mean(),
            f"{name}__trend": np.polyfit(np.arange(len(series)), series, 1)[0],
            f"{name}__skew": np.mean((series - mu) ** 3) / (sigma ** 3 + 1e-10),
            f"{name}__kurt": np.mean((series - mu) ** 4) / (sigma ** 4 + 1e-10),
        })
    return pd.concat([pd.Series(r) for r in rows])


def load_train_data(path, max_windows_per_region=None):
    """Load train.csv and construct sliding window samples.
    max_windows_per_region=None means use ALL available windows.
    Uses .npy cache for features/labels after first run.
    """
    cache_x = path.replace(".csv", "_X_v2.npy")
    cache_y = path.replace(".csv", "_y_v2.npy")
    cache_r = path.replace(".csv", "_regions_v2.npy")
    cache_w = str(max_windows_per_region)

    if os.path.exists(cache_x):
        X = pd.DataFrame(np.load(cache_x, allow_pickle=True))
        y = np.load(cache_y)
        regions = list(np.load(cache_r, allow_pickle=True))
        if len(X) > 0 and X.shape[1] > 0:
            return X, y, regions

    X, y, regions = _build_train_features(path, max_windows_per_region)
    np.save(cache_x, X.values)
    np.save(cache_y, y)
    np.save(cache_r, np.array(regions, dtype=object))
    return X, y, regions


def _build_train_features(path, max_windows_per_region):
    df = pd.read_csv(path)
    meta_cols = ["region_id", "date", "score"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    score_vals = df["score"].values

    X_list, y_list, region_list = [], [], []

    for region_id, grp in df.groupby("region_id", sort=False):
        indices = grp.index.values
        score_mask = pd.notna(score_vals[indices])
        score_positions = np.where(score_mask)[0]

        if max_windows_per_region is not None:
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


def _build_test_features(path):
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


def load_test_data(path):
    cache_x = path.replace(".csv", "_X_test_v2.npy")
    cache_r = path.replace(".csv", "_regions_test_v2.npy")
    if os.path.exists(cache_x):
        X = pd.DataFrame(np.load(cache_x, allow_pickle=True))
        regions = list(np.load(cache_r, allow_pickle=True))
        if len(X) > 0:
            return X, regions
    X, regions = _build_test_features(path)
    np.save(cache_x, X.values)
    np.save(cache_r, np.array(regions, dtype=object))
    return X, regions


def generate_submission(region_ids, preds, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Version tracking
    base, ext = os.path.splitext(output_path)
    version = 1
    while os.path.exists(f"{base}_v{version}{ext}"):
        version += 1
    versioned_path = f"{base}_v{version}{ext}"

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

    sub.to_csv(versioned_path, index=False)
    print(f"Saved {len(sub)} rows -> {versioned_path}")

    # Auto-track
    tracker_path = os.path.join(os.path.dirname(output_path) or ".", "SUBMISSIONS.md")
    from datetime import datetime
    date_str = datetime.now().strftime("%b %d")
    entry = f"| {os.path.basename(versioned_path)} | {date_str} | ? | ? | ? | ? | auto-generated |\n"
    if not os.path.exists(tracker_path):
        with open(tracker_path, "w") as f:
            f.write("# Final Project Submission Tracker\n\n| File | Date | Score | Model | Features | Notes |\n|------|------|-------|-------|----------|-------|\n")
    with open(tracker_path, "a") as f:
        f.write(entry)
