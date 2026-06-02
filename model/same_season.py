"""Same-season (month-of-year) severity features for tree model augmentation."""

import numpy as np
import pandas as pd


def build_region_month_severity_map(train_df, window_days=91):
    score_vals = train_df["score"].values
    date_col = pd.to_datetime(train_df["date"], errors="coerce")
    if date_col.isna().any():
        date_col = date_col.fillna(pd.Timestamp("2020-07-01"))

    region_month_stats = {}

    for region_id, grp in train_df.groupby("region_id", sort=False):
        indices = grp.index.values
        dates = date_col.iloc[indices]
        months = dates.dt.month.values

        month_stats = {}
        for m in range(1, 13):
            mask = months == m
            scores = score_vals[indices][mask]
            scores = scores[~np.isnan(scores)]
            if len(scores) == 0:
                month_stats[m] = {"mean": 0.0, "median": 0.0, "count": 0.0, "zero_frac": 0.0}
            else:
                month_stats[m] = {
                    "mean": float(scores.mean()),
                    "median": float(np.median(scores)),
                    "count": float(len(scores)),
                    "zero_frac": float(np.mean(scores == 0.0)),
                }
        region_month_stats[region_id] = month_stats

    return region_month_stats


def build_train_same_season_features(train_df, max_windows_per_region=None, window_days=91):
    score_vals = train_df["score"].values
    date_col = pd.to_datetime(train_df["date"], errors="coerce")
    if date_col.isna().any():
        date_col = date_col.fillna(pd.Timestamp("2020-07-01"))

    epoch_vals = (date_col - pd.Timestamp("1970-01-01")).dt.days.values

    region_month_map = build_region_month_severity_map(train_df, window_days)

    rows = []
    grouped = train_df.groupby("region_id", sort=False)
    for region_id, grp in grouped:
        indices = grp.index.values
        score_mask = pd.notna(score_vals[indices])
        score_positions = np.where(score_mask)[0]

        if max_windows_per_region is not None:
            start_idx = max(0, len(score_positions) - 5 - max_windows_per_region)
            score_positions = score_positions[start_idx:]

        month_stats = region_month_map[region_id]

        for start in range(0, len(score_positions) - 4):
            label_pos = score_positions[start:start + 5]
            first_label_pos = label_pos[0]
            if first_label_pos < window_days:
                continue

            cutoff_pos = first_label_pos - window_days
            cutoff_abs = epoch_vals[indices[cutoff_pos]]

            target_month = pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(cutoff_abs))
            target_month = target_month.month

            st = month_stats[target_month]
            rows.append({
                "season_month": float(target_month),
                "season_mean": st["mean"],
                "season_median": st["median"],
                "season_count": st["count"],
                "season_zero_frac": st["zero_frac"],
            })

    return pd.DataFrame(rows).reset_index(drop=True)


def build_test_same_season_features(train_df, test_df, window_days=91):
    test_dates = pd.to_datetime(test_df["date"], errors="coerce")
    if test_dates.isna().any():
        test_dates = test_dates.fillna(pd.Timestamp("2020-07-01"))

    region_month_map = build_region_month_severity_map(train_df, window_days)

    rows = []
    for region_id, grp in test_df.groupby("region_id", sort=False):
        last_date = test_dates.iloc[grp.index[-1]]
        target_month = last_date.month

        st = region_month_map.get(region_id, {}).get(target_month, {
            "mean": 0.0, "median": 0.0, "count": 0.0, "zero_frac": 0.0,
        })
        rows.append({
            "season_month": float(target_month),
            "season_mean": st["mean"],
            "season_median": st["median"],
            "season_count": st["count"],
            "season_zero_frac": st["zero_frac"],
        })

    return pd.DataFrame(rows).reset_index(drop=True)
