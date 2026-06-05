"""Feature matrices for ordinal threshold classifiers."""

import pandas as pd

from model import severity_history, temporal_features
from model.same_season import build_train_same_season_features, build_test_same_season_features

ORDINAL_FEATURE_SETS = ("hybrid", "hybrid_season", "hybrid_blackout", "history_only")
BLACKOUT_WINDOW_DAYS = 91


def _concat_aligned_features(X, extra, label):
    if len(X) != len(extra):
        raise ValueError(
            f"{label} row mismatch: hybrid {len(X)} vs {label} {len(extra)}"
        )
    X_reset = X.reset_index(drop=True)
    extra_reset = extra.reset_index(drop=True)
    return pd.concat([X_reset, extra_reset], axis=1)


def load_ordinal_train_test(
    train_path,
    test_path,
    max_windows_per_region=52,
    feature_set="hybrid",
):
    """Load aligned train/test features for ordinal caching or training.

    Returns X_train, y_train, train_regions, X_test, test_regions. Row order matches
    temporal_features window construction; test_regions follows test groupby order
    (blend scripts reorder to sample_submission via reorder_class_probs).
    """
    if feature_set not in ORDINAL_FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {ORDINAL_FEATURE_SETS}")

    X_train, y_train, train_regions = temporal_features.load_temporal_train_data(
        train_path,
        max_windows_per_region=max_windows_per_region,
        feature_set="hybrid",
    )
    X_test, test_regions = temporal_features.load_temporal_test_data(
        test_path, feature_set="hybrid"
    )

    if feature_set == "hybrid":
        return X_train, y_train, train_regions, X_test, test_regions

    train_df = pd.read_csv(train_path)

    if feature_set == "history_only":
        _, y_train, train_regions = temporal_features.load_temporal_train_data(
            train_path,
            max_windows_per_region=max_windows_per_region,
            feature_set="hybrid",
        )
        _, test_regions = temporal_features.load_temporal_test_data(
            test_path, feature_set="hybrid"
        )
        X_train = severity_history.build_train_blackout_history_features_from_frame(
            train_df,
            max_windows_per_region=max_windows_per_region,
            window_days=BLACKOUT_WINDOW_DAYS,
        )
        X_test = severity_history.build_test_blackout_history_features_from_frame(
            train_df,
            test_regions,
            window_days=BLACKOUT_WINDOW_DAYS,
        )
        if len(X_train) != len(y_train):
            raise ValueError(
                f"history_only row mismatch: history {len(X_train)} vs labels {len(y_train)}"
            )
        X_train = X_train.add_prefix("history__")
        X_test = X_test.add_prefix("history__")
        return X_train, y_train, train_regions, X_test, test_regions

    if feature_set == "hybrid_season":
        test_df = pd.read_csv(test_path)
        season_train = build_train_same_season_features(
            train_df, max_windows_per_region=max_windows_per_region
        )
        season_test = build_test_same_season_features(train_df, test_df)
        X_train = _concat_aligned_features(X_train, season_train, "season")
        X_test = _concat_aligned_features(X_test, season_test, "season")
        return X_train, y_train, train_regions, X_test, test_regions

    train_history = severity_history.build_train_blackout_history_features_from_frame(
        train_df,
        max_windows_per_region=max_windows_per_region,
        window_days=BLACKOUT_WINDOW_DAYS,
    )
    test_history = severity_history.build_test_blackout_history_features_from_frame(
        train_df,
        test_regions,
        window_days=BLACKOUT_WINDOW_DAYS,
    )
    train_history = train_history.add_prefix("history__")
    test_history = test_history.add_prefix("history__")
    X_train = _concat_aligned_features(X_train, train_history, "blackout")
    X_test = _concat_aligned_features(X_test, test_history, "blackout")
    return X_train, y_train, train_regions, X_test, test_regions
