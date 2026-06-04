"""Feature matrices for ordinal threshold classifiers."""

import pandas as pd

from model import temporal_features
from model.same_season import build_train_same_season_features, build_test_same_season_features

ORDINAL_FEATURE_SETS = ("hybrid", "hybrid_season")


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
    test_df = pd.read_csv(test_path)
    season_train = build_train_same_season_features(
        train_df, max_windows_per_region=max_windows_per_region
    )
    season_test = build_test_same_season_features(train_df, test_df)

    if len(season_train) != len(X_train):
        raise ValueError(
            f"Train row mismatch: hybrid {len(X_train)} vs season {len(season_train)}"
        )
    if len(season_test) != len(X_test):
        raise ValueError(
            f"Test row mismatch: hybrid {len(X_test)} vs season {len(season_test)}"
        )

    X_train = pd.concat([X_train.reset_index(drop=True), season_train], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), season_test], axis=1)
    return X_train, y_train, train_regions, X_test, test_regions
