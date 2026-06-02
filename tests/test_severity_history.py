import numpy as np
import pandas as pd

from model.severity_history import (
    build_region_test_history,
    build_test_blackout_history_features_from_frame,
    build_test_history_features_from_frame,
    build_train_blackout_history_features_from_frame,
    build_train_history_features_from_frame,
    summarize_score_history,
)


def test_summarize_score_history_uses_only_scores_before_cutoff():
    scores = np.array([np.nan, 0.0, 1.0, 2.0, 5.0, 4.0])

    features = summarize_score_history(scores, cutoff_pos=4)

    assert features["score_last_known"] == 2.0
    assert features["score_mean_all"] == 1.0
    assert features["score_median_all"] == 1.0
    assert features["score_mean_last5"] == 1.0
    assert features["score_class_frac_0"] == 1 / 3
    assert features["score_class_frac_1"] == 1 / 3
    assert features["score_class_frac_2"] == 1 / 3
    assert features["score_class_frac_5"] == 0.0


def test_summarize_score_history_returns_zero_features_without_history():
    scores = np.array([np.nan, 3.0, 4.0])

    features = summarize_score_history(scores, cutoff_pos=1)

    assert features["score_last_known"] == 0.0
    assert features["score_mean_all"] == 0.0
    assert features["score_zero_frac"] == 0.0
    assert features["score_class_frac_3"] == 0.0


def test_build_region_test_history_uses_all_train_scores_for_region():
    train = pd.DataFrame(
        {
            "region_id": ["R1", "R1", "R2", "R2"],
            "score": [0.0, 2.0, np.nan, 5.0],
        }
    )

    features = build_region_test_history(train)

    r1 = features.loc["R1"]
    r2 = features.loc["R2"]

    assert r1["score_last_known"] == 2.0
    assert r1["score_mean_all"] == 1.0
    assert r2["score_last_known"] == 5.0
    assert r2["score_mean_all"] == 5.0


def test_build_train_history_features_aligns_with_temporal_windows_without_leakage():
    rows = []
    for day in range(97):
        score = np.nan
        if day == 91:
            score = 3.0
        elif day == 92:
            score = 4.0
        elif day == 93:
            score = 5.0
        elif day == 94:
            score = 2.0
        elif day == 95:
            score = 1.0
        elif day == 96:
            score = 0.0
        rows.append({"region_id": "R1", "date": f"2020-01-01", "score": score, "f1": 1.0})
    train = pd.DataFrame(rows)

    features = build_train_history_features_from_frame(train)

    assert len(features) == 2
    assert features.loc[0, "score_last_known"] == 0.0
    assert features.loc[0, "score_mean_all"] == 0.0
    assert features.loc[1, "score_last_known"] == 3.0
    assert features.loc[1, "score_mean_all"] == 3.0
    assert features.loc[1, "score_class_frac_3"] == 1.0
    assert features.loc[1, "score_class_frac_4"] == 0.0


def test_build_test_history_features_aligns_to_test_region_order():
    train = pd.DataFrame(
        {
            "region_id": ["R2", "R1", "R1", "R3"],
            "score": [5.0, 0.0, 2.0, np.nan],
        }
    )

    features = build_test_history_features_from_frame(train, ["R1", "R2", "R3"])

    assert features.loc[0, "score_last_known"] == 2.0
    assert features.loc[0, "score_mean_all"] == 1.0
    assert features.loc[1, "score_last_known"] == 5.0
    assert features.loc[2, "score_last_known"] == 0.0


def test_build_train_blackout_history_features_excludes_weather_window_scores():
    rows = []
    for day in range(97):
        score = np.nan
        if day == 0:
            score = 2.0
        elif day == 91:
            score = 3.0
        elif day == 92:
            score = 4.0
        elif day == 93:
            score = 5.0
        elif day == 94:
            score = 1.0
        elif day == 95:
            score = 0.0
        elif day == 96:
            score = 1.0
        rows.append({"region_id": "R1", "score": score})
    train = pd.DataFrame(rows)

    features = build_train_blackout_history_features_from_frame(train, window_days=91)

    assert len(features) == 2
    assert features.loc[0, "score_history_count"] == 0.0
    assert features.loc[0, "score_last_known"] == 0.0
    assert features.loc[1, "score_history_count"] == 1.0
    assert features.loc[1, "score_last_known"] == 2.0
    assert features.loc[1, "score_class_frac_2"] == 1.0
    assert features.loc[1, "score_class_frac_3"] == 0.0
    assert features.loc[1, "score_history_gap_positions"] == 92.0


def test_build_test_blackout_history_features_uses_all_train_scores_in_test_order():
    train = pd.DataFrame(
        {
            "region_id": ["R2", "R1", "R1", "R3"],
            "score": [5.0, 0.0, 2.0, np.nan],
        }
    )

    features = build_test_blackout_history_features_from_frame(train, ["R1", "R2", "R3"], window_days=91)

    assert features.loc[0, "score_history_count"] == 2.0
    assert features.loc[0, "score_last_known"] == 2.0
    assert features.loc[1, "score_history_count"] == 1.0
    assert features.loc[1, "score_last_known"] == 5.0
    assert features.loc[2, "score_history_count"] == 0.0
    assert features.loc[0, "score_history_gap_positions"] == 91.0
