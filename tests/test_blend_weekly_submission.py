import pandas as pd
import pytest

from scripts.blend_weekly_submission import blend_weekly_submission_frames


def _write_submission(path, values):
    pd.DataFrame(
        {
            "ID": ["r1", "r2"],
            "week1": [values[0], values[5]],
            "week2": [values[1], values[6]],
            "week3": [values[2], values[7]],
            "week4": [values[3], values[8]],
            "week5": [values[4], values[9]],
        }
    ).to_csv(path, index=False)


def test_blend_weekly_submission_uses_column_weights(tmp_path):
    candidate_path = tmp_path / "candidate.csv"
    reference_path = tmp_path / "reference.csv"
    _write_submission(candidate_path, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    _write_submission(reference_path, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    blended = blend_weekly_submission_frames(
        candidate_path,
        reference_path,
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )

    assert blended[["week1", "week2", "week3", "week4", "week5"]].iloc[0].tolist() == [1.0, 1.75, 3.5, 5.0, 5.0]
    assert blended[["week1", "week2", "week3", "week4", "week5"]].iloc[1].tolist() == [1.0, 4.25, 5.0, 5.0, 5.0]


def test_blend_weekly_submission_rejects_wrong_weight_count(tmp_path):
    candidate_path = tmp_path / "candidate.csv"
    reference_path = tmp_path / "reference.csv"
    _write_submission(candidate_path, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    _write_submission(reference_path, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    with pytest.raises(ValueError, match="five weekly weights"):
        blend_weekly_submission_frames(candidate_path, reference_path, [0.1, 0.2])
