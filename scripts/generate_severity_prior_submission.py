import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from model.experiments import build_submission, validate_submission
from model.severity_history import build_region_test_history


def main():
    train = pd.read_csv("data/train.csv", usecols=["region_id", "score"])
    sample = pd.read_csv("data/sample_submission.csv")
    history = build_region_test_history(train)
    preds = []
    for region_id in sample.iloc[:, 0]:
        if region_id in history.index:
            value = history.loc[region_id, "score_mean_last20"]
        else:
            value = train["score"].dropna().mean()
        preds.append([value] * 5)
    preds = np.clip(np.asarray(preds, dtype=float), 0.0, 5.0)
    submission = build_submission(sample.iloc[:, 0].tolist(), preds, sample)
    validate_submission(submission, sample)
    out_path = "output/daily_candidates/severity_prior_recent_mean_20260524.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved severity prior submission: {out_path}")


if __name__ == "__main__":
    main()
