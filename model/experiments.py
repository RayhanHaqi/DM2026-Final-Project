import os

import numpy as np
import pandas as pd


def set_thread_limits(n_threads):
    """Set BLAS/OpenMP thread env vars for repeatable training runs."""
    value = str(int(n_threads))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = value


def clip_predictions(preds):
    return np.clip(preds, 0.0, 5.0)


def blend_predictions(candidate, current_best, candidate_weight):
    return candidate_weight * candidate + (1.0 - candidate_weight) * current_best


def build_submission(region_ids, preds, sample):
    id_col = sample.columns[0]
    target_cols = list(sample.columns[1:])
    pred_by_region = {rid: preds[i] for i, rid in enumerate(region_ids)}
    rows = []

    for rid in sample[id_col]:
        row = {id_col: rid}
        for j, col_name in enumerate(target_cols):
            row[col_name] = pred_by_region[rid][min(j, preds.shape[1] - 1)]
        rows.append(row)

    return pd.DataFrame(rows, columns=list(sample.columns))


def validate_submission(sub, sample):
    messages = []
    if list(sub.columns) != list(sample.columns):
        messages.append("Columns do not match sample submission")
    if sub.shape != sample.shape:
        messages.append("Shape does not match sample submission")
    if sub.iloc[:, 0].tolist() != sample.iloc[:, 0].tolist():
        messages.append("ID order does not match sample submission")
    if sub.isna().any().any():
        messages.append("Submission contains null values")
    return len(messages) == 0, messages


