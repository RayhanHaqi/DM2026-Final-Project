import os
from datetime import datetime

import numpy as np
import pandas as pd

from model import train


DEFAULT_DAILY_CANDIDATES = [
    {
        "name": "xgb_a_depth4_300",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.04,
            "max_depth": 4,
            "n_jobs": 2,
        },
    },
    {
        "name": "xgb_b_depth5_300",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 5,
            "n_jobs": 2,
        },
    },
    {
        "name": "xgb_c_reg_depth4_200",
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "reg_alpha": 0.3,
            "reg_lambda": 2.0,
            "n_jobs": 2,
        },
    },
]


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


def select_top_candidates(results, limit=3):
    return sorted(results, key=lambda result: result["cv_mae"])[:limit]


def _write_validated_submission(sub, sample, output_dir, name, timestamp):
    ok, messages = validate_submission(sub, sample)
    if not ok:
        raise ValueError(f"Invalid submission {name}: {'; '.join(messages)}")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}_{timestamp}.csv")
    sub.to_csv(path, index=False)
    return path


def run_daily_candidates(
    X_train,
    y_train,
    train_regions,
    X_test,
    test_regions,
    sample_path="data/sample_submission.csv",
    current_best_path=None,
    output_dir="output/daily_candidates",
    candidates=None,
    limit=3,
    n_splits=5,
    timestamp=None,
):
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    sample = pd.read_csv(sample_path)
    candidates = candidates or DEFAULT_DAILY_CANDIDATES
    evaluated = []

    for candidate in candidates:
        scores, mean_mae, std_mae = train.cv_evaluate(
            X_train,
            y_train,
            train_regions,
            n_splits=n_splits,
            params_override=candidate["params"],
        )
        model = train.train_xgboost(X_train, y_train, params_override=candidate["params"])
        preds = clip_predictions(model.predict(X_test))
        sub = build_submission(test_regions, preds, sample)
        evaluated.append({
            "name": candidate["name"],
            "cv_mae": mean_mae,
            "cv_std": std_mae,
            "cv_scores": scores,
            "params": candidate["params"],
            "preds": preds,
            "submission": sub,
        })

    ranked = select_top_candidates(evaluated, limit=2 if current_best_path else limit)

    if current_best_path:
        current_best = pd.read_csv(current_best_path)
        current_preds = current_best.iloc[:, 1:].to_numpy()
        best = ranked[0]
        blend_preds = clip_predictions(blend_predictions(best["preds"], current_preds, candidate_weight=0.7))
        blend_sub = sample.copy().astype({col: float for col in sample.columns[1:]})
        blend_sub.iloc[:, 1:] = blend_preds
        ranked.append({
            "name": f"blend_{best['name']}_070",
            "cv_mae": best["cv_mae"],
            "cv_std": best["cv_std"],
            "cv_scores": best["cv_scores"],
            "params": {"blend_candidate": best["name"], "candidate_weight": 0.7},
            "preds": blend_preds,
            "submission": blend_sub,
        })

    selected = ranked[:limit]
    summary_rows = []
    for result in selected:
        path = _write_validated_submission(result["submission"], sample, output_dir, result["name"], timestamp)
        result["path"] = path
        summary_rows.append({
            "name": result["name"],
            "cv_mae": result["cv_mae"],
            "cv_std": result["cv_std"],
            "path": path,
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, f"summary_{timestamp}.csv"), index=False)
    return selected
