import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
from xgboost import XGBRegressor


DEFAULT_PARAMS = {
    "n_estimators": 250,
    "max_depth": 4,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.2,
    "reg_lambda": 1.5,
    "random_state": 42,
    "n_jobs": 2,
}


def train_week_models(X, y, params_override=None):
    params = dict(DEFAULT_PARAMS)
    if params_override:
        params.update(params_override)

    models = []
    for week_idx in range(y.shape[1]):
        model = XGBRegressor(**params)
        model.fit(X, y[:, week_idx])
        models.append(model)
    return models


def predict_week_models(models, X):
    preds = [model.predict(X) for model in models]
    return np.column_stack(preds)


def cv_evaluate_week_models(X, y, groups, n_splits=5, params_override=None):
    kf = GroupKFold(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tqdm(kf.split(X, y, groups), total=n_splits, desc="Temporal tree CV"):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        models = train_week_models(X_tr, y[train_idx], params_override=params_override)
        preds = predict_week_models(models, X_val)
        scores.append(mean_absolute_error(y[val_idx], preds))

    return scores, float(np.mean(scores)), float(np.std(scores))
