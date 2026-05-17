import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def train_xgboost(X, y, params_override=None):
    """Train XGBoost multi-output regressor.

    Args:
        X: feature DataFrame/array
        y: target array of shape (n_samples, 5)
        params_override: optional dict overriding defaults

    Returns:
        Fitted MultiOutputRegressor
    """
    defaults = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params_override:
        defaults.update(params_override)

    model = MultiOutputRegressor(XGBRegressor(**defaults))
    model.fit(X, y)
    return model


def cv_evaluate(X, y, groups, n_splits=5):
    """GroupKFold CV by region — avoids data leakage between windows of same region.

    Returns:
        (per_fold_scores, mean, std)
    """
    kf = GroupKFold(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in kf.split(X, y, groups):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]

        fold_model = train_xgboost(X_tr, y_tr)
        preds = fold_model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        scores.append(mae)

    return scores, np.mean(scores), np.std(scores)
