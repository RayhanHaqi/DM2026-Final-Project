import importlib.util

import numpy as np
import pandas as pd


def add_calendar_features(df):
    out = df.copy()
    dates = pd.to_datetime(out["date"])
    day_of_year = dates.dt.dayofyear.astype("float32")
    month = dates.dt.month.astype("float32")
    week = dates.dt.isocalendar().week.astype("float32")

    out["calendar__doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 366.0).astype("float32")
    out["calendar__doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 366.0).astype("float32")
    out["calendar__month_sin"] = np.sin(2.0 * np.pi * month / 12.0).astype("float32")
    out["calendar__month_cos"] = np.cos(2.0 * np.pi * month / 12.0).astype("float32")
    out["calendar__week_sin"] = np.sin(2.0 * np.pi * week / 53.0).astype("float32")
    out["calendar__week_cos"] = np.cos(2.0 * np.pi * week / 53.0).astype("float32")
    return out


def require_deep_learning_backend():
    if importlib.util.find_spec("torch") is not None:
        return "torch"
    if importlib.util.find_spec("tensorflow") is not None:
        return "tensorflow"
    raise RuntimeError("Small 1D CNN requires PyTorch or TensorFlow. Install one before generating a CNN submission.")


def build_sequence_train_data_from_frame(df, max_windows_per_region=52, include_calendar=False):
    if include_calendar:
        df = add_calendar_features(df)
    meta_cols = ["region_id", "date", "score"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    score_vals = df["score"].values
    X_list, y_list, region_list = [], [], []

    for region_id, grp in df.groupby("region_id", sort=False):
        indices = grp.index.values
        score_positions = np.where(pd.notna(score_vals[indices]))[0]

        if max_windows_per_region is not None:
            start_idx = max(0, len(score_positions) - 5 - max_windows_per_region)
            score_positions = score_positions[start_idx:]

        for start in range(0, len(score_positions) - 4):
            label_pos = score_positions[start:start + 5]
            first_label_pos = label_pos[0]
            if first_label_pos < 91:
                continue
            window_indices = indices[first_label_pos - 91:first_label_pos]
            X_list.append(df.iloc[window_indices][feat_cols].fillna(0).values.astype("float32"))
            y_list.append(score_vals[indices[label_pos]].astype("float32"))
            region_list.append(region_id)

    return np.array(X_list, dtype="float32"), np.array(y_list, dtype="float32"), region_list, feat_cols


def build_sequence_test_data_from_frame(df, feat_cols):
    X_list, region_list = [], []
    for region_id, grp in df.groupby("region_id", sort=False):
        X_list.append(grp[feat_cols].fillna(0).values[-91:].astype("float32"))
        region_list.append(region_id)
    return np.array(X_list, dtype="float32"), region_list


def standardize_sequences(X_train, X_test):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (X_train - mean) / std, (X_test - mean) / std


def build_torch_model(model_name, n_features, dropout=0.15):
    backend = require_deep_learning_backend()
    if backend != "torch":
        raise RuntimeError("This CNN runner currently supports PyTorch. Install torch before running it.")

    import torch

    class SmallCnn(torch.nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(32, 64, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(64, 5),
            )

        def forward(self, x):
            return self.net(x.transpose(1, 2))

    class V2Cnn(torch.nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
                torch.nn.BatchNorm1d(32),
                torch.nn.ReLU(),
                torch.nn.Conv1d(32, 64, kernel_size=5, padding=2),
                torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(),
                torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 5),
            )

        def forward(self, x):
            return self.net(x.transpose(1, 2))

    class CnnGru(torch.nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv1d(32, 64, kernel_size=5, padding=2),
                torch.nn.ReLU(),
            )
            self.gru = torch.nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
            self.dropout = torch.nn.Dropout(dropout)
            self.head = torch.nn.Linear(64, 5)

        def forward(self, x):
            features = self.conv(x.transpose(1, 2)).transpose(1, 2)
            _, hidden = self.gru(features)
            return self.head(self.dropout(hidden[-1]))

    if model_name == "small":
        return SmallCnn(n_features)
    if model_name == "v2":
        return V2Cnn(n_features)
    if model_name == "cnn_gru":
        return CnnGru(n_features)
    raise ValueError("model_name must be 'small', 'v2', or 'cnn_gru'")


def train_torch_cnn(
    X_train,
    y_train,
    groups,
    epochs=25,
    batch_size=256,
    lr=1e-3,
    seed=42,
    model_name="small",
    patience=5,
    dropout=0.15,
    weight_decay=1e-3,
    scheduler=False,
):
    backend = require_deep_learning_backend()
    if backend != "torch":
        raise RuntimeError("This CNN runner currently supports PyTorch. Install torch before running it.")

    import torch
    from sklearn.model_selection import GroupShuffleSplit

    torch.manual_seed(seed)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(X_train, y_train, groups))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torch_model(model_name, X_train.shape[2], dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2) if scheduler else None
    loss_fn = torch.nn.L1Loss()

    X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X_train[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32).to(device)

    best_state = None
    best_val = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X_tr))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            xb = X_tr[batch_idx].to(device)
            yb = y_tr[batch_idx].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mae = loss_fn(model(X_val), y_val).item()
        if lr_scheduler is not None:
            lr_scheduler.step(val_mae)
        print(f"epoch={epoch} val_mae={val_mae:.6f}")

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val


def predict_torch_cnn(model, X_test, batch_size=512):
    import torch

    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            xb = torch.tensor(X_test[start:start + batch_size], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.vstack(preds)
