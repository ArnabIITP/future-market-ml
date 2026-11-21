"""Leakage-conscious features available at each daily forecast origin."""
from __future__ import annotations

import pandas as pd

from .preprocessing import add_targets


def make_dataset(data: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 21, 63)) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return features, log-variance target, and volatility-unit target."""
    frame = add_targets(data)
    features = pd.DataFrame(index=frame.index)
    for lag in (1, 2, 3, 5, 10):
        features[f"return_lag_{lag}"] = frame["log_return"].shift(lag - 1)
        features[f"abs_return_lag_{lag}"] = frame["log_return"].abs().shift(lag - 1)
    for window in windows:
        features[f"vol_{window}"] = frame["log_return"].rolling(window, min_periods=window).std()
        features[f"mean_abs_return_{window}"] = frame["log_return"].abs().rolling(window, min_periods=window).mean()
    volume_change = frame["volume"].replace(0, float("nan")).pct_change(fill_method=None)
    features["volume_change"] = volume_change.clip(-5, 5)
    features["volume_change_missing"] = volume_change.isna().astype(float)
    joined = features.join(frame[["target_log_variance", "target_volatility"]]).replace([float("inf"), -float("inf")], float("nan")).dropna()
    x = joined[features.columns]
    return x, joined["target_log_variance"], joined["target_volatility"]
