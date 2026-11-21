"""Input validation and target construction for daily market data."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_prices(frame: pd.DataFrame, date_column: str = "Date", price_column: str = "Adj Close") -> pd.DataFrame:
    """Normalize dates, sort, de-duplicate, and clean price/volume observations."""
    data = frame.copy()
    if date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce", utc=True).dt.tz_localize(None)
        data = data.set_index(date_column)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Input must have a DatetimeIndex or a date_column.")
    if price_column not in data.columns:
        if price_column == "Adj Close" and "Close" in data.columns:
            price_column = "Close"
        else:
            raise ValueError(f"Missing price column: {price_column}")
    data = data.sort_index()
    data = data.loc[~data.index.duplicated(keep="last")]
    data["price"] = pd.to_numeric(data[price_column], errors="coerce")
    data["volume"] = pd.to_numeric(data["Volume"], errors="coerce") if "Volume" in data else np.nan
    data = data.loc[data["price"].gt(0), ["price", "volume"]]
    if data.empty:
        raise ValueError("No positive, usable price observations found.")
    return data


def add_targets(data: pd.DataFrame) -> pd.DataFrame:
    """Add daily log return and next-session log squared-return target."""
    out = data.copy()
    out["log_return"] = np.log(out["price"]).diff()
    out["target_log_variance"] = np.log(out["log_return"].shift(-1).pow(2).clip(lower=1e-12))
    out["target_volatility"] = out["log_return"].shift(-1).abs()
    return out
