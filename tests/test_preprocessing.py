import pandas as pd

from src.features import make_dataset
from src.preprocessing import prepare_prices


def test_prepare_prices_sorts_deduplicates_and_uses_close_fallback():
    raw = pd.DataFrame({
        "Date": ["2020-01-02", "2020-01-01", "2020-01-02"],
        "Close": [101.0, 100.0, 102.0],
    })
    result = prepare_prices(raw)
    assert list(result.index) == list(pd.to_datetime(["2020-01-01", "2020-01-02"]))
    assert result.loc["2020-01-02", "price"] == 102.0


def test_target_is_next_session_absolute_log_return():
    dates = pd.date_range("2020-01-01", periods=75, freq="B")
    price = pd.Series([100 * (1.01 ** i) for i in range(75)], index=dates)
    data = pd.DataFrame({"price": price, "volume": 1000.0})
    x, log_var, volatility = make_dataset(data, windows=(5,))
    t = x.index[10]
    expected = abs(__import__("math").log(price.loc[dates[11]] / price.loc[dates[10]]))
    assert abs(volatility.loc[t] - expected) < 1e-12
    assert abs(log_var.loc[t] - __import__("math").log(expected ** 2)) < 1e-10
