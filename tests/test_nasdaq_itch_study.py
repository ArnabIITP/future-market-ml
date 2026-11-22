import numpy as np
import pandas as pd

from src.nasdaq_itch_study import (
    CLASSES,
    _metrics,
    feature_shift_by_session,
    make_features,
    sessions_for,
)


def test_session_split_is_chronological_and_final_days_are_separate():
    train = sessions_for("train")
    validation = sessions_for("validation")
    test = sessions_for("test")
    assert len(train) == 3
    assert len(validation) == 2
    assert len(test) == 3
    assert max(s.session_date for s in train) < min(s.session_date for s in validation)
    assert max(s.session_date for s in validation) < min(s.session_date for s in test)
    assert not set(s.session_date for s in train + validation) & {
        s.session_date for s in test
    }


def test_target_horizon_does_not_cross_session_or_symbol_boundaries():
    rows = []
    for day, ticker, base in [
        ("2019-08-30", "AAPL", 100_000),
        ("2019-08-30", "MSFT", 200_000),
        ("2019-10-30", "AAPL", 300_000),
    ]:
        for i in range(15):
            rows.append({
                "session_date": day,
                "ticker": ticker,
                "timestamp_ns": i * 100_000_000,
                "mid_price": base + i * 10,
                "spread_bps": 2.0,
                "imbalance_l1": 0.1,
                "imbalance_5": -0.1,
                "microprice_offset_bps": 0.01,
            })
    result = make_features(pd.DataFrame(rows))
    tails = result.groupby(["session_date", "ticker"], sort=False).tail(10)
    assert tails.target.isna().all()
    assert result.target.notna().sum() == 15
    assert result.loc[result.target.notna(), "target"].isin(CLASSES).all()
    reference = result[result.session_date == "2019-08-30"]
    comparison = result[result.session_date == "2019-10-30"]
    shift = feature_shift_by_session(reference, comparison)
    assert len(shift) == 1
    assert shift.iloc[0].features_scored > 0
    assert shift.iloc[0].median_feature_psi >= 0


def test_probability_metrics_include_calibration_and_are_finite():
    y = np.array([-1, 0, 1, -1, 0, 1])
    probability = np.array([
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
    ])
    metrics = _metrics(y, probability)
    assert set(metrics) == {
        "macro_f1", "log_loss", "multiclass_brier", "top_label_ece_10bin",
        "label_share_down", "label_share_flat", "label_share_up",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["log_loss"] > 0
    assert metrics["top_label_ece_10bin"] >= 0
