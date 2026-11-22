import numpy as np
import pandas as pd
import pytest

from src.run_lobster_sample import (class_order_probabilities, make_targets,
                                    build_execution_volume_index,
                                    simulate_passive_queue, simulate_taker_touch)


class _Model:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, x):
        return np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])


def test_class_probabilities_are_mapped_to_down_flat_up():
    actual = class_order_probabilities(_Model(), pd.DataFrame(index=range(2)), encoded=True)
    np.testing.assert_allclose(actual, [[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    np.testing.assert_allclose(actual.sum(axis=1), 1.0)


def test_future_midprice_labels_drop_incomplete_horizon():
    frame = pd.DataFrame({
        "mid_ticks": [100.0, 100.0, 101.0, 100.0, 100.0],
        "signed_event_size": [0.0, 1.0, -1.0, 0.0, 0.0],
    })
    prepared = make_targets(frame, horizon=2, flat_threshold=0.5)
    assert prepared["target"].iloc[:3].tolist() == [1, 0, -1]
    assert prepared["target"].iloc[3:].isna().all()
    assert len(prepared) == 5
    assert prepared["flow_10"].iloc[-1] == 0.0


def test_taker_touch_charges_spread_and_round_trip_fee_without_overlap():
    frame = pd.DataFrame({
        "best_bid_ticks": np.full(102, 9999.0),
        "best_ask_ticks": np.full(102, 10001.0),
        "bid_size_1": np.full(102, 10.0),
        "ask_size_1": np.full(102, 10.0),
    })
    for level in range(1, 6):
        frame[f"ask_price_level{level}"] = np.full(102, 10001.0 + (level - 1) * 2)
        frame[f"ask_size_level{level}"] = np.full(102, 10.0)
        frame[f"bid_price_level{level}"] = np.full(102, 9999.0 - (level - 1) * 2)
        frame[f"bid_size_level{level}"] = np.full(102, 10.0)
    probability = np.tile([0.1, 0.1, 0.8], (2, 1))
    rows = simulate_taker_touch(frame, 0, probability, "TEST", "stub")
    assert rows[0]["n_trades"] == 1
    assert rows[0]["gross_mean_bps_per_trade"] == pytest.approx(-2.0)
    assert rows[1]["net_mean_bps_per_trade"] == pytest.approx(-12.0)


def test_five_level_sweep_accounts_for_partial_fill_and_price_walk():
    from src.run_lobster_sample import sweep_vwap

    row = pd.Series({
        **{f"ask_price_level{level}": 10000.0 + level for level in range(1, 6)},
        **{f"ask_size_level{level}": (2.0 if level <= 2 else 0.0) for level in range(1, 6)},
        **{f"bid_price_level{level}": 9999.0 - level for level in range(1, 6)},
        **{f"bid_size_level{level}": (2.0 if level <= 2 else 0.0) for level in range(1, 6)},
    })
    filled, vwap = sweep_vwap(row, buy=True, quantity=5.0)
    assert filled == 4.0
    assert vwap == pytest.approx((2 * 10001 + 2 * 10002) / 4)


def test_passive_queue_proxy_requires_queue_ahead_to_be_consumed():
    frame = pd.DataFrame({
        "event_type": np.zeros(102),
        "event_price_ticks": np.zeros(102),
        "direction": np.ones(102),
        "order_size": np.zeros(102),
        "mid_ticks": np.full(102, 10000.0),
        "best_bid_ticks": np.full(102, 9999.0),
        "best_ask_ticks": np.full(102, 10001.0),
        "bid_size_1": np.full(102, 10.0),
        "ask_size_1": np.full(102, 10.0),
    })
    frame.loc[2, ["event_type", "event_price_ticks", "direction", "order_size"]] = [4, 9999, 1, 15]
    for level in range(1, 6):
        frame[f"ask_price_level{level}"] = np.full(102, 10001.0 + (level - 1) * 2)
        frame[f"ask_size_level{level}"] = np.full(102, 10.0)
        frame[f"bid_price_level{level}"] = np.full(102, 9999.0 - (level - 1) * 2)
        frame[f"bid_size_level{level}"] = np.full(102, 10.0)
    index = build_execution_volume_index(frame)
    probability = np.tile([0.1, 0.1, 0.8], (2, 1))
    rows = simulate_passive_queue(frame, 0, probability, "TEST", "stub", index,
                                  latency_events=(1,), order_sizes=(10.0,),
                                  queue_multipliers=(1.0, 2.0))
    fill_1x = next(row for row in rows if row["queue_ahead_multiplier"] == 1.0)
    fill_2x = next(row for row in rows if row["queue_ahead_multiplier"] == 2.0)
    assert fill_1x["entry_fill_probability"] == pytest.approx(1.0)
    assert fill_1x["unconditional_entry_fill_ratio"] == pytest.approx(0.5)
    assert fill_1x["partial_entry_orders"] == 1
    assert fill_2x["entry_fill_probability"] == pytest.approx(0.0)
