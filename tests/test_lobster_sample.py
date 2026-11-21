import numpy as np
import pandas as pd
import pytest

from src.run_lobster_sample import (class_order_probabilities, make_targets,
                                    simulate_taker_touch)


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
    probability = np.tile([0.1, 0.1, 0.8], (2, 1))
    rows = simulate_taker_touch(frame, 0, probability, "TEST", "stub")
    assert rows[0]["n_trades"] == 1
    assert rows[0]["gross_mean_bps_per_trade"] == pytest.approx(-2.0)
    assert rows[1]["net_mean_bps_per_trade"] == pytest.approx(-12.0)
