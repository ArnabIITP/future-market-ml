import pandas as pd

from src.nasdaq_queue_simulation import OrderBook, simulate_passive_signals


def test_order_book_applies_cancel_replace_and_fifo_add_order():
    events = pd.DataFrame([
        [1, 1, "AAPL", "add", 1, None, "B", 100, 10],
        [2, 2, "AAPL", "replace", 1, 3, "B", 99, 7],
        [3, 3, "AAPL", "add", 2, None, "B", 100, 4],
        [4, 4, "AAPL", "cancel", 2, None, "B", 100, 4],
    ], columns=["timestamp_ns", "sequence", "ticker", "kind", "order_id",
                "new_order_id", "side", "price", "quantity"])
    book = OrderBook()
    for event in events.itertuples(index=False):
        book.apply(event)
    assert book.best("B") == (99, 7)
    assert book.orders == {3: ("B", 99, 7)}


def test_passive_fill_waits_for_initial_queue_to_deplete_and_tracks_partial_size():
    events = pd.DataFrame([
        [1, 1, "AAPL", "add", 1, None, "B", 100, 10],
        [1, 2, "AAPL", "add", 2, None, "S", 102, 10],
        [1, 3, "AAPL", "add", 3, None, "B", 99, 10],
        [2, 4, "AAPL", "cancel", 1, None, "B", 100, 10],
        [3, 5, "AAPL", "add", 4, None, "B", 100, 5],
        [4, 6, "AAPL", "execute", 4, None, "B", 100, 5],
    ], columns=["timestamp_ns", "sequence", "ticker", "kind", "order_id",
                "new_order_id", "side", "price", "quantity"])
    signals = pd.DataFrame([{"timestamp_ns": 1, "prediction": 1,
                             "bid_price": 100, "ask_price": 102}])
    result = simulate_passive_signals(events, signals, delay_ms=0,
                                      order_shares=10, horizon_ms=1,
                                      fee_bps_per_side=0)
    assert result["n_signals"] == 1
    assert result["n_orders_with_entry_fill"] == 1
    assert result["unconditional_entry_fill_ratio"] == 0.5
    assert result["mean_initial_queue_ahead_shares"] == 10
    assert result["priority_anomaly_events"] == 0
