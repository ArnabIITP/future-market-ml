"""Order-ID-aware passive queue stress for held-out Nasdaq BX ITCH sessions.

The hypothetical order joins behind every displayed order at its limit price.
Observed executions/cancellations/deletions ahead reduce that queue; later
same-price executions can fill the virtual order only after the initial queue
has depleted. This is a small-order counterfactual, not a market-impact model.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .nasdaq_itch_study import RESULTS_DIR, SYMBOLS


@dataclass
class OrderBook:
    bids: dict[int, OrderedDict] = field(default_factory=dict)
    asks: dict[int, OrderedDict] = field(default_factory=dict)
    orders: dict[int, tuple[str, int, int]] = field(default_factory=dict)

    def _side(self, side: str) -> dict[int, OrderedDict]:
        return self.bids if side == "B" else self.asks

    def _remove(self, order_id: int, quantity: int | None = None) -> int:
        item = self.orders.get(order_id)
        if item is None:
            return 0
        side, price, remaining = item
        removed = remaining if quantity is None else min(remaining, quantity)
        level = self._side(side)[price]
        left = remaining - removed
        if left:
            self.orders[order_id] = (side, price, left)
            level[order_id] = left
        else:
            self.orders.pop(order_id)
            level.pop(order_id, None)
            if not level:
                self._side(side).pop(price, None)
        return removed

    def apply(self, row) -> None:
        kind = row.kind
        order_id = int(row.order_id)
        if kind == "add":
            side, price, qty = str(row.side), int(row.price), int(row.quantity)
            if order_id in self.orders or qty <= 0:
                raise ValueError(f"Invalid add for order {order_id}")
            self.orders[order_id] = (side, price, qty)
            self._side(side).setdefault(price, OrderedDict())[order_id] = qty
        elif kind in {"execute", "cancel", "delete"}:
            if order_id not in self.orders:
                raise ValueError(f"Unknown order {order_id} in {kind}")
            remaining = self.orders[order_id][2]
            qty = remaining if kind == "delete" else int(row.quantity)
            if qty <= 0 or qty > remaining:
                raise ValueError(f"Invalid event size for {order_id}: {qty}/{remaining}")
            self._remove(order_id, qty)
        elif kind == "replace":
            if order_id not in self.orders:
                raise ValueError(f"Unknown replaced order {order_id}")
            self._remove(order_id)
            new_id, side = int(row.new_order_id), str(row.side)
            price, qty = int(row.price), int(row.quantity)
            if new_id in self.orders or qty <= 0:
                raise ValueError(f"Invalid replacement id {new_id}")
            self.orders[new_id] = (side, price, qty)
            self._side(side).setdefault(price, OrderedDict())[new_id] = qty
        else:
            raise ValueError(f"Unsupported order event: {kind}")

    def levels(self, side: str, count: int = 5) -> list[tuple[int, int]]:
        book = self._side(side)
        prices = sorted(book, reverse=(side == "B"))[:count]
        return [(p, sum(book[p].values())) for p in prices]

    def best(self, side: str) -> tuple[int, int] | None:
        levels = self.levels(side, 1)
        return levels[0] if levels else None


def _sweep(book: OrderBook, side: str, requested: float) -> tuple[float, float, float]:
    levels = book.levels(side, 5)
    remaining, notional, filled = requested, 0.0, 0.0
    for price, size in levels:
        take = min(remaining, size)
        notional += take * price
        remaining -= take
        filled += take
        if remaining <= 0:
            break
    bid, ask = book.best("B"), book.best("S")
    mid = (bid[0] + ask[0]) / 2.0 if bid and ask else np.nan
    if remaining > 0 and np.isfinite(mid):
        notional += remaining * mid
    effective = notional / requested if requested > 0 and np.isfinite(mid) else np.nan
    return effective, filled, remaining


def simulate_passive_signals(events: pd.DataFrame, signals: pd.DataFrame, *,
                             delay_ms: float, order_shares: float = 10.0,
                             horizon_ms: float = 1_000.0,
                             fee_bps_per_side: float = 5.0) -> dict[str, float]:
    """Replay a fixed set of predictions against exact public order IDs."""
    events = events.sort_values(["timestamp_ns", "sequence"], kind="stable")
    signals = signals.sort_values("timestamp_ns", kind="stable")
    event_rows = list(events.itertuples(index=False))
    signal_rows = list(signals.itertuples(index=False))
    book = OrderBook()
    event_idx = 0
    next_available_ns = -1
    considered = filled_orders = filled_shares_total = 0.0
    gross_results: list[float] = []
    net_results: list[float] = []
    initial_queues: list[float] = []
    exit_fill_ratios: list[float] = []
    crossed_at_arrival = 0
    priority_anomalies = 0
    delay_ns = int(delay_ms * 1_000_000)
    horizon_ns = int(horizon_ms * 1_000_000)

    def advance(until_ns: int) -> None:
        nonlocal event_idx
        while event_idx < len(event_rows) and int(event_rows[event_idx].timestamp_ns) <= until_ns:
            book.apply(event_rows[event_idx])
            event_idx += 1

    for signal in signal_rows:
        signal_ns = int(signal.timestamp_ns)
        advance(signal_ns)
        if signal_ns < next_available_ns:
            continue
        direction = int(signal.prediction)
        if direction not in (-1, 1):
            continue
        price = float(signal.bid_price if direction > 0 else signal.ask_price)
        if not np.isfinite(price) or price <= 0:
            continue
        considered += 1
        arrival_ns = signal_ns + delay_ns
        advance(arrival_ns)
        bid, ask = book.best("B"), book.best("S")
        if not bid or not ask:
            continue
        # Keep the decision-time price fixed. Skip if latency made it marketable.
        if (direction > 0 and price >= ask[0]) or (direction < 0 and price <= bid[0]):
            crossed_at_arrival += 1
            continue
        entry_side = "B" if direction > 0 else "S"
        level = book._side(entry_side).get(int(price), OrderedDict())
        ahead = dict(level)
        ahead_shares = float(sum(ahead.values()))
        initial_queues.append(ahead_shares)
        filled_shares = 0.0
        deadline_ns = signal_ns + horizon_ns
        while event_idx < len(event_rows) and int(event_rows[event_idx].timestamp_ns) <= deadline_ns:
            event = event_rows[event_idx]
            old_id = int(event.order_id)
            if event.kind in {"execute", "cancel", "delete", "replace"} and old_id in ahead:
                before = ahead[old_id]
                removed = before if event.kind in {"delete", "replace"} else min(before, int(event.quantity))
                ahead[old_id] = before - removed
                ahead_shares -= removed
                if ahead[old_id] <= 0:
                    del ahead[old_id]
            elif event.kind == "execute" and str(event.side) == entry_side and int(event.price) == int(price):
                if ahead_shares <= 0:
                    filled_shares += min(order_shares - filled_shares, int(event.quantity))
                else:
                    priority_anomalies += 1
            book.apply(event)
            event_idx += 1
        next_available_ns = deadline_ns
        if filled_shares <= 0:
            continue
        filled_orders += 1
        filled_shares_total += filled_shares
        exit_side = "B" if direction > 0 else "S"
        exit_price, exit_filled, residual = _sweep(book, exit_side, filled_shares)
        if not np.isfinite(exit_price):
            continue
        exit_fill_ratios.append(float(exit_filled / filled_shares))
        gross = direction * (exit_price - price) / price * 10_000.0
        # Residual shares marked to mid are already included in the effective exit.
        gross_results.append(float(gross))
        net_results.append(float(gross - 2.0 * fee_bps_per_side))

    return {
        "n_signals": int(considered), "n_orders_with_entry_fill": int(filled_orders),
        "entry_fill_probability": float(filled_orders / considered) if considered else 0.0,
        "unconditional_entry_fill_ratio": float(filled_shares_total / (considered * order_shares)) if considered else 0.0,
        "mean_initial_queue_ahead_shares": float(np.mean(initial_queues)) if initial_queues else np.nan,
        "mean_exit_displayed_fill_ratio": float(np.mean(exit_fill_ratios)) if exit_fill_ratios else np.nan,
        "mean_gross_bps_per_filled_trade": float(np.mean(gross_results)) if gross_results else np.nan,
        "mean_net_bps_per_filled_trade": float(np.mean(net_results)) if net_results else np.nan,
        "crossed_at_arrival": crossed_at_arrival,
        "priority_anomaly_events": priority_anomalies,
    }


def run_queue_sensitivity(*, split: str = "test",
                           predictions_path: Path | None = None,
                           event_dir: Path | None = None,
                           output_dir: Path = RESULTS_DIR,
                           models: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Run selected-model and queue-logistic passive fills on one frozen split."""
    from .nasdaq_itch_study import SESSIONS, implementation_fingerprint, sessions_for

    if split not in {"validation", "test"}:
        raise ValueError("Queue sensitivity is defined only for validation or final test dates")

    predictions_path = predictions_path or (output_dir / f"{split}_predictions.parquet")
    event_dir = event_dir or (Path(__file__).resolve().parents[1] / "data" / "nasdaq_bx_itch" / "snapshots" / "events")
    pred = pd.read_parquet(predictions_path)
    lock = json.loads((output_dir / "protocol_lock.json").read_text(encoding="utf-8"))
    if lock.get("implementation_sha256") != implementation_fingerprint():
        raise ValueError("Study code changed after the protocol lock; do not use this holdout")
    selected = str(lock["selected_model"])
    models = models or tuple(dict.fromkeys((selected, "queue_imbalance_logistic")))
    pred["prediction"] = np.array([-1, 0, 1])[pred[["p_down", "p_flat", "p_up"]].to_numpy().argmax(axis=1)]
    result = []
    selected_sessions = sessions_for(split)
    for day in sorted(s.session_date for s in selected_sessions):
        for symbol in SYMBOLS:
            event_path = event_dir / f"{day}_{symbol}.parquet"
            events = pd.read_parquet(event_path)
            for model_name in models:
                signals = pred[(pred.session_date == day) & (pred.ticker == symbol) & (pred.model == model_name)]
                for delay in (1.0, 10.0, 50.0):
                    for fee in (0.0, 1.0, 5.0):
                        metrics = simulate_passive_signals(events, signals, delay_ms=delay,
                                                           fee_bps_per_side=fee)
                        result.append({"session_date": day, "ticker": symbol,
                                       "model": model_name, "delay_ms": delay,
                                       "requested_order_shares": 10.0,
                                       "holding_horizon_ms": 1_000.0,
                                       "fee_bps_per_side": fee, **metrics})
    output = pd.DataFrame(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "passive_queue_sensitivity.csv" if split == "test" else "validation_passive_queue_sensitivity.csv"
    output.to_csv(output_dir / filename, index=False, float_format="%.8f")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    output = run_queue_sensitivity(split=args.split)
    print(output.to_string(index=False))
