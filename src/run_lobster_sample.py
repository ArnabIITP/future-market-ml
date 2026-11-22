"""Explore chronological prediction on LOBSTER's public one-day samples.

This is a data-pipeline and method demonstration, not evidence of persistent
alpha. It requires a C++17 compiler and scikit-learn, xgboost, pandas, numpy,
and matplotlib from the project's optional dependencies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .download_lobster_samples import SAMPLE_SHA256, download_samples


LABELS = np.array([-1, 0, 1], dtype=np.int64)
HORIZON_EVENTS = 100
FLAT_THRESHOLD_PRICE_UNITS = 50.0  # half a cent; sample prices are dollars * 10,000
FEATURES = ["spread_ticks", "queue_imbalance_l1", "queue_imbalance_5",
            "ofi_l1", "signed_event_size", "flow_10", "flow_100"]
FEATURE_COLUMNS = ["time_seconds", "event_type", "order_size", "direction",
                   "signed_event_size", "mid_ticks", "spread_ticks",
                   "best_bid_ticks", "best_ask_ticks", "bid_size_1", "ask_size_1",
                   "queue_imbalance_l1", "queue_imbalance_5", "ofi_l1",
                   "event_order_id", "event_price_ticks"] + [
                       f"{side}_{kind}_level{level}" for level in range(1, 6)
                       for side, kind in (("ask", "price"), ("ask", "size"),
                                          ("bid", "price"), ("bid", "size"))]


def compile_extractor(project_root: Path, output_dir: Path) -> Path:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise RuntimeError("A C++17 compiler (g++ or clang++) is required.")
    executable = output_dir / ("lobster_features.exe" if __import__("os").name == "nt" else "lobster_features")
    subprocess.run([compiler, "-std=c++17", "-O3", "-DNDEBUG",
                    str(project_root / "src" / "lobster_features.cpp"), "-o", str(executable)],
                   check=True)
    return executable


def compile_features(executable: Path, sample_dir: Path, ticker: str,
                     output_dir: Path) -> tuple[Path, dict[str, float]]:
    messages = next(sample_dir.glob(f"{ticker}_*_message_5.csv"))
    orderbook = next(sample_dir.glob(f"{ticker}_*_orderbook_5.csv"))
    output = output_dir / f"{ticker}_features.bin"
    started = time.perf_counter()
    completed = subprocess.run([str(executable), str(messages), str(orderbook), str(output)],
                               capture_output=True, text=True, check=True)
    elapsed = time.perf_counter() - started
    print(f"{ticker}: {completed.stderr.strip()}")
    match = re.search(r"processed (\d+) aligned events in ([0-9.]+) seconds \(([0-9.]+) events/s\)",
                      completed.stderr)
    if not match:
        raise RuntimeError(f"C++ extractor omitted its event count or timing for {ticker}")
    return output, {"events": float(match.group(1)), "parser_seconds": float(match.group(2)),
                    "parser_events_per_second": float(match.group(3)),
                    "end_to_end_seconds": elapsed}


def make_targets(features: pd.DataFrame, horizon: int = HORIZON_EVENTS,
                 flat_threshold: float = FLAT_THRESHOLD_PRICE_UNITS) -> pd.DataFrame:
    """Label the future mid-price move, retaining only complete horizons."""
    result = features.copy()
    future_mid = result["mid_ticks"].shift(-horizon)
    move = future_mid - result["mid_ticks"]
    result["target"] = np.select([move > flat_threshold, move < -flat_threshold], [1, -1], default=0)
    result.loc[future_mid.isna(), "target"] = np.nan
    result["flow_10"] = result["signed_event_size"].rolling(10, min_periods=1).sum()
    result["flow_100"] = result["signed_event_size"].rolling(100, min_periods=1).sum()
    return result.reset_index(drop=True)


def class_order_probabilities(model, x: pd.DataFrame, encoded: bool = False) -> np.ndarray:
    raw = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int) - (1 if encoded else 0)
    out = np.zeros((len(x), len(LABELS)), dtype=float)
    for column, label in enumerate(classes):
        out[:, np.where(LABELS == label)[0][0]] = raw[:, column]
    totals = out.sum(axis=1, keepdims=True)
    return np.divide(out, totals, out=np.full_like(out, 1 / len(LABELS)), where=totals > 0)


def score(y_true: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=LABELS, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probability, labels=LABELS)),
    }


def sweep_vwap(row: pd.Series, *, buy: bool, quantity: float) -> tuple[float, float]:
    """Return shares filled and VWAP by sweeping the displayed five levels."""
    remaining = float(quantity)
    notional = 0.0
    for level in range(1, 6):
        price = float(row[f"{'ask' if buy else 'bid'}_price_level{level}"])
        size = max(0.0, float(row[f"{'ask' if buy else 'bid'}_size_level{level}"]))
        if size <= 0 or price <= 0:
            continue
        filled = min(remaining, size)
        notional += filled * price
        remaining -= filled
        if remaining <= 1e-12:
            break
    filled = quantity - remaining
    return float(filled), float(notional / filled) if filled > 0 else float("nan")


def _sweep_arrays(index: int, *, buy: bool, quantity: float,
                  arrays: dict[str, np.ndarray]) -> tuple[float, float]:
    """Array-backed book sweep used in the repeated execution loop."""
    side = "ask" if buy else "bid"
    remaining = float(quantity)
    notional = 0.0
    for level in range(1, 6):
        price = arrays[f"{side}_price_level{level}"][index]
        size = max(0.0, arrays[f"{side}_size_level{level}"][index])
        if size <= 0 or price <= 0:
            continue
        filled = min(remaining, size)
        notional += filled * price
        remaining -= filled
        if remaining <= 1e-12:
            break
    filled = quantity - remaining
    return float(filled), float(notional / filled) if filled > 0 else float("nan")


def build_execution_volume_index(frame: pd.DataFrame) -> dict[tuple[float, int], tuple[np.ndarray, np.ndarray]]:
    """Index visible executions by price and resting order side for queue proxy."""
    event_type = frame["event_type"].to_numpy(dtype=np.int64, copy=False)
    event_price = frame["event_price_ticks"].to_numpy(dtype=np.float64, copy=False)
    direction = frame["direction"].to_numpy(dtype=np.int64, copy=False)
    size = frame["order_size"].to_numpy(dtype=np.float64, copy=False)
    indices = np.flatnonzero((event_type == 4) & (event_price > 0) & (size > 0))
    if len(indices) == 0:
        return {}
    order = np.lexsort((indices, direction[indices], event_price[indices]))
    ordered_indices = indices[order]
    prices = event_price[ordered_indices]
    sides = direction[ordered_indices]
    sizes = size[ordered_indices]
    boundaries = np.r_[0, np.flatnonzero((prices[1:] != prices[:-1]) |
                                          (sides[1:] != sides[:-1])) + 1, len(order)]
    result: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        idx = ordered_indices[start:stop]
        result[(float(prices[start]), int(sides[start]))] = (idx, np.cumsum(sizes[start:stop]))
    return result


def _executed_at_price(execution_index: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]],
                       price: float, side: int, start: int, stop: int) -> float:
    values = execution_index.get((float(price), int(side)))
    if values is None:
        return 0.0
    indices, cumulative = values
    left = int(np.searchsorted(indices, start, side="right"))
    right = int(np.searchsorted(indices, stop, side="right"))
    if right <= left:
        return 0.0
    return float(cumulative[right - 1] - (cumulative[left - 1] if left else 0.0))


def simulate_passive_queue(frame: pd.DataFrame, test_start: int,
                           probability: np.ndarray, ticker: str, model_name: str,
                           execution_index: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]],
                           *, latency_events: tuple[int, ...] = (1,),
                           order_sizes: tuple[float, ...] = (1.0,),
                           queue_multipliers: tuple[float, ...] = (1.0,)) -> list[dict]:
    """Stress passive touch fills under explicit aggregate queue-ahead assumptions.

    A predicted up signal joins the displayed bid; down joins the displayed ask.
    The queue ahead is the displayed top-level depth times a stress multiplier.
    Only subsequent visible executions reduce it; cancellations are ignored.
    This is a conservative aggregate queue proxy, not order-ID/FIFO replay.
    """
    rows = []
    needed = ["best_bid_ticks", "best_ask_ticks", "bid_size_1", "ask_size_1"] + [
        f"{side}_{kind}_level{level}" for level in range(1, 6)
        for side, kind in (("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size"))]
    arrays = {name: frame[name].to_numpy(dtype=np.float64, copy=False) for name in needed}
    predicted_side = LABELS[np.asarray(probability).argmax(axis=1)]
    for latency in latency_events:
        signal_indices = np.flatnonzero(predicted_side != 0) + test_start
        selected: list[tuple[int, int, int]] = []
        cursor = test_start
        for signal_idx in signal_indices:
            if signal_idx < cursor:
                continue
            entry_idx = int(signal_idx + latency)
            exit_idx = int(signal_idx + HORIZON_EVENTS + latency)
            if exit_idx >= len(frame):
                continue
            selected.append((int(predicted_side[signal_idx - test_start]), entry_idx, exit_idx))
            cursor = exit_idx + 1

        for queue_multiplier in queue_multipliers:
            for requested_qty in order_sizes:
                fills: list[tuple[int, int, float, float, float, float]] = []
                filled_orders = partial_entries = partial_exits = 0
                total_entry_shares = total_exit_shares = 0.0
                for side, entry_idx, exit_idx in selected:
                    buy_entry = side > 0
                    limit_price = arrays["best_bid_ticks"][entry_idx] if buy_entry else arrays["best_ask_ticks"][entry_idx]
                    displayed_queue = arrays["bid_size_1"][entry_idx] if buy_entry else arrays["ask_size_1"][entry_idx]
                    ahead = max(0.0, displayed_queue) * queue_multiplier
                    traded_volume = _executed_at_price(execution_index, limit_price, side,
                                                       entry_idx, exit_idx)
                    entry_filled = min(requested_qty, max(0.0, traded_volume - ahead))
                    if entry_filled <= 0:
                        continue
                    filled_orders += 1
                    total_entry_shares += entry_filled
                    if entry_filled < requested_qty:
                        partial_entries += 1
                    exit_filled, exit_vwap = _sweep_arrays(exit_idx, buy=side < 0,
                                                           quantity=entry_filled, arrays=arrays)
                    if exit_filled <= 0:
                        continue
                    total_exit_shares += exit_filled
                    if exit_filled < entry_filled:
                        partial_exits += 1
                    exit_mid = (arrays["best_bid_ticks"][exit_idx] + arrays["best_ask_ticks"][exit_idx]) / 2
                    remaining = entry_filled - exit_filled
                    gross_cash = side * (exit_vwap - limit_price) * exit_filled
                    gross_cash += side * (exit_mid - limit_price) * remaining
                    entry_mid = (arrays["best_bid_ticks"][entry_idx] + arrays["best_ask_ticks"][entry_idx]) / 2
                    gross_bps = float(gross_cash / (entry_mid * entry_filled) * 10000)
                    exit_ratio = exit_filled / entry_filled
                    fills.append((side, entry_idx, entry_filled, exit_filled,
                                  gross_bps, exit_ratio))

                gross = np.asarray([row[4] for row in fills], dtype=float)
                exit_ratios = np.asarray([row[5] for row in fills], dtype=float)
                for fee_bps in (0.0, 5.0, 10.0):
                    net = gross - fee_bps * (1.0 + exit_ratios)
                    rows.append({
                        "ticker": ticker, "model": model_name,
                        "entry_delay_events": latency,
                        "holding_horizon_events": HORIZON_EVENTS,
                        "requested_order_shares": requested_qty,
                        "queue_ahead_multiplier": queue_multiplier,
                        "fee_bps_per_side": fee_bps,
                        "n_signal_orders": len(selected), "n_orders_with_entry_fill": filled_orders,
                        "total_entry_filled_shares": total_entry_shares,
                        "entry_fill_probability": filled_orders / len(selected) if selected else np.nan,
                        "unconditional_entry_fill_ratio": total_entry_shares /
                            (len(selected) * requested_qty) if selected else np.nan,
                        "mean_entry_fill_ratio_conditional":
                            float(total_entry_shares / (filled_orders * requested_qty)) if filled_orders else np.nan,
                        "partial_entry_orders": partial_entries,
                        "partial_exit_orders": partial_exits,
                        "n_closed_or_marked_trades": int(len(fills)),
                        "gross_mean_bps_per_trade": float(gross.mean()) if len(gross) else np.nan,
                        "net_mean_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                        "net_sum_bps": float(net.sum()) if len(net) else 0.0,
                        "mean_exit_fill_ratio": float(exit_ratios.mean()) if len(exit_ratios) else np.nan,
                    })
    return rows


def simulate_taker_touch(frame: pd.DataFrame, test_start: int,
                         probability: np.ndarray, ticker: str, model_name: str,
                         *, latency_events: tuple[int, ...] = (1,),
                         order_sizes: tuple[float, ...] = (1.0,)) -> list[dict]:
    """Stress a fixed taker rule over event-latency and displayed-depth scenarios.

    Entry and exit sweep the available five-level book. Partial entry and exit
    fills are tracked; any residual at the scheduled exit is marked to the
    midprice, clearly separating marked inventory from fully executed shares.
    This remains a counterfactual and does not model queue priority or impact
    beyond the supplied five levels.
    """
    rows = []
    needed = ["best_bid_ticks", "best_ask_ticks"] + [
        f"{side}_{kind}_level{level}" for level in range(1, 6)
        for side, kind in (("ask", "price"), ("ask", "size"), ("bid", "price"), ("bid", "size"))]
    arrays = {name: frame[name].to_numpy(dtype=np.float64, copy=False) for name in needed}
    predicted_side = LABELS[np.asarray(probability).argmax(axis=1)]
    for latency in latency_events:
        signal_indices = np.flatnonzero(predicted_side != 0) + test_start
        selected: list[tuple[int, int, int, int]] = []
        cursor = test_start
        for signal_idx in signal_indices:
            if signal_idx < cursor:
                continue
            entry_idx = int(signal_idx + latency)
            exit_idx = int(signal_idx + HORIZON_EVENTS + latency)
            if exit_idx >= len(frame):
                continue
            side = int(predicted_side[signal_idx - test_start])
            selected.append((side, entry_idx, exit_idx, int(signal_idx)))
            cursor = exit_idx + 1  # avoid overlapping positions
        for requested_qty in order_sizes:
            gross_bps_rows: list[float] = []
            entry_fill_ratios: list[float] = []
            exit_fill_ratios: list[float] = []
            partial_entries = partial_exits = 0
            for side, entry_idx, exit_idx, _ in selected:
                entry_filled, entry_vwap = _sweep_arrays(entry_idx, buy=side > 0,
                                                         quantity=requested_qty, arrays=arrays)
                if entry_filled <= 0:
                    continue
                exit_filled, exit_vwap = _sweep_arrays(exit_idx, buy=side < 0,
                                                       quantity=entry_filled, arrays=arrays)
                if exit_filled <= 0:
                    continue
                if entry_filled < requested_qty:
                    partial_entries += 1
                if exit_filled < entry_filled:
                    partial_exits += 1
                entry_fill_ratios.append(entry_filled / requested_qty)
                exit_fill_ratios.append(exit_filled / entry_filled)
                entry_mid = (arrays["best_bid_ticks"][entry_idx] + arrays["best_ask_ticks"][entry_idx]) / 2
                exit_mid = (arrays["best_bid_ticks"][exit_idx] + arrays["best_ask_ticks"][exit_idx]) / 2
                remaining = entry_filled - exit_filled
                gross_cash = side * (exit_vwap - entry_vwap) * exit_filled
                gross_cash += side * (exit_mid - entry_vwap) * remaining
                gross_bps_rows.append(float(gross_cash / (entry_mid * entry_filled) * 10000))

            gross = np.asarray(gross_bps_rows, dtype=float)
            for fee_bps in (0.0, 5.0, 10.0):
                # Fees are charged on entry and on the shares actually exited.
                net = gross - fee_bps * (1.0 + np.asarray(exit_fill_ratios, dtype=float))
                rows.append({
                    "ticker": ticker, "model": model_name,
                    "fee_bps_per_side": fee_bps, "entry_delay_events": latency,
                    "holding_horizon_events": HORIZON_EVENTS,
                    "requested_order_shares": requested_qty,
                    "n_trades": int(len(net)), "partial_entry_orders": partial_entries,
                    "partial_exit_orders": partial_exits,
                    "mean_entry_fill_ratio": float(np.mean(entry_fill_ratios)) if entry_fill_ratios else np.nan,
                    "mean_exit_fill_ratio": float(np.mean(exit_fill_ratios)) if exit_fill_ratios else np.nan,
                    "gross_mean_bps_per_trade": float(gross.mean()) if len(gross) else np.nan,
                    "net_mean_bps_per_trade": float(net.mean()) if len(net) else np.nan,
                    "net_sum_bps": float(net.sum()) if len(net) else 0.0,
                })
    return rows


def evaluate_symbol(frame: pd.DataFrame, ticker: str, train_fraction: float = 0.70
                    ) -> tuple[list[dict], list[dict], list[dict]]:
    from xgboost import XGBClassifier

    split = int(len(frame) * train_fraction)
    train_stop = split - HORIZON_EVENTS  # purge labels whose forward window touches test
    test_start = split
    test_stop = len(frame) - HORIZON_EVENTS
    if train_stop < 1 or test_stop - test_start < 100:
        raise ValueError(f"Insufficient observations for a purged chronological split: {ticker}")
    train = frame.iloc[:train_stop].dropna(subset=["target"])
    test = frame.iloc[test_start:test_stop].dropna(subset=["target"])
    x_train, y_train = train[FEATURES], train["target"].astype(int).to_numpy()
    x_test, y_test = test[FEATURES], test["target"].astype(int).to_numpy()
    observed_classes = np.unique(y_train)
    if len(observed_classes) != len(LABELS):
        raise ValueError(f"Training window lacks a target class for {ticker}: {observed_classes}")

    counts = np.array([(y_train == label).sum() for label in LABELS], dtype=float)
    prior = counts / counts.sum()
    majority = np.full(len(y_test), LABELS[np.argmax(counts)])
    rows = [{"ticker": ticker, "model": "training_class_prior", "n_train": len(train),
             "n_test": len(test), "purged_train_events": HORIZON_EVENTS,
             **score(y_test, majority, np.tile(prior, (len(y_test), 1)))}]
    strategy_rows = simulate_taker_touch(frame, test_start, np.tile(prior, (len(test), 1)),
                                         ticker, "training_class_prior",
                                         latency_events=(1, 10, 50), order_sizes=(1.0, 10.0, 100.0))
    execution_index = build_execution_volume_index(frame)
    queue_rows = simulate_passive_queue(
        frame, test_start, np.tile(prior, (len(test), 1)), ticker, "training_class_prior",
        execution_index, latency_events=(1, 10, 50), order_sizes=(1.0, 10.0, 100.0),
        queue_multipliers=(1.0, 2.0, 4.0))

    models = {
        "queue_imbalance_logistic": (
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300,
                                                                class_weight="balanced", random_state=17)),
            ["queue_imbalance_l1", "queue_imbalance_5"]),
        "engineered_logistic": (
            make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300,
                                                                class_weight="balanced", random_state=17)),
            FEATURES),
        "random_forest": (
            RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=50,
                                   max_features="sqrt", class_weight="balanced_subsample",
                                   n_jobs=4, random_state=17), FEATURES),
        "xgboost": (
            XGBClassifier(objective="multi:softprob", num_class=3, n_estimators=120,
                          max_depth=4, learning_rate=0.05, min_child_weight=30,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                          tree_method="hist", n_jobs=4, random_state=17,
                          eval_metric="mlogloss"), FEATURES),
    }
    for name, (model, columns) in models.items():
        fit_labels = y_train + 1 if name == "xgboost" else y_train
        model.fit(x_train[columns], fit_labels)
        probability = class_order_probabilities(model, x_test[columns], encoded=name == "xgboost")
        prediction = LABELS[probability.argmax(axis=1)]
        rows.append({"ticker": ticker, "model": name, "n_train": len(train),
                     "n_test": len(test), "purged_train_events": HORIZON_EVENTS,
                     **score(y_test, prediction, probability)})
        strategy_rows.extend(simulate_taker_touch(frame, test_start, probability, ticker, name,
                                                  latency_events=(1, 10, 50),
                                                  order_sizes=(1.0, 10.0, 100.0)))
        queue_rows.extend(simulate_passive_queue(
            frame, test_start, probability, ticker, name, execution_index,
            latency_events=(1, 10, 50), order_sizes=(1.0, 10.0, 100.0),
            queue_multipliers=(1.0, 2.0, 4.0)))
    return rows, strategy_rows, queue_rows


def render_plot(metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    pivot = metrics.pivot(index="ticker", columns="model", values="macro_f1")
    pretty = {"engineered_logistic": "Engineered logistic",
              "queue_imbalance_logistic": "Queue imbalance logistic",
              "random_forest": "Random forest",
              "training_class_prior": "Training class prior",
              "xgboost": "XGBoost"}
    pivot = pivot.rename(columns=pretty)
    ax = pivot.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=0,
                    title="Chronological test macro-F1 — LOBSTER sample day")
    ax.set_ylabel("Macro-F1 (higher is better)")
    ax.set_xlabel("Ticker (2012-06-21)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", frameon=False, fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def render_execution_plot(strategy: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    strategy = strategy.loc[(strategy.entry_delay_events == 1) &
                            (strategy.requested_order_shares == 1.0)]
    grouped = strategy.groupby(["model", "fee_bps_per_side"], as_index=False).agg(
        n_trades=("n_trades", "sum"), net_sum=("net_sum_bps", "sum"))
    grouped["mean_net_bps"] = grouped["net_sum"] / grouped["n_trades"].replace(0, np.nan)
    pivot = grouped.pivot(index="model", columns="fee_bps_per_side", values="mean_net_bps")
    pivot = pivot.rename(index={"engineered_logistic": "Engineered logistic",
                                "queue_imbalance_logistic": "Queue imbalance logistic",
                                "random_forest": "Random forest",
                                "training_class_prior": "Training class prior",
                                "xgboost": "XGBoost"})
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=0,
                    title="One-share taker-touch counterfactual — mean net bps/trade")
    ax.axhline(0, color="#263238", linewidth=1)
    ax.set_ylabel("Mean net basis points per trade")
    ax.set_xlabel("Model")
    ax.legend(title="Fee per side (bps)", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def render_execution_sensitivity_plot(strategy: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = strategy.loc[(strategy.fee_bps_per_side == 5.0) &
                          strategy.model.isin(["training_class_prior", "random_forest", "xgboost"])]
    grouped = subset.groupby(["model", "entry_delay_events", "requested_order_shares"], as_index=False).agg(
        n_trades=("n_trades", "sum"), net_sum=("net_sum_bps", "sum"))
    grouped["mean_net_bps"] = grouped["net_sum"] / grouped["n_trades"].replace(0, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    labels = {"training_class_prior": "Class prior", "random_forest": "Random forest", "xgboost": "XGBoost"}
    for ax, quantity in zip(axes, (1.0, 10.0, 100.0)):
        part = grouped.loc[grouped.requested_order_shares == quantity]
        for model, model_rows in part.groupby("model"):
            ax.plot(model_rows.entry_delay_events, model_rows.mean_net_bps, marker="o",
                    label=labels.get(model, model))
        ax.axhline(0, color="#263238", linewidth=1)
        ax.set_title(f"Requested size: {quantity:g} shares")
        ax.set_xlabel("Entry delay (book events)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean net bps/trade at 5 bps/side")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Taker execution sensitivity: delay and size")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_queue_sensitivity_plot(queue: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = queue.loc[(queue.fee_bps_per_side == 5.0) &
                       (queue.entry_delay_events == 1) &
                       (queue.requested_order_shares == 10.0) &
                       queue.model.isin(["training_class_prior", "random_forest", "xgboost"])]
    grouped = subset.groupby(["model", "queue_ahead_multiplier"], as_index=False).agg(
        signal_orders=("n_signal_orders", "sum"), filled_orders=("n_orders_with_entry_fill", "sum"))
    grouped["entry_fill_probability"] = grouped.filled_orders / grouped.signal_orders.replace(0, np.nan)
    labels = {"training_class_prior": "Class prior", "random_forest": "Random forest", "xgboost": "XGBoost"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, rows in grouped.groupby("model"):
        ax.plot(rows.queue_ahead_multiplier, rows.entry_fill_probability, marker="o",
                label=labels.get(model, model))
    ax.set(title="Passive fill probability under queue-ahead stress",
           xlabel="Queue ahead / displayed top-level depth",
           ylabel="Orders with any entry fill / signals", ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_report(metrics: pd.DataFrame, strategy: pd.DataFrame, queue: pd.DataFrame,
                  metadata: dict, path: Path) -> None:
    lines = [
        "# LOBSTER raw-event sample study",
        "",
        "## Scope",
        "",
        "This engineering and modeling demonstration uses LOBSTER's public five-level sample files for five US stocks on 2012-06-21. It is one session, so it cannot establish persistence across days or regimes.",
        "",
        f"The C++17 extractor processed {sum(metadata['events_per_symbol'].values()):,} aligned message/book events in total. It derives top-of-book spread, queue imbalance, order-flow imbalance, and signed event flow. Raw sample archives are kept outside Git; their SHA-256 hashes are in `run_metadata.json`.",
        "",
        "## Evaluation",
        "",
        f"The target is the mid-price direction {HORIZON_EVENTS} book events ahead; moves within half a cent are labeled stationary. Each ticker uses a chronological 70/30 split. The last {HORIZON_EVENTS} training events are purged so training labels cannot extend into the test period.",
        "",
        "| Ticker | Model | Train events | Test events | Macro-F1 | Balanced accuracy | Log loss |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.sort_values(["ticker", "model"]).itertuples(index=False):
        lines.append(f"| {row.ticker} | {row.model} | {row.n_train:,} | {row.n_test:,} | {row.macro_f1:.3f} | {row.balanced_accuracy:.3f} | {row.log_loss:.3f} |")
    lines.extend(["", "![Walk-forward classification scores by ticker](chronological_macro_f1.png)",
                  "", "## Taker-touch execution sensitivity", "",
                  "Predicted directional classes are entered after 1, 10, or 50 events and exited 100 events later. Orders request 1, 10, or 100 shares and sweep the displayed five-level book; partial entry/exit fills and residual inventory marked to mid are tracked. Fees are scenario assumptions per side.",
                  "",
                  "This is a counterfactual benchmark, not a full simulator: delays are event counts rather than elapsed time; market impact beyond five displayed levels and participant-specific fees are not modeled. Residual inventory is marked to mid rather than assumed filled. It must not be read as a profitable strategy result.",
                  "",
                  "| Model | Delay (events) | Requested shares | Fee per side (bps) | Trades | Entry fill ratio | Exit fill ratio | Mean net bps/trade |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    grouped = strategy.groupby(["model", "entry_delay_events", "requested_order_shares", "fee_bps_per_side"], as_index=False).agg(
        n_trades=("n_trades", "sum"), net_sum=("net_sum_bps", "sum"),
        entry_fill=("mean_entry_fill_ratio", "mean"), exit_fill=("mean_exit_fill_ratio", "mean"))
    for row in grouped.sort_values(["model", "fee_bps_per_side"]).itertuples(index=False):
        mean = row.net_sum / row.n_trades if row.n_trades else float("nan")
        mean_text = f"{mean:.3f}" if np.isfinite(mean) else "n/a"
        entry_fill = f"{row.entry_fill:.3f}" if np.isfinite(row.entry_fill) else "n/a"
        exit_fill = f"{row.exit_fill:.3f}" if np.isfinite(row.exit_fill) else "n/a"
        lines.append(f"| {row.model} | {row.entry_delay_events} | {row.requested_order_shares:g} | {row.fee_bps_per_side:.1f} | {row.n_trades:,} | {entry_fill} | {exit_fill} | {mean_text} |")
    lines.extend(["", "![Mean net result per trade under fee scenarios](taker_touch_fee_sensitivity.png)",
                  "", "![Net result sensitivity to event delay and requested order size](execution_sensitivity.png)",
                  "", "## Passive queue-ahead proxy", "",
                  "The separate maker-side stress joins the displayed best bid for predicted up moves and best ask for down moves. Queue ahead starts at displayed top-level depth multiplied by 1x, 2x, or 4x. Only visible executions at that exact price and side consume the queue; cancellations are ignored. This is an aggregate queue-ahead proxy, not exact FIFO/order-ID replay. Orders are canceled after the 100-event horizon; filled shares are exited with a five-level sweep and residual shares are marked to mid.",
                  "",
                  "| Model | Delay (events) | Queue multiplier | Signals | Entry fill probability | Unconditional size fill | Mean net bps/trade at 5 bps/side |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    queue_summary = queue.loc[(queue.fee_bps_per_side == 5.0) &
                              (queue.requested_order_shares == 10.0) &
                              queue.model.isin(["training_class_prior", "xgboost"])]
    queue_summary = queue_summary.groupby(["model", "entry_delay_events", "queue_ahead_multiplier"], as_index=False).agg(
        signals=("n_signal_orders", "sum"), filled_orders=("n_orders_with_entry_fill", "sum"),
        entry_shares=("total_entry_filled_shares", "sum"),
        net_sum=("net_sum_bps", "sum"), trades=("n_closed_or_marked_trades", "sum"))
    for row in queue_summary.sort_values(["model", "entry_delay_events", "queue_ahead_multiplier"]).itertuples(index=False):
        fill_prob = row.filled_orders / row.signals if row.signals else float("nan")
        unconditional_fill = row.entry_shares / (row.signals * 10.0) if row.signals else float("nan")
        mean_net = row.net_sum / row.trades if row.trades else float("nan")
        lines.append(f"| {row.model} | {row.entry_delay_events} | {row.queue_ahead_multiplier:g}x | {row.signals:,} | {fill_prob:.3f} | {unconditional_fill:.3f} | {mean_net:.3f} |")
    lines.extend(["", "![Passive fill probability by queue-ahead stress](passive_queue_sensitivity.png)",
                  "", "## Reproduction", "",
                  "```powershell", "pip install -e \".[microstructure]\"",
                  "python -m src.run_lobster_sample", "```", "",
                  "The runner downloads the official sample ZIPs, verifies SHA-256, compiles the C++ feature extractor, and writes metrics and figures under `results/market_microstructure/lobster_sample/`.", "",
                  "## Source and data handling", "",
                  "- LOBSTER public sample files: [official sample page](https://php.lobsterdata.com/info/DataSamples.php).",
                  "- LOBSTER access and terms: [access options](https://php.lobsterdata.com/info/AccessOptions.php) and [terms](https://php.lobsterdata.com/termsAndConditions.php).",
                  "- The raw sample files are not committed or redistributed. Check the source terms before using the data outside this demonstration.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(project_root: Path, *, reuse_features: bool = False) -> pd.DataFrame:
    hashes = download_samples(project_root)
    output_dir = project_root / "data" / "market_microstructure" / "processed" / "lobster_sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    elapsed_by_symbol = {}
    event_counts = {}
    rows = []
    strategy_rows = []
    queue_rows = []
    with tempfile.TemporaryDirectory(prefix="future_lobster_") as temp:
        executable = None if reuse_features else compile_extractor(project_root, Path(temp))
        for ticker in SAMPLE_SHA256:
            feature_path = output_dir / f"{ticker}_features.bin"
            if not reuse_features:
                sample_dir = project_root / "data" / "market_microstructure" / "raw" / "lobster_samples" / ticker
                feature_path, timing = compile_features(executable, sample_dir, ticker, output_dir)
                elapsed_by_symbol[ticker] = timing
            elif not feature_path.exists():
                raise FileNotFoundError(f"No cached features at {feature_path}; run once without --reuse-features.")
            raw = np.fromfile(feature_path, dtype="<f8")
            if raw.size % len(FEATURE_COLUMNS):
                raise ValueError(f"Truncated binary feature file: {feature_path}")
            features = pd.DataFrame(raw.reshape(-1, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
            event_counts[ticker] = len(features)
            prepared = make_targets(features)
            ticker_metrics, ticker_strategy, ticker_queue = evaluate_symbol(prepared, ticker)
            rows.extend(ticker_metrics)
            strategy_rows.extend(ticker_strategy)
            queue_rows.extend(ticker_queue)

    results_dir = project_root / "results" / "market_microstructure" / "lobster_sample"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / "summary_metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame(strategy_rows).to_csv(results_dir / "strategy_sensitivity.csv", index=False,
                                       float_format="%.6f")
    queue_frame = pd.DataFrame(queue_rows)
    queue_frame.to_csv(results_dir / "passive_queue_sensitivity.csv", index=False,
                       float_format="%.6f")
    render_plot(metrics, results_dir / "chronological_macro_f1.png")
    strategy_frame = pd.DataFrame(strategy_rows)
    render_execution_plot(strategy_frame, results_dir / "taker_touch_fee_sensitivity.png")
    render_execution_sensitivity_plot(strategy_frame, results_dir / "execution_sensitivity.png")
    render_queue_sensitivity_plot(queue_frame, results_dir / "passive_queue_sensitivity.png")
    package_versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "xgboost"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not-installed"
    metadata = {
        "dataset": "Official LOBSTER five-level public sample files",
        "source_url": "https://php.lobsterdata.com/info/DataSamples.php",
        "date": "2012-06-21",
        "tickers": list(SAMPLE_SHA256),
        "archive_sha256": hashes,
        "events_per_symbol": event_counts,
        "target": "mid-price direction over the next 100 book events; flat when the move is within half a cent (50 LOBSTER price units)",
        "horizon_events": HORIZON_EVENTS,
        "chronological_train_fraction": 0.70,
        "purged_train_events": HORIZON_EVENTS,
        "feature_columns": FEATURE_COLUMNS,
        "models": {
            "training_class_prior": "training-window class frequencies",
            "queue_imbalance_logistic": "C=0.1, balanced classes, top-level and five-level queue imbalance",
            "engineered_logistic": "C=0.1, balanced classes, standardized seven-feature set",
            "random_forest": "150 trees, max_depth=12, min_samples_leaf=50, balanced_subsample",
            "xgboost": "120 trees, max_depth=4, learning_rate=0.05, fixed parameters",
        },
        "strategy_counterfactual": {
            "decision": "take the predicted up/down class; skip predicted stationary",
            "entry": "marketable sweep of up to five displayed levels after 1, 10, or 50 events",
            "exit": "opposite-side sweep 100 events after entry; residual shares marked to mid",
            "position_overlap": "prevented; next signal is considered after the prior exit",
            "requested_order_shares": [1.0, 10.0, 100.0],
            "fees_bps_per_side": [0.0, 5.0, 10.0],
            "latency_scenarios_events": [1, 10, 50],
            "limits": "event-count delay is not measured latency; impact beyond five levels is omitted; one-day counterfactual only",
        },
        "passive_queue_proxy": {
            "entry": "post at displayed bid for predicted up moves and displayed ask for predicted down moves",
            "queue_ahead": "displayed top-level depth times 1x, 2x, or 4x stress multiplier",
            "queue_depletion": "visible executions at exact price and side only; cancellations ignored",
            "exit": "five-level taker sweep after the 100-event horizon; residual marked to mid",
            "limitations": "aggregate conservative queue-ahead proxy, not exact FIFO or order-ID replay; no maker rebates",
        },
        "feature_extraction_runtime_seconds_by_symbol": elapsed_by_symbol,
        "compiler": "g++/clang++ C++17 -O3",
        "runtime_platform": platform.platform(),
        "feature_output": "little-endian float64 records including message identifiers, event prices, and five book levels; column order defined by FEATURE_COLUMNS in src/run_lobster_sample.py",
        "package_versions": package_versions,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["one sample trading day only", "not a distribution-shift or persistence claim",
                        "event-count delay is not measured latency", "passive queue proxy is not exact FIFO replay",
                        "counterfactual five-level sweep is not a full execution model"],
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    render_report(metrics, strategy_frame, queue_frame, metadata, results_dir / "report.md")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reuse-features", action="store_true",
                        help="skip C++ feature extraction and reuse local ignored binary files")
    args = parser.parse_args()
    print(run(args.project_root, reuse_features=args.reuse_features).to_string(index=False))
