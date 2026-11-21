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
                   "queue_imbalance_l1", "queue_imbalance_5", "ofi_l1"]


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


def simulate_taker_touch(frame: pd.DataFrame, test_start: int,
                         probability: np.ndarray, ticker: str, model_name: str) -> list[dict]:
    """Evaluate a fixed, one-share taker-touch counterfactual over 100 events.

    Entry is at the next observed event's BBO and exit is at the 100-event
    horizon. Trades are sequential and require one share displayed at both
    entry and exit. This omits market impact, queueing, and production latency.
    """
    candidate_trades: list[tuple[float, float]] = []
    cursor = test_start
    stop = len(frame) - HORIZON_EVENTS
    while cursor < stop:
        test_offset = cursor - test_start
        if test_offset >= len(probability):
            break
        side = int(LABELS[int(np.argmax(probability[test_offset]))])
        if side == 0:
            cursor += 1
            continue
        entry_idx = cursor + 1
        exit_idx = cursor + HORIZON_EVENTS
        entry = frame.iloc[entry_idx]
        exit_row = frame.iloc[exit_idx]
        if entry.bid_size_1 < 1 or entry.ask_size_1 < 1 or exit_row.bid_size_1 < 1 or exit_row.ask_size_1 < 1:
            cursor += 1
            continue
        entry_mid = (entry.best_bid_ticks + entry.best_ask_ticks) / 2
        if side > 0:
            gross_bps = (exit_row.best_bid_ticks - entry.best_ask_ticks) / entry_mid * 10000
        else:
            gross_bps = (entry.best_bid_ticks - exit_row.best_ask_ticks) / entry_mid * 10000
        candidate_trades.append((float(gross_bps), float(side)))
        cursor = exit_idx + 1  # avoid overlapping positions

    gross = np.asarray([trade[0] for trade in candidate_trades], dtype=float)
    rows = []
    for fee_bps in (0.0, 5.0, 10.0):
        net = gross - 2 * fee_bps
        cumulative = np.cumsum(net)
        drawdown = cumulative - np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[1:] if len(net) else np.array([])
        rows.append({
            "ticker": ticker, "model": model_name, "fee_bps_per_side": fee_bps,
            "entry_delay_events": 1, "holding_horizon_events": HORIZON_EVENTS,
            "n_trades": int(len(net)),
            "gross_mean_bps_per_trade": float(gross.mean()) if len(gross) else np.nan,
            "gross_win_rate": float((gross > 0).mean()) if len(gross) else np.nan,
            "net_mean_bps_per_trade": float(net.mean()) if len(net) else np.nan,
            "net_sum_bps_sequential": float(net.sum()) if len(net) else 0.0,
            "net_max_drawdown_bps": float(drawdown.min()) if len(drawdown) else 0.0,
        })
    return rows


def evaluate_symbol(frame: pd.DataFrame, ticker: str, train_fraction: float = 0.70) -> list[dict]:
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
                                         ticker, "training_class_prior")

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
        strategy_rows.extend(simulate_taker_touch(frame, test_start, probability, ticker, name))
    return rows, strategy_rows


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

    grouped = strategy.groupby(["model", "fee_bps_per_side"], as_index=False).agg(
        n_trades=("n_trades", "sum"), net_sum=("net_sum_bps_sequential", "sum"))
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


def render_report(metrics: pd.DataFrame, strategy: pd.DataFrame,
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
                  "", "## Taker-touch cost sensitivity", "",
                  "Predicted directional classes are entered at the next event's displayed best price and exited at the opposite best price 100 events after the signal. Trades are one share, sequential, and skipped if displayed top-level size is below one share. Fees are scenario assumptions per side; spread crossing is included in gross P&L.",
                  "",
                  "This is a counterfactual benchmark, not a full simulator: it does not model queue priority, market impact, fees specific to a participant, or latency beyond the next event. It must not be read as a profitable strategy result.",
                  "",
                  "| Model | Fee per side (bps) | Trades | Mean net bps/trade | Sum of sequential trade bps/share |",
                  "|---|---:|---:|---:|---:|"])
    grouped = strategy.groupby(["model", "fee_bps_per_side"], as_index=False).agg(
        n_trades=("n_trades", "sum"), net_sum=("net_sum_bps_sequential", "sum"))
    for row in grouped.sort_values(["model", "fee_bps_per_side"]).itertuples(index=False):
        mean = row.net_sum / row.n_trades if row.n_trades else float("nan")
        mean_text = f"{mean:.3f}" if np.isfinite(mean) else "n/a"
        lines.append(f"| {row.model} | {row.fee_bps_per_side:.1f} | {row.n_trades:,} | {mean_text} | {row.net_sum:.1f} |")
    lines.extend(["", "![Mean net result per trade under fee scenarios](taker_touch_fee_sensitivity.png)",
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
            ticker_metrics, ticker_strategy = evaluate_symbol(prepared, ticker)
            rows.extend(ticker_metrics)
            strategy_rows.extend(ticker_strategy)

    results_dir = project_root / "results" / "market_microstructure" / "lobster_sample"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_dir / "summary_metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame(strategy_rows).to_csv(results_dir / "strategy_sensitivity.csv", index=False,
                                       float_format="%.6f")
    render_plot(metrics, results_dir / "chronological_macro_f1.png")
    render_execution_plot(pd.DataFrame(strategy_rows), results_dir / "taker_touch_fee_sensitivity.png")
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
            "entry": "one share at the next event's displayed top of book",
            "exit": "opposite displayed top of book 100 events after the signal",
            "position_overlap": "prevented; next signal is considered after the prior exit",
            "fees_bps_per_side": [0.0, 5.0, 10.0],
            "limits": "no market impact or queue model; one-day scenarios, not production fills or profitability evidence",
        },
        "feature_extraction_runtime_seconds_by_symbol": elapsed_by_symbol,
        "compiler": "g++/clang++ C++17 -O3",
        "runtime_platform": platform.platform(),
        "feature_output": "little-endian float64 records; column order defined by FEATURE_COLUMNS in src/run_lobster_sample.py",
        "package_versions": package_versions,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": ["one sample trading day only", "not a distribution-shift or persistence claim",
                        "counterfactual taker-touch P&L is not a full execution model"],
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    render_report(metrics, pd.DataFrame(strategy_rows), metadata, results_dir / "report.md")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reuse-features", action="store_true",
                        help="skip C++ feature extraction and reuse local ignored binary files")
    args = parser.parse_args()
    print(run(args.project_root, reuse_features=args.reuse_features).to_string(index=False))
