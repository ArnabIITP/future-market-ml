"""Multi-session NASDAQ BX ITCH study with a locked chronological holdout.

Raw files are official Nasdaq sample data and remain under the git-ignored
``data/`` tree. Market-message decoding and book reconstruction use MeatPy;
the study-specific sampling, features, split, metrics, and reports live here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time as day_time, timedelta
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "nasdaq_bx_itch" / "raw"
SNAPSHOT_DIR = ROOT / "data" / "nasdaq_bx_itch" / "snapshots"
RESULTS_DIR = ROOT / "results" / "nasdaq_bx_itch"
BASE_URL = "https://emi.nasdaq.com/ITCH/Nasdaq%20BX%20ITCH/"
MARCH_URL = BASE_URL + "March%2020/"
SYMBOLS = ("AAPL", "MSFT", "INTC")
SAMPLE_INTERVAL_NS = 100_000_000  # 100 ms snapshots
HORIZON_SNAPSHOTS = 10  # one second
FLAT_MOVE_PRICE_UNITS = 100  # $0.01 in ITCH's $0.0001 integer price units
CLASSES = np.array([-1, 0, 1], dtype=np.int64)
CLASS_NAMES = { -1: "down", 0: "flat", 1: "up" }
FEATURES = [
    "spread_bps", "imbalance_l1", "imbalance_5", "microprice_offset_bps",
    "mid_return_1", "mid_return_10", "mid_return_50", "realized_vol_1s",
    "time_of_day_sin", "time_of_day_cos", "ticker_AAPL", "ticker_MSFT",
]
BASELINE_FEATURES = ["imbalance_l1", "imbalance_5"]


@dataclass(frozen=True)
class Session:
    session_date: str
    filename: str
    url: str
    expected_bytes: int
    split: str


# The final three dates are frozen as the untouched test period. They are not
# loaded or scored during model selection. Dates are official exchange samples.
SESSIONS = (
    Session("2019-01-30", "20190130.BX_ITCH_50.gz", BASE_URL + "20190130.BX_ITCH_50.gz", 1_097_205_622, "train"),
    Session("2019-05-30", "20190530.BX_ITCH_50.gz", BASE_URL + "20190530.BX_ITCH_50.gz", 497_884_404, "train"),
    Session("2019-07-30", "20190730.BX_ITCH_50.gz", BASE_URL + "20190730.BX_ITCH_50.gz", 391_242_214, "train"),
    Session("2019-08-30", "20190830.BX_ITCH_50.gz", BASE_URL + "20190830.BX_ITCH_50.gz", 449_303_567, "validation"),
    Session("2019-10-30", "20191030.BX_ITCH_50.gz", BASE_URL + "20191030.BX_ITCH_50.gz", 403_851_104, "validation"),
    Session("2019-12-30", "20191230.BX_ITCH_50.gz", BASE_URL + "20191230.BX_ITCH_50.gz", 390_561_039, "test"),
    Session("2020-01-30", "20200130.BX_ITCH_50.gz", BASE_URL + "20200130.BX_ITCH_50.gz", 721_906_618, "test"),
    Session("2020-03-02", "S030220-v50-bx.txt.gz", MARCH_URL + "S030220-v50-bx.txt.gz", 1_503_031_926, "test"),
)


def sessions_for(split: str) -> tuple[Session, ...]:
    if split not in {"train", "validation", "test", "development", "all"}:
        raise ValueError("split must be train, validation, test, development, or all")
    if split == "all":
        return SESSIONS
    if split == "development":
        return tuple(s for s in SESSIONS if s.split != "test")
    return tuple(s for s in SESSIONS if s.split == split)


def download_sessions(split: str = "development", *, data_dir: Path = RAW_DIR) -> list[dict]:
    """Download official sample archives; verify size, gzip CRC, and SHA-256."""
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for session in sessions_for(split):
        path = data_dir / session.filename
        if not path.exists() or path.stat().st_size != session.expected_bytes:
            path.unlink(missing_ok=True)
            partial = path.with_suffix(path.suffix + ".part")
            partial.unlink(missing_ok=True)
            written = 0
            with requests.get(session.url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) != session.expected_bytes:
                    raise ValueError(f"Unexpected Content-Length for {session.filename}: {content_length}")
                with partial.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            out.write(chunk)
                            written += len(chunk)
            if written != session.expected_bytes:
                partial.unlink(missing_ok=True)
                raise ValueError(f"Incomplete {session.filename}: {written} bytes")
            partial.replace(path)
        digest = sha256_file(path)
        # The subsequent full event-stream pass validates the gzip CRC as it
        # reconstructs the book, avoiding a second full decompression here.
        records.append({**asdict(session), "local_path": path.relative_to(ROOT).as_posix(), "sha256": digest})
        print(f"Verified {session.session_date}: {path.stat().st_size:,} bytes; SHA-256 {digest}", flush=True)
    return records


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), ROOT / "src" / "nasdaq_queue_simulation.py", ROOT / "pyproject.toml"):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _level_features(lob) -> dict[str, float]:
    if lob is None or not lob.bid_levels or not lob.ask_levels:
        empty = {k: np.nan for k in (
            "bid_price", "ask_price", "bid_size_1", "ask_size_1", "mid_price",
            "spread_bps", "imbalance_l1", "imbalance_5", "microprice_offset_bps",
        )}
        for level in range(1, 6):
            empty[f"bid_price_{level}"] = empty[f"bid_size_{level}"] = np.nan
            empty[f"ask_price_{level}"] = empty[f"ask_size_{level}"] = np.nan
        return empty
    bids, asks = lob.bid_levels[:5], lob.ask_levels[:5]
    bid = float(bids[0].price)
    ask = float(asks[0].price)
    bid1 = float(bids[0].volume)
    ask1 = float(asks[0].volume)
    bid5 = float(sum(x.volume for x in bids))
    ask5 = float(sum(x.volume for x in asks))
    mid = (bid + ask) / 2.0
    microprice = (ask * bid1 + bid * ask1) / (bid1 + ask1) if bid1 + ask1 else mid
    result = {
        "bid_price": bid, "ask_price": ask, "bid_size_1": bid1,
        "ask_size_1": ask1, "mid_price": mid,
        "spread_bps": (ask - bid) / mid * 10_000 if mid else np.nan,
        "imbalance_l1": (bid1 - ask1) / (bid1 + ask1) if bid1 + ask1 else 0.0,
        "imbalance_5": (bid5 - ask5) / (bid5 + ask5) if bid5 + ask5 else 0.0,
        "microprice_offset_bps": (microprice - mid) / mid * 10_000 if mid else np.nan,
    }
    for level in range(1, 6):
        result[f"bid_price_{level}"] = float(bids[level - 1].price) if level <= len(bids) else np.nan
        result[f"bid_size_{level}"] = float(bids[level - 1].volume) if level <= len(bids) else 0.0
        result[f"ask_price_{level}"] = float(asks[level - 1].price) if level <= len(asks) else np.nan
        result[f"ask_size_{level}"] = float(asks[level - 1].volume) if level <= len(asks) else 0.0
    return result


def extract_session(session: Session, *, raw_dir: Path = RAW_DIR,
                    output_dir: Path = SNAPSHOT_DIR,
                    symbols: Iterable[str] = SYMBOLS,
                    interval_ns: int = SAMPLE_INTERVAL_NS,
                    include_order_events: bool = False) -> pd.DataFrame:
    """Reconstruct selected books from one raw ITCH session and sample every 100ms."""
    from meatpy.itch50 import (
        AddOrderMessage, AddOrderMPIDMessage, ITCH50MarketProcessor,
        ITCH50MessageReader, OrderCancelMessage, OrderDeleteMessage,
        OrderExecutedMessage, OrderExecutedPriceMessage, OrderReplaceMessage,
        StockDirectoryMessage,
    )

    symbols = tuple(symbols)
    file_path = raw_dir / session.filename
    if not file_path.exists() or file_path.stat().st_size != session.expected_bytes:
        raise FileNotFoundError(f"Missing or incomplete raw session: {file_path}")
    session_day = date.fromisoformat(session.session_date)
    book_date = datetime.combine(session_day, day_time())
    processors = {
        symbol: ITCH50MarketProcessor((symbol + " " * (8 - len(symbol))).encode(), book_date)
        for symbol in symbols
    }
    loc_to_symbol: dict[int, str] = {}
    open_ns = (9 * 3600 + 30 * 60) * 1_000_000_000
    close_ns = (16 * 3600) * 1_000_000_000
    next_sample = open_ns
    rows: list[dict] = []
    event_state: dict[str, dict[int, tuple[str, int, int]]] = {s: {} for s in symbols}
    event_buffers: dict[str, list[dict]] = {s: [] for s in symbols}
    event_writers: dict[str, object] = {}
    raw_messages = routed_messages = 0
    started = time.perf_counter()

    def write_event_batch(symbol: str, *, close: bool = False) -> None:
        if not include_order_events:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        buffer = event_buffers[symbol]
        if buffer:
            table = pa.Table.from_pylist(buffer)
            writer = event_writers.get(symbol)
            if writer is None:
                path = output_dir / "events" / f"{session.session_date}_{symbol}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                event_writers[symbol] = writer
            writer.write_table(table, row_group_size=len(buffer))
            buffer.clear()
        if close and symbol in event_writers:
            event_writers.pop(symbol).close()

    def event_row(symbol: str, message, sequence: int) -> dict | None:
        """Capture only public-book mutations, retaining exact message order."""
        state = event_state[symbol]
        if isinstance(message, (AddOrderMessage, AddOrderMPIDMessage)):
            order_id = int(message.order_ref)
            if order_id in state:
                raise ValueError(f"Duplicate live order id {order_id} in {symbol}")
            side = message.bsindicator.decode("ascii")
            price, quantity = int(message.price), int(message.shares)
            state[order_id] = (side, price, quantity)
            return {"timestamp_ns": int(message.timestamp), "sequence": sequence,
                    "ticker": symbol, "kind": "add", "order_id": order_id,
                    "new_order_id": None, "side": side, "price": price,
                    "quantity": quantity}
        if isinstance(message, (OrderExecutedMessage, OrderExecutedPriceMessage,
                                OrderCancelMessage, OrderDeleteMessage)):
            order_id = int(message.order_ref)
            if order_id not in state:
                raise ValueError(f"Unknown order id {order_id} in {symbol} {type(message).__name__}")
            side, price, remaining = state[order_id]
            if isinstance(message, (OrderExecutedMessage, OrderExecutedPriceMessage)):
                kind = "execute"
                quantity = int(message.shares)
            elif isinstance(message, OrderCancelMessage):
                kind = "cancel"
                quantity = int(message.canceled_shares)
            else:
                kind, quantity = "delete", remaining
            if quantity <= 0 or quantity > remaining:
                raise ValueError(f"Invalid {kind} size {quantity} for order {order_id} ({remaining})")
            left = remaining - quantity
            if left:
                state[order_id] = (side, price, left)
            else:
                del state[order_id]
            return {"timestamp_ns": int(message.timestamp), "sequence": sequence,
                    "ticker": symbol, "kind": kind, "order_id": order_id,
                    "new_order_id": None, "side": side, "price": price,
                    "quantity": quantity}
        if isinstance(message, OrderReplaceMessage):
            old_id, new_id = int(message.original_ref), int(message.new_ref)
            if old_id not in state or new_id in state:
                raise ValueError(f"Invalid replace {old_id}->{new_id} in {symbol}")
            side, _, _ = state.pop(old_id)
            price, quantity = int(message.price), int(message.shares)
            state[new_id] = (side, price, quantity)
            return {"timestamp_ns": int(message.timestamp), "sequence": sequence,
                    "ticker": symbol, "kind": "replace", "order_id": old_id,
                    "new_order_id": new_id, "side": side, "price": price,
                    "quantity": quantity}
        return None

    def sample(at_ns: int) -> None:
        for symbol, processor in processors.items():
            row = {"session_date": session.session_date, "ticker": symbol,
                   "timestamp_ns": at_ns, **_level_features(processor.current_lob)}
            rows.append(row)

    with ITCH50MessageReader(file_path) as reader:
        for message in reader:
            raw_messages += 1
            timestamp_ns = int(getattr(message, "timestamp", 0))
            # Snapshot the prior message state at each grid time, then apply the
            # current event. This preserves causal ordering at the time boundary.
            while next_sample < min(timestamp_ns, close_ns):
                sample(next_sample)
                next_sample += interval_ns
            if isinstance(message, StockDirectoryMessage):
                symbol = message.stock.decode("ascii", errors="ignore").strip()
                loc_to_symbol[message.stock_locate] = symbol
                if symbol in processors:
                    processors[symbol].process_message(message)
                    routed_messages += 1
                continue
            symbol = None
            raw_stock = getattr(message, "stock", None)
            if raw_stock:
                symbol = raw_stock.decode("ascii", errors="ignore").strip()
            if symbol not in processors:
                symbol = loc_to_symbol.get(getattr(message, "stock_locate", -1))
            if symbol in processors:
                if include_order_events:
                    row = event_row(symbol, message, raw_messages)
                    if row is not None:
                        event_buffers[symbol].append(row)
                        if len(event_buffers[symbol]) >= 50_000:
                            write_event_batch(symbol)
                processors[symbol].process_message(message)
                routed_messages += 1
            # If the feed has an event exactly on the sample boundary, include
            # that event in the corresponding end-of-interval book state.
            if timestamp_ns == next_sample and next_sample < close_ns:
                sample(next_sample)
                next_sample += interval_ns
    while next_sample < close_ns:
        sample(next_sample)
        next_sample += interval_ns
    for processor in processors.values():
        if include_order_events:
            lob = processor.current_lob
            engine_state: dict[int, tuple[str, int, int]] = {}
            if lob is not None:
                for side, levels in (("B", lob.bid_levels), ("S", lob.ask_levels)):
                    for level in levels:
                        for order in level.queue:
                            engine_state[int(order.order_id)] = (
                                side, int(level.price), int(order.volume)
                            )
            if engine_state != event_state[str(processor.instrument.decode().strip())]:
                raise ValueError("Independent event-log state does not reconcile to MeatPy's reconstructed book")
        processor.processing_done()
    for symbol in symbols:
        write_event_batch(symbol, close=True)

    frame = pd.DataFrame(rows)
    if frame.empty or frame["mid_price"].notna().sum() == 0:
        raise ValueError(f"No usable regular-session book snapshots in {session.session_date}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{session.session_date}.parquet"
    frame.to_parquet(out_path, index=False, compression="zstd")
    metadata = {
        "session_date": session.session_date, "source_file": session.filename,
        "raw_sha256": sha256_file(file_path), "raw_messages": raw_messages,
        "routed_messages": routed_messages, "snapshots_per_symbol": len(frame) // len(symbols),
        "order_event_logs_written": include_order_events,
        "symbols": list(symbols), "interval_ns": interval_ns,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "output_path": str(out_path), "output_sha256": sha256_file(out_path),
    }
    metadata_path = out_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{session.session_date}: {raw_messages:,} messages; {routed_messages:,} routed; "
          f"{len(frame):,} snapshots; {metadata['elapsed_seconds']:.1f}s", flush=True)
    return frame


def load_snapshots(sessions: Iterable[Session], *, snapshot_dir: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    parts = []
    for session in sessions:
        path = snapshot_dir / f"{session.session_date}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Extract snapshots first: {path}")
        part = pd.read_parquet(path)
        part["split"] = session.split
        parts.append(part)
    return make_features(pd.concat(parts, ignore_index=True))


def feature_shift_by_session(reference: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    """Compute train-binned PSI summaries per session and ticker."""
    rows = []
    for (day, ticker), sample in comparison.groupby(["session_date", "ticker"], sort=True):
        base = reference[reference.ticker == ticker]
        psi_values = []
        for feature in FEATURES:
            train_values = base[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            test_values = sample[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(train_values) == 0 or len(test_values) == 0:
                continue
            edges = np.unique(np.quantile(train_values, np.linspace(0, 1, 11)))
            if len(edges) < 2:
                continue
            edges[0], edges[-1] = -np.inf, np.inf
            train_share = np.histogram(train_values, bins=edges)[0].astype(float)
            test_share = np.histogram(test_values, bins=edges)[0].astype(float)
            epsilon = 1e-6
            train_share = (train_share + epsilon) / (len(train_values) + epsilon * len(train_share))
            test_share = (test_share + epsilon) / (len(test_values) + epsilon * len(test_share))
            psi_values.append(float(np.sum((test_share - train_share) * np.log(test_share / train_share))))
        rows.append({
            "session_date": day, "ticker": ticker, "features_scored": len(psi_values),
            "mean_feature_psi": float(np.mean(psi_values)) if psi_values else np.nan,
            "median_feature_psi": float(np.median(psi_values)) if psi_values else np.nan,
            "p90_feature_psi": float(np.quantile(psi_values, 0.9)) if psi_values else np.nan,
            "max_feature_psi": float(np.max(psi_values)) if psi_values else np.nan,
        })
    return pd.DataFrame(rows)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal returns and one-second realized volatility per day/symbol."""
    frame = frame.sort_values(["session_date", "ticker", "timestamp_ns"]).copy()
    grouped = frame.groupby(["session_date", "ticker"], sort=False)
    for lag in (1, 10, 50):
        frame[f"mid_return_{lag}"] = grouped["mid_price"].transform(lambda x: x.pct_change(lag) * 10_000)
    frame["realized_vol_1s"] = frame.groupby(["session_date", "ticker"], sort=False)["mid_return_1"].transform(
        lambda x: x.rolling(10, min_periods=5).std(ddof=0)
    )
    seconds = (frame["timestamp_ns"] % (24 * 3600 * 1_000_000_000)) / 1e9
    phase = 2.0 * np.pi * (seconds - 9.5 * 3600) / (6.5 * 3600)
    frame["time_of_day_sin"] = np.sin(phase)
    frame["time_of_day_cos"] = np.cos(phase)
    frame["ticker_AAPL"] = (frame["ticker"] == "AAPL").astype(float)
    frame["ticker_MSFT"] = (frame["ticker"] == "MSFT").astype(float)
    future = frame.groupby(["session_date", "ticker"], sort=False)["mid_price"].shift(-HORIZON_SNAPSHOTS)
    delta = future - frame["mid_price"]
    frame["target"] = np.select(
        [delta < -FLAT_MOVE_PRICE_UNITS, delta > FLAT_MOVE_PRICE_UNITS],
        [-1, 1], default=0,
    ).astype(float)
    frame.loc[future.isna(), "target"] = np.nan
    # Replace infinities one numeric column at a time. DataFrames span several
    # million rows here; DataFrame.replace builds a full-width temporary mask.
    for column in frame.columns:
        if not pd.api.types.is_float_dtype(frame[column].dtype):
            continue
        values = frame[column].to_numpy(copy=False)
        invalid = ~np.isfinite(values)
        if invalid.any():
            values[invalid] = np.nan
    return frame


def _model_set(seed: int = 2026) -> dict:
    return {
        "queue_imbalance_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.05, max_iter=300, class_weight="balanced", random_state=seed)
        ),
        "engineered_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.05, max_iter=300, class_weight="balanced", random_state=seed)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=180, max_depth=12, min_samples_leaf=60, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=1, random_state=seed,
        ),
        "xgboost": XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=180, max_depth=4,
            learning_rate=0.05, min_child_weight=50, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=5.0, tree_method="hist",
            n_jobs=4, random_state=seed, eval_metric="mlogloss",
        ),
    }


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = CLASSES[probability.argmax(axis=1)]
    truth = np.eye(3)[y + 1]
    confidence = probability.max(axis=1)
    correct = prediction == y
    # Ten fixed bins; this is a descriptive calibration error, not a test.
    ece = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        selected = (confidence >= low) & (confidence < low + 0.1)
        if selected.any():
            ece += selected.mean() * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return {
        "macro_f1": float(f1_score(y, prediction, labels=CLASSES, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probability, labels=CLASSES)),
        "multiclass_brier": float(np.mean(np.sum((probability - truth) ** 2, axis=1))),
        "top_label_ece_10bin": float(ece),
        "label_share_down": float(np.mean(y == -1)),
        "label_share_flat": float(np.mean(y == 0)),
        "label_share_up": float(np.mean(y == 1)),
    }


def run_development(*, snapshot_dir: Path = SNAPSHOT_DIR,
                    results_dir: Path = RESULTS_DIR) -> tuple[pd.DataFrame, str]:
    """Fit on train dates, select by validation log loss, and write the lock."""
    results_dir.mkdir(parents=True, exist_ok=True)
    development = load_snapshots(sessions_for("development"), snapshot_dir=snapshot_dir)
    train = development[(development.split == "train") & development.target.notna()]
    valid = development[(development.split == "validation") & development.target.notna()]
    train = train.dropna(subset=FEATURES)
    valid = valid.dropna(subset=FEATURES)
    if train.empty or valid.empty:
        raise ValueError("Need non-empty training and validation folds")
    feature_shift_by_session(train, valid).to_csv(
        results_dir / "validation_feature_shift.csv", index=False, float_format="%.8f"
    )
    features_by_model = {
        "queue_imbalance_logistic": BASELINE_FEATURES,
        "engineered_logistic": FEATURES,
        "random_forest": FEATURES,
        "xgboost": FEATURES,
    }
    rows, validation_predictions = [], []
    for name, model in _model_set().items():
        columns = features_by_model[name]
        train_y = train.target.astype(int).to_numpy()
        model_y = train_y + 1 if name == "xgboost" else train_y
        model.fit(train[columns].to_numpy(), model_y)
        probabilities = model.predict_proba(valid[columns].to_numpy())
        # Ensure stable class column order, including rare missing classes.
        probabilities = _reorder_probabilities(model, probabilities,
                                               encoded=name == "xgboost")
        valid_results = valid[["session_date", "ticker", "timestamp_ns", "target",
                               "bid_price", "ask_price"]].copy()
        valid_results["p_down"], valid_results["p_flat"], valid_results["p_up"] = probabilities.T
        valid_results["model"] = name
        validation_predictions.append(valid_results)
        for (day, ticker), group in valid_results.groupby(["session_date", "ticker"]):
            p = group[["p_down", "p_flat", "p_up"]].to_numpy()
            rows.append({"model": name, "session_date": day, "ticker": ticker,
                         "n": len(group), **_metrics(group.target.astype(int).to_numpy(), p)})
    prior = np.array([(train.target.astype(int) == c).mean() for c in CLASSES])
    prior_results = valid[["session_date", "ticker", "timestamp_ns", "target",
                           "bid_price", "ask_price"]].copy()
    prior_results[["p_down", "p_flat", "p_up"]] = prior
    prior_results["model"] = "training_class_prior"
    validation_predictions.append(prior_results)
    for (day, ticker), group in prior_results.groupby(["session_date", "ticker"]):
        p = group[["p_down", "p_flat", "p_up"]].to_numpy()
        rows.append({"model": "training_class_prior", "session_date": day, "ticker": ticker,
                     "n": len(group), **_metrics(group.target.astype(int).to_numpy(), p)})
    day_metrics = pd.DataFrame(rows)
    day_metrics.to_csv(results_dir / "validation_by_day.csv", index=False, float_format="%.8f")
    pd.concat(validation_predictions, ignore_index=True).to_parquet(
        results_dir / "validation_predictions.parquet", index=False, compression="zstd"
    )
    rank = day_metrics[day_metrics.model != "training_class_prior"].groupby("model").agg(
        validation_log_loss=("log_loss", "mean"), validation_macro_f1=("macro_f1", "mean"),
        validation_brier=("multiclass_brier", "mean"),
    ).sort_values(["validation_log_loss", "validation_macro_f1"], ascending=[True, False])
    selected = str(rank.index[0])
    lock = {
        "study": "multi-session Nasdaq BX ITCH 5.0 prediction under temporal shift",
        "locked_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "train_sessions": [s.session_date for s in sessions_for("train")],
        "validation_sessions": [s.session_date for s in sessions_for("validation")],
        "untouched_test_sessions": [s.session_date for s in sessions_for("test")],
        "symbols": list(SYMBOLS), "snapshot_interval_ns": SAMPLE_INTERVAL_NS,
        "target_horizon_snapshots": HORIZON_SNAPSHOTS,
        "target_horizon_ms": HORIZON_SNAPSHOTS * SAMPLE_INTERVAL_NS / 1e6,
        "flat_threshold_price_units": FLAT_MOVE_PRICE_UNITS,
        "primary_selection_metric": "mean validation log loss per day and symbol; macro-F1 tie-break",
        "selected_model": selected,
        "feature_sets": features_by_model,
        "model_parameters": {
            "queue_imbalance_logistic": "StandardScaler; balanced LogisticRegression C=0.05, max_iter=300",
            "engineered_logistic": "StandardScaler; balanced LogisticRegression C=0.05, max_iter=300",
            "random_forest": "180 trees, depth 12, leaf 60, sqrt features, balanced_subsample, n_jobs=1",
            "xgboost": "180 trees, depth 4, learning_rate 0.05, min_child_weight 50, fixed settings",
        },
        "random_seed": 2026,
        "software_versions": {
            package: __import__("importlib.metadata", fromlist=["version"]).version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "meatpy", "pyarrow")
        },
        "metrics": ["macro_f1", "log_loss", "multiclass_brier"],
        "calibration_metric": "10-bin top-label expected calibration error (descriptive)",
        "passive_execution": {
            "order_shares": 10,
            "entry_delays_ms": [1, 10, 50],
            "order_lifetime_ms": 1000,
            "fees_bps_per_side": [0, 1, 5],
            "non_overlapping_positions_per_symbol": True,
            "queue_ahead": "exact public order IDs at the posted price; executions, cancels, deletes, and replaces reduce ahead",
            "exit": "sweep five displayed levels; residual marked to mid",
        },
        "implementation_sha256": implementation_fingerprint(),
        "validation_rank": rank.reset_index().to_dict(orient="records"),
        "final_test_will_be_run_once_after_this_lock": True,
    }
    lock_path = results_dir / "protocol_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return day_metrics, selected


def _reorder_probabilities(model, raw: np.ndarray, *, encoded: bool = False) -> np.ndarray:
    output = np.zeros((len(raw), 3), dtype=float)
    for i, label in enumerate(model.classes_):
        output[:, int(label) if encoded else int(label) + 1] = raw[:, i]
    return output


def run_final_evaluation(*, snapshot_dir: Path = SNAPSHOT_DIR,
                         results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Run the predeclared final dates once, after a protocol lock exists."""
    results_dir.mkdir(parents=True, exist_ok=True)
    lock_path = results_dir / "protocol_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("Run development/model selection before final evaluation")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    marker = results_dir / "final_test_evaluated.json"
    if marker.exists():
        raise FileExistsError("Final-test marker exists; the holdout is a one-run evaluation")
    if lock.get("untouched_test_sessions") != [s.session_date for s in sessions_for("test")]:
        raise ValueError("Locked final-test dates do not match the frozen session split")
    if lock.get("implementation_sha256") != implementation_fingerprint():
        raise ValueError("Study code changed after the protocol lock; do not evaluate this holdout")
    # Refit each predeclared model on train+validation only. No test-day outcomes
    # feed fitting, feature selection, thresholding, or model selection.
    trainval_sessions = tuple(s for s in SESSIONS if s.split in {"train", "validation"})
    trainval = load_snapshots(trainval_sessions, snapshot_dir=snapshot_dir)
    test = load_snapshots(sessions_for("test"), snapshot_dir=snapshot_dir)
    trainval = trainval[trainval.target.notna()].dropna(subset=FEATURES)
    test = test[test.target.notna()].dropna(subset=FEATURES)
    feature_shift_by_session(trainval, test).to_csv(
        results_dir / "final_test_feature_shift.csv", index=False, float_format="%.8f"
    )
    features_by_model = lock["feature_sets"]
    predictions, metrics = [], []
    for name, model in _model_set().items():
        columns = features_by_model[name]
        train_y = trainval.target.astype(int).to_numpy()
        model_y = train_y + 1 if name == "xgboost" else train_y
        model.fit(trainval[columns].to_numpy(), model_y)
        probability = _reorder_probabilities(
            model, model.predict_proba(test[columns].to_numpy()), encoded=name == "xgboost"
        )
        pframe = test[["session_date", "ticker", "timestamp_ns", "target",
                       "bid_price", "ask_price"]].copy()
        pframe["model"] = name
        pframe[["p_down", "p_flat", "p_up"]] = probability
        predictions.append(pframe)
        for (day, ticker), group in pframe.groupby(["session_date", "ticker"]):
            p = group[["p_down", "p_flat", "p_up"]].to_numpy()
            metrics.append({"model": name, "session_date": day, "ticker": ticker,
                            "n": len(group), **_metrics(group.target.astype(int).to_numpy(), p)})
    prior = np.array([(trainval.target.astype(int) == c).mean() for c in CLASSES])
    prior_p = np.tile(prior, (len(test), 1))
    prior_frame = test[["session_date", "ticker", "timestamp_ns", "target",
                        "bid_price", "ask_price"]].copy()
    prior_frame["model"] = "training_class_prior"
    prior_frame[["p_down", "p_flat", "p_up"]] = prior_p
    predictions.append(prior_frame)
    for (day, ticker), group in prior_frame.groupby(["session_date", "ticker"]):
        p = group[["p_down", "p_flat", "p_up"]].to_numpy()
        metrics.append({"model": "training_class_prior", "session_date": day, "ticker": ticker,
                        "n": len(group), **_metrics(group.target.astype(int).to_numpy(), p)})
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    # Ensure a final test cannot be rerun with different code after data were read.
    prediction_frame.to_parquet(results_dir / "final_test_predictions.parquet", index=False, compression="zstd")
    metric_frame.to_csv(results_dir / "final_test_by_day.csv", index=False, float_format="%.8f")
    lock["final_test_evaluation_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    lock["final_test_predictions_sha256"] = sha256_file(results_dir / "final_test_predictions.parquet")
    lock["final_test_metrics_sha256"] = sha256_file(results_dir / "final_test_by_day.csv")
    lock["final_test_used_for_model_selection"] = False
    marker.write_text(json.dumps({"evaluated_once": True, "selected_model": lock["selected_model"],
                                  "results": "final_test_by_day.csv"}, indent=2), encoding="utf-8")
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return metric_frame


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("--split", choices=("train", "validation", "development", "test", "all"), default="development")
    extract = sub.add_parser("extract")
    extract.add_argument("--split", choices=("train", "validation", "development", "test", "all"), default="development")
    extract.add_argument("--include-order-events", action="store_true",
                         help="Write order-ID event logs for passive queue replay")
    sub.add_parser("development")
    sub.add_parser("final")
    args = parser.parse_args()
    if args.command == "download":
        if args.split in {"test", "all"} and not (RESULTS_DIR / "protocol_lock.json").exists():
            raise SystemExit("Lock development choices before downloading the final test period.")
        records = download_sessions(args.split)
        (RESULTS_DIR / "data_provenance.json").parent.mkdir(parents=True, exist_ok=True)
        existing = RESULTS_DIR / "data_provenance.json"
        previous = json.loads(existing.read_text()) if existing.exists() else {}
        previous.update({row["session_date"]: row for row in records})
        existing.write_text(json.dumps(previous, indent=2), encoding="utf-8")
    elif args.command == "extract":
        if args.split in {"test", "all"} and not (RESULTS_DIR / "protocol_lock.json").exists():
            raise SystemExit("Lock development choices before extracting the final test period.")
        if args.split in {"test", "all"} and (RESULTS_DIR / "final_test_evaluated.json").exists():
            raise SystemExit("The final-test holdout was already evaluated; do not reprocess it for tuning.")
        for session in sessions_for(args.split):
            extract_session(session, include_order_events=args.include_order_events)
    elif args.command == "development":
        metrics, selected = run_development()
        print(metrics.to_string(index=False))
        print(f"Validation selected: {selected}")
    else:
        metrics = run_final_evaluation()
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
