"""Download market data and run reproducible walk-forward experiments."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from .evaluation import annual_scores, evaluate_high_volatility, evaluate_models, paired_loss_tests, regime_scores
from .features import make_dataset
from .preprocessing import prepare_prices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="SPY,QQQ,IWM,TLT,GLD", help="Comma-separated liquid ETFs: equities, Treasuries, and gold.")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None, help="Exclusive end date; omitted means latest available session.")
    parser.add_argument("--input-csv", type=Path, help="Run one local CSV instead of downloading the ticker universe.")
    parser.add_argument("--local-data-dir", type=Path, help="Use existing per-ticker CSV snapshots in this directory.")
    parser.add_argument("--ticker-label", help="Optional label for the local CSV series.")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--price-column", default="Adj Close")
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--min-train-size", type=int, default=2520, help="Initial training window in sessions (about 10 years).")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.input_csv:
        raw = pd.read_csv(args.input_csv)
        ticker_data = [(args.ticker_label or args.input_csv.stem.split("_")[0], raw, args.price_column, "local CSV", args.input_csv)]
    elif args.local_data_dir:
        ticker_data = []
        for ticker in [item.strip().upper() for item in args.tickers.split(",") if item.strip()]:
            matches = sorted(args.local_data_dir.glob(f"{ticker}_*.csv"))
            if not matches:
                raise SystemExit(f"No cached CSV snapshot found for {ticker} in {args.local_data_dir}.")
            data_path = matches[-1]
            ticker_data.append((ticker, pd.read_csv(data_path), "Close", "Yahoo Finance via yfinance; auto_adjust=True (local snapshot)", data_path))
    else:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise SystemExit("Install the download extra with: pip install -e '.[download]'") from exc
        ticker_data = []
        Path("data").mkdir(exist_ok=True)
        for ticker in [item.strip().upper() for item in args.tickers.split(",") if item.strip()]:
            raw = yf.download(ticker, start=args.start, end=args.end, auto_adjust=True, progress=False, multi_level_index=False).reset_index()
            if raw.empty:
                raise SystemExit(f"No data returned for {ticker}; check symbol, date range, and provider availability.")
            data_path = Path("data") / f"{ticker}_{args.start}_{args.end or 'latest'}.csv"
            raw.to_csv(data_path, index=False)
            ticker_data.append((ticker, raw, "Close", "Yahoo Finance via yfinance; auto_adjust=True", data_path))

    all_predictions, all_scores, all_event_predictions, all_event_scores = [], [], [], []
    data_hashes, usable_counts, first_forecast_dates, last_dates = {}, {}, {}, {}
    for ticker, raw, price_column, source, data_path in ticker_data:
        prices = prepare_prices(raw, args.date_column, price_column)
        x, y, y_vol = make_dataset(prices)
        if len(x) <= args.min_train_size:
            raise SystemExit(f"{ticker} has {len(x)} usable observations, not enough for initial train size {args.min_train_size}.")
        pred, fold_scores = evaluate_models(x, y, y_vol, args.min_train_size, args.test_size)
        event_pred, event_scores = evaluate_high_volatility(x, y_vol, args.min_train_size, args.test_size)
        for frame in (pred, fold_scores, event_pred, event_scores):
            frame["ticker"] = ticker
        all_predictions.append(pred)
        all_scores.append(fold_scores)
        all_event_predictions.append(event_pred)
        all_event_scores.append(event_scores)
        data_hashes[ticker] = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
        usable_counts[ticker] = int(len(x))
        first_forecast_dates[ticker] = str(x.index[args.min_train_size])
        last_dates[ticker] = str(x.index[-1])

    predictions = pd.concat(all_predictions, ignore_index=True)
    scores = pd.concat(all_scores, ignore_index=True)
    event_predictions = pd.concat(all_event_predictions, ignore_index=True)
    event_scores = pd.concat(all_event_scores, ignore_index=True)

    regime_frames, annual_frames, loss_frames = [], [], []
    for ticker in predictions.ticker.unique():
        ticker_predictions = predictions.loc[predictions.ticker == ticker]
        ticker_events = event_predictions.loc[(event_predictions.ticker == ticker) & (event_predictions.model == "logistic_regression")]
        thresholds = ticker_events.groupby("fold").threshold.first().to_dict()
        regime_frames.append(regime_scores(ticker_predictions, thresholds).assign(ticker=ticker))
        annual_frames.append(annual_scores(ticker_predictions).assign(ticker=ticker))
        loss_frames.append(paired_loss_tests(ticker_predictions).assign(ticker=ticker))

    out_dir = args.output_dir
    out_dir.mkdir(exist_ok=True)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    scores.to_csv(out_dir / "fold_metrics.csv", index=False)
    event_predictions.to_csv(out_dir / "high_volatility_predictions.csv", index=False)
    event_scores.to_csv(out_dir / "high_volatility_metrics.csv", index=False)
    pd.concat(regime_frames, ignore_index=True).to_csv(out_dir / "regime_metrics.csv", index=False)
    pd.concat(annual_frames, ignore_index=True).to_csv(out_dir / "annual_metrics.csv", index=False)
    combined_loss_tests = pd.concat(loss_frames, ignore_index=True)
    p = combined_loss_tests.p_value_two_sided.to_numpy()
    order = p.argsort()
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    combined_loss_tests["p_value_holm"] = adjusted
    combined_loss_tests.to_csv(out_dir / "paired_loss_tests.csv", index=False)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    summary = predictions.groupby(["ticker", "model"], as_index=False).apply(
        lambda g: pd.Series({"n": len(g), "mae": mean_absolute_error(g.actual_volatility, g.forecast_volatility), "rmse": mean_squared_error(g.actual_volatility, g.forecast_volatility) ** 0.5}),
        include_groups=False,
    ).reset_index(drop=True).sort_values(["ticker", "mae"])
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    event_summary = event_scores.groupby(["ticker", "model"], as_index=False)[["log_loss", "brier", "roc_auc", "average_precision", "ece_10_bins"]].mean()
    event_summary.to_csv(out_dir / "high_volatility_summary.csv", index=False)

    metadata = {
        "tickers": [item[0] for item in ticker_data],
        "data_sources": {item[0]: item[3] for item in ticker_data},
        "download_start": args.start if not args.input_csv else None,
        "download_end_exclusive": args.end if not args.input_csv else None,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "usable_observations_by_ticker": usable_counts,
        "first_forecast_date_by_ticker": first_forecast_dates,
        "last_observation_date_by_ticker": last_dates,
        "initial_train_size": args.min_train_size,
        "test_size_sessions": args.test_size,
        "random_seed": 17,
        "target": "next-session absolute close-to-close log return; regression models trained on log squared return",
        "data_sha256_by_ticker": data_hashes,
        "packages": {
            name: importlib.metadata.version(name)
            for name, module in {"numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn", "xgboost": "xgboost", "yfinance": "yfinance"}.items()
            if importlib.util.find_spec(module)
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Regression metrics (pooled out-of-sample observations; lower is better):")
    print(summary.to_string(index=False))
    print("\nHigh-volatility classification metrics (mean across folds):")
    print(event_summary.to_string(index=False))
    print("\nPaired loss comparisons; positive MAE reduction favors candidate:")
    print(combined_loss_tests.to_string(index=False))


if __name__ == "__main__":
    main()
