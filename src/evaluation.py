"""Leakage-safe walk-forward evaluation and shift diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def walk_forward_splits(n: int, min_train_size: int, test_size: int):
    """Yield expanding training indices and contiguous, non-overlapping test blocks."""
    if test_size < 1 or min_train_size < 1 or min_train_size >= n:
        raise ValueError("Require 1 <= min_train_size < n and test_size >= 1.")
    start = min_train_size
    while start < n:
        stop = min(start + test_size, n)
        yield np.arange(start), np.arange(start, stop)
        start = stop


def _min_train_size(n: int, value: int | None) -> int:
    result = value if value is not None else max(252, n // 2)
    if result >= n:
        raise ValueError(f"Need more observations: {n} rows, requested minimum train size {result}.")
    return result


def evaluate_models(
    x: pd.DataFrame,
    y_log_variance: pd.Series,
    y_volatility: pd.Series,
    min_train_size: int | None = None,
    test_size: int = 252,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit forecasts using past-only data; report errors in absolute-return units."""
    from .models import regression_models

    n = len(x)
    min_train_size = _min_train_size(n, min_train_size)
    models = regression_models(seed)
    predictions, scores = [], []
    vol_feature = "vol_21"
    if vol_feature not in x:
        raise ValueError("The historical-volatility baseline requires the vol_21 feature.")

    for fold, (train_idx, test_idx) in enumerate(walk_forward_splits(n, min_train_size, test_size), start=1):
        train_y = y_log_variance.iloc[train_idx]
        actual = y_volatility.iloc[test_idx].to_numpy()
        # For a Gaussian return, E|r| = sigma*sqrt(2/pi); the scale is fixed ex ante.
        forecasts = {
            "historical_volatility": x.iloc[test_idx][vol_feature].to_numpy() * np.sqrt(2 / np.pi)
        }
        for name, model in models.items():
            model.fit(x.iloc[train_idx], train_y)
            forecasts[name] = np.exp(np.clip(model.predict(x.iloc[test_idx]), -30, 10) / 2)

        for name, forecast in forecasts.items():
            for j, idx in enumerate(test_idx):
                predictions.append({
                    "date": x.index[idx], "fold": fold, "model": name,
                    "actual_volatility": actual[j], "forecast_volatility": forecast[j],
                })
            scores.append({
                "fold": fold, "model": name, "n": len(test_idx),
                "mae": mean_absolute_error(actual, forecast),
                "rmse": mean_squared_error(actual, forecast) ** 0.5,
            })
    return pd.DataFrame(predictions), pd.DataFrame(scores)


def regime_scores(predictions: pd.DataFrame, train_thresholds: dict[int, float]) -> pd.DataFrame:
    """Report forecast error when realized volatility is above/below a train-only cutoff."""
    data = predictions.copy()
    data["regime"] = data.apply(
        lambda row: "high" if row["actual_volatility"] > train_thresholds[row["fold"]] else "low", axis=1
    )
    rows = []
    for (fold, model, regime), group in data.groupby(["fold", "model", "regime"]):
        rows.append({
            "fold": fold, "model": model, "regime": regime, "n": len(group),
            "mae": mean_absolute_error(group.actual_volatility, group.forecast_volatility),
            "rmse": mean_squared_error(group.actual_volatility, group.forecast_volatility) ** 0.5,
        })
    return pd.DataFrame(rows)


def evaluate_high_volatility(
    x: pd.DataFrame,
    y_volatility: pd.Series,
    min_train_size: int | None = None,
    test_size: int = 252,
    seed: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast whether next-session volatility exceeds the training 80th percentile."""
    from .models import high_volatility_classifier

    min_train_size = _min_train_size(len(x), min_train_size)
    predictions, scores = [], []
    for fold, (train_idx, test_idx) in enumerate(walk_forward_splits(len(x), min_train_size, test_size), start=1):
        threshold = float(y_volatility.iloc[train_idx].quantile(0.8))
        y_train = (y_volatility.iloc[train_idx] > threshold).astype(int)
        y_test = (y_volatility.iloc[test_idx] > threshold).astype(int)
        model = high_volatility_classifier(seed)
        model.fit(x.iloc[train_idx], y_train)
        prior = float(y_train.mean())
        model_probability = np.clip(model.predict_proba(x.iloc[test_idx])[:, 1], 1e-6, 1 - 1e-6)
        probability_by_model = {
            "training_base_rate": np.repeat(prior, len(test_idx)),
            "logistic_regression": model_probability,
        }
        truth = y_test.to_numpy()
        for name, probability in probability_by_model.items():
            for i, idx in enumerate(test_idx):
                predictions.append({
                    "date": x.index[idx], "fold": fold, "model": name,
                    "threshold": threshold, "actual_event": truth[i], "probability": probability[i],
                })
            scores.append({
                "fold": fold, "model": name, "n": len(test_idx), "threshold": threshold,
                "event_rate": float(truth.mean()), "mean_probability": float(probability.mean()),
                "log_loss": log_loss(truth, probability, labels=[0, 1]),
                "brier": brier_score_loss(truth, probability),
                "roc_auc": roc_auc_score(truth, probability) if np.unique(truth).size == 2 else np.nan,
                "average_precision": average_precision_score(truth, probability),
                "ece_10_bins": float(_expected_calibration_error(truth, probability)),
            })
    return pd.DataFrame(predictions), pd.DataFrame(scores)


def _expected_calibration_error(truth: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """Equal-width expected calibration error (lower is better)."""
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    return float(sum(
        np.mean(bucket == i) * abs(np.mean(truth[bucket == i]) - np.mean(probability[bucket == i]))
        for i in range(bins) if np.any(bucket == i)
    ))


def annual_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year performance to expose time-localized failures."""
    data = predictions.copy()
    data["year"] = pd.to_datetime(data["date"]).dt.year
    return data.groupby(["year", "model"], as_index=False).apply(
        lambda g: pd.Series({
            "n": len(g), "mae": mean_absolute_error(g.actual_volatility, g.forecast_volatility),
            "rmse": mean_squared_error(g.actual_volatility, g.forecast_volatility) ** 0.5,
        }), include_groups=False
    ).reset_index(drop=True)


def paired_loss_tests(predictions: pd.DataFrame, baseline: str = "historical_volatility", max_lag: int = 21) -> pd.DataFrame:
    """HAC-adjusted paired absolute-error comparisons against the baseline.

    Positive mean difference means the candidate has lower MAE. Inference is
    descriptive because only one asset and one test span are evaluated.
    """
    from scipy.stats import norm

    data = predictions.copy()
    data["absolute_error"] = (data.actual_volatility - data.forecast_volatility).abs()
    wide = data.pivot(index="date", columns="model", values="absolute_error").sort_index()
    if baseline not in wide:
        raise ValueError(f"Baseline model {baseline!r} is absent.")
    rows = []
    for candidate in wide.columns:
        if candidate == baseline:
            continue
        delta = (wide[baseline] - wide[candidate]).dropna().to_numpy()
        n = len(delta)
        centered = delta - delta.mean()
        long_run_var = float(centered @ centered / n)
        for lag in range(1, min(max_lag, n - 1) + 1):
            gamma = float(centered[lag:] @ centered[:-lag] / n)
            long_run_var += 2 * (1 - lag / (max_lag + 1)) * gamma
        se = np.sqrt(max(long_run_var, 0) / n)
        z = delta.mean() / se if se else np.nan
        rows.append({
            "candidate": candidate, "baseline": baseline, "n": n,
            "mean_mae_reduction": float(delta.mean()), "hac_se": float(se),
            "z_stat": float(z), "p_value_two_sided": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
        })
    return pd.DataFrame(rows)
