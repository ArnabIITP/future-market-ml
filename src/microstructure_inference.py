"""Day-block uncertainty summaries for the FI-2010 anchored folds.

The nine fold test days are chronological and not independent. We therefore
report descriptive moving-block bootstrap intervals, not p-values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def moving_block_bootstrap_ci(values: np.ndarray, *, block_length: int = 2,
                              n_bootstrap: int = 20_000, seed: int = 2026,
                              confidence: float = 0.95) -> tuple[float, float]:
    """Percentile interval for a mean using non-circular blocks over ordered days."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a nonempty finite vector")
    if block_length < 1 or block_length > len(values):
        raise ValueError("block_length must be between 1 and len(values)")
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be positive and confidence between 0 and 1")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(values) / block_length))
    offsets = np.arange(block_length)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        starts = rng.integers(0, len(values) - block_length + 1, size=n_blocks)
        indices = (starts[:, None] + offsets).ravel()[:len(values)]
        means[draw] = values[indices].mean()
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def build_inference(fold_metrics: pd.DataFrame, feature_shift: pd.DataFrame,
                    *, block_length: int = 2, n_bootstrap: int = 20_000,
                    seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"fold", "model", "macro_f1", "log_loss"}
    if not required.issubset(fold_metrics.columns):
        raise ValueError(f"fold_metrics is missing columns: {sorted(required - set(fold_metrics.columns))}")
    baseline_name = "training_class_prior"
    baseline = fold_metrics.loc[fold_metrics.model == baseline_name].set_index("fold")
    if baseline.empty:
        raise ValueError(f"Missing {baseline_name} baseline")
    rows: list[dict] = []
    models = sorted(set(fold_metrics.model) - {baseline_name})
    metrics = (("macro_f1", "higher_is_better"), ("log_loss", "lower_is_better"))
    for model in models:
        candidate = fold_metrics.loc[fold_metrics.model == model].set_index("fold")
        common = baseline.index.intersection(candidate.index).sort_values()
        if len(common) < 3:
            continue
        for metric, direction in metrics:
            raw_delta = candidate.loc[common, metric].to_numpy() - baseline.loc[common, metric].to_numpy()
            delta = raw_delta if direction == "higher_is_better" else -raw_delta
            low, high = moving_block_bootstrap_ci(
                delta, block_length=min(block_length, len(delta)), n_bootstrap=n_bootstrap,
                seed=seed + len(rows))
            rows.append({
                "model": model, "metric": metric, "interpretation": "positive_delta_favors_model",
                "test_days": int(len(common)), "block_length_days": min(block_length, len(common)),
                "mean_model": float(candidate.loc[common, metric].mean()),
                "mean_class_prior": float(baseline.loc[common, metric].mean()),
                "mean_delta_favoring_model": float(delta.mean()),
                "block_bootstrap_95_low": low, "block_bootstrap_95_high": high,
                "inference_note": "Descriptive interval over nine ordered folds; not a significance test.",
            })
    inference = pd.DataFrame(rows)

    merged = fold_metrics.merge(feature_shift[["fold", "mean_feature_psi"]], on="fold", how="left")
    shift_rows = []
    for model in models:
        part = merged.loc[merged.model == model]
        if len(part) >= 3:
            shift_rows.append({
                "model": model, "test_days": int(len(part)),
                "spearman_psi_macro_f1": (float(part["mean_feature_psi"].corr(part.macro_f1, method="spearman"))
                                           if part["mean_feature_psi"].nunique() > 1 and part.macro_f1.nunique() > 1 else np.nan),
                "spearman_psi_log_loss": (float(part["mean_feature_psi"].corr(part.log_loss, method="spearman"))
                                          if part["mean_feature_psi"].nunique() > 1 and part.log_loss.nunique() > 1 else np.nan),
                "note": "Descriptive rank association only; nine days do not support causal or stable-shift claims.",
            })
    return inference, pd.DataFrame(shift_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/market_microstructure")
    parser.add_argument("--block-length", type=int, default=2)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()
    metrics = pd.read_csv(args.results_dir / "fold_metrics.csv")
    shifts = pd.read_csv(args.results_dir / "feature_shift.csv")
    inference, shift_association = build_inference(
        metrics, shifts, block_length=args.block_length, n_bootstrap=args.bootstrap_draws)
    inference.to_csv(args.results_dir / "day_block_inference.csv", index=False, float_format="%.6f")
    shift_association.to_csv(args.results_dir / "shift_performance_association.csv", index=False,
                             float_format="%.6f")
    print(inference.to_string(index=False))
    print("\nDescriptive shift associations")
    print(shift_association.to_string(index=False))


if __name__ == "__main__":
    main()
