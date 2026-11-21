"""Walk-forward FI-2010 limit-order-book classification experiment.

The official benchmark provides 144 precomputed features and five labels. This
module evaluates the 10-event label on all nine anchored day folds.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, log_loss, precision_score,
                             recall_score)
from xgboost import XGBClassifier


DATASET_ID = "73eb48d7-4dbc-4a10-a52a-da745b47a649"
EXPECTED_PACKAGE_SHA256 = "bcc89a5aa7d8067dda98374393444eb885a4283a41fd33e323496380e057e1a6"
CLASS_LABELS = np.array([1, 2, 3], dtype=np.int64)  # up, stationary, down in source label convention
HORIZON_ROW = 148  # rows 145..149 correspond to event horizons 1, 2, 3, 5, 10


def extract_benchmark(archive: Path, output_dir: Path) -> None:
    """Extract only the train/test files needed for anchored folds 1 through 9."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for fold in range(1, 10):
            names = (
                f"BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_{fold}.txt",
                f"BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_{fold}.txt",
            )
            for name in names:
                target = output_dir / Path(name).name
                if target.exists() and target.stat().st_size == zf.getinfo(name).file_size:
                    continue
                with zf.open(name) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)


def load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an official 149-by-N text matrix and return X[N,144], y[N]."""
    values = np.loadtxt(path, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 149:
        raise ValueError(f"Expected 149 rows in {path}; got {values.shape}")
    x = np.ascontiguousarray(values[:144].T)
    y = values[HORIZON_ROW].astype(np.int64)
    if not np.isfinite(x).all() or not np.isin(y, CLASS_LABELS).all():
        raise ValueError(f"Non-finite features or unknown labels in {path}")
    return x, y


def multiclass_brier(y: np.ndarray, probability: np.ndarray) -> float:
    truth = np.column_stack([y == c for c in CLASS_LABELS]).astype(np.float64)
    return float(np.mean(np.sum((probability - truth) ** 2, axis=1)))


def evaluate(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, dtype=np.float64)
    probability = probability / probability.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, labels=CLASS_LABELS, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probability, labels=CLASS_LABELS)),
        "multiclass_brier": multiclass_brier(y, probability),
    }


def _probability_in_class_order(model, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    return raw[:, [list(model.classes_).index(c) for c in CLASS_LABELS]]


def feature_psi_by_fold(x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Return train-binned PSI for each feature on a held-out fold."""
    values = []
    for feature_idx in range(x_train.shape[1]):
        train_feature = x_train[:, feature_idx]
        test_feature = x_test[:, feature_idx]
        edges = np.unique(np.quantile(train_feature, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        train_share = np.histogram(train_feature, bins=edges)[0].astype(float)
        test_share = np.histogram(test_feature, bins=edges)[0].astype(float)
        train_share = (train_share + 1e-6) / (train_share.sum() + 1e-6 * len(train_share))
        test_share = (test_share + 1e-6) / (test_share.sum() + 1e-6 * len(test_share))
        values.append(float(np.sum((test_share - train_share) * np.log(test_share / train_share))))
    return np.asarray(values)


def run_experiment(data_dir: Path, results_dir: Path, *, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    shifts: list[dict] = []
    elapsed: dict[str, float] = {}

    for fold in range(1, 10):
        print(f"Loading anchored fold {fold}/9 (test day {fold + 1})", flush=True)
        train_path = data_dir / f"Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
        test_path = data_dir / f"Test_Dst_NoAuction_ZScore_CF_{fold}.txt"
        x_train, y_train = load_matrix(train_path)
        x_test, y_test = load_matrix(test_path)
        # PSI uses only train-derived decile boundaries, then compares the
        # held-out day's feature marginals with the training distribution.
        psi_values = feature_psi_by_fold(x_train, x_test)
        shifts.append({"fold": fold, "test_day": fold + 1, "mean_feature_psi": float(np.mean(psi_values)),
                       "median_feature_psi": float(np.median(psi_values)), "p90_feature_psi": float(np.quantile(psi_values, 0.90)),
                       "max_feature_psi": float(np.max(psi_values))})
        print(f"  train={len(y_train):,}, test={len(y_test):,}, mean feature PSI={shifts[-1]['mean_feature_psi']:.4f}", flush=True)
        counts = np.array([(y_train == c).sum() for c in CLASS_LABELS], dtype=np.float64)
        prior = counts / counts.sum()
        majority = np.full(y_test.shape, CLASS_LABELS[np.argmax(counts)])
        prior_prob = np.tile(prior, (len(y_test), 1))
        base = evaluate(y_test, majority, prior_prob)
        rows.append({"fold": fold, "test_day": fold + 1, "model": "training_class_prior", "n_train": len(y_train), "n_test": len(y_test), **base})
        pred_frame = pd.DataFrame({"fold": fold, "test_day": fold + 1, "y_true": y_test, "training_prior_pred": majority,
                                   "p_prior_up": prior_prob[:, 0], "p_prior_stationary": prior_prob[:, 1], "p_prior_down": prior_prob[:, 2]})

        models = {
            "logistic_regression": LogisticRegression(C=0.1, max_iter=250, solver="lbfgs", random_state=seed),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=18, min_samples_leaf=30, max_features="sqrt",
                class_weight="balanced_subsample", max_samples=0.5, n_jobs=4,
                random_state=seed,
            ),
            "xgboost": XGBClassifier(
                objective="multi:softprob", num_class=3, n_estimators=240, max_depth=5,
                learning_rate=0.06, min_child_weight=20, subsample=0.85,
                colsample_bytree=0.85, reg_lambda=5.0, tree_method="hist",
                n_jobs=4, random_state=seed, eval_metric="mlogloss",
            ),
        }
        for model_name, model in models.items():
            start = time.perf_counter()
            model_y = y_train - 1 if model_name == "xgboost" else y_train
            model.fit(x_train, model_y)
            if model_name == "xgboost":
                # XGBoost requires contiguous class ids 0..2; they preserve
                # source class order after subtracting one.
                probability = model.predict_proba(x_test)
            else:
                probability = _probability_in_class_order(model, x_test)
            pred = CLASS_LABELS[probability.argmax(axis=1)]
            elapsed[f"fold_{fold}_{model_name}"] = time.perf_counter() - start
            metrics = evaluate(y_test, pred, probability)
            rows.append({"fold": fold, "test_day": fold + 1, "model": model_name, "n_train": len(y_train), "n_test": len(y_test), **metrics})
            pred_frame[f"{model_name}_pred"] = pred
            for col, class_name in enumerate(("up", "stationary", "down")):
                pred_frame[f"{model_name}_p_{class_name}"] = probability[:, col]
        predictions.append(pred_frame)
        del x_train, y_train, x_test, y_test, models

    metrics_frame = pd.DataFrame(rows)
    pred_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame.to_csv(results_dir / "fold_metrics.csv", index=False)
    pred_frame.to_csv(results_dir / "test_predictions.csv.gz", index=False, compression="gzip")
    metrics_frame.groupby("model", as_index=False).agg(
        fold_count=("fold", "nunique"),
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_log_loss=("log_loss", "mean"),
        mean_multiclass_brier=("multiclass_brier", "mean"),
    ).to_csv(results_dir / "summary_metrics.csv", index=False)

    test_dist = pred_frame.groupby("test_day")["y_true"].value_counts(normalize=True).unstack(fill_value=0)
    test_dist = test_dist.reindex(columns=CLASS_LABELS, fill_value=0)
    test_dist.columns = ["share_up", "share_stationary", "share_down"]
    test_dist.to_csv(results_dir / "test_day_class_distribution.csv")
    pd.DataFrame(shifts).to_csv(results_dir / "feature_shift.csv", index=False)
    metadata = {
        "dataset": "FI-2010 official Fairdata benchmark",
        "dataset_id": DATASET_ID,
        "license": "CC BY 4.0",
        "package_sha256": EXPECTED_PACKAGE_SHA256,
        "data_variant": "NoAuction, ZScore",
        "features": 144,
        "label_row_zero_based": HORIZON_ROW,
        "horizon_events": 10,
        "label_mapping": {"1": "up", "2": "stationary", "3": "down"},
        "folds": {str(k): {"training_file": f"Train_Dst_NoAuction_ZScore_CF_{k}.txt", "test_file": f"Test_Dst_NoAuction_ZScore_CF_{k}.txt", "test_day": k + 1} for k in range(1, 10)},
        "models": {
            "logistic_regression": "C=0.1, lbfgs, max_iter=250",
            "random_forest": "100 trees, max_depth=18, min_samples_leaf=30, max_features=sqrt, max_samples=0.5",
            "xgboost": "240 trees, depth 5, learning_rate 0.06, fixed hyperparameters",
        },
        "training_seconds": elapsed,
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics_frame, pred_frame


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    source_archive = root / "data/market_microstructure/raw/BenchmarkDatasets.zip"
    data_dir = root / "data/market_microstructure/processed"
    if not all((data_dir / f"Train_Dst_NoAuction_ZScore_CF_{k}.txt").exists() for k in range(1, 10)):
        extract_benchmark(source_archive, data_dir)
    metrics, _ = run_experiment(data_dir, root / "results/market_microstructure")
    print(metrics.to_string(index=False))
