import numpy as np
import pandas as pd
import pytest

from src.microstructure_inference import build_inference, moving_block_bootstrap_ci


def test_block_bootstrap_interval_is_deterministic_and_contains_mean():
    values = np.array([1.0, 2.0, 3.0, 2.0, 4.0, 3.0])
    first = moving_block_bootstrap_ci(values, block_length=2, n_bootstrap=2_000, seed=9)
    second = moving_block_bootstrap_ci(values, block_length=2, n_bootstrap=2_000, seed=9)
    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_block_bootstrap_rejects_invalid_input():
    with pytest.raises(ValueError):
        moving_block_bootstrap_ci(np.array([]))
    with pytest.raises(ValueError):
        moving_block_bootstrap_ci(np.array([1.0, 2.0]), block_length=3)


def test_inference_uses_paired_ordered_day_deltas_and_shift_associations():
    rows = []
    for fold in range(1, 10):
        rows.extend([
            {"fold": fold, "model": "training_class_prior", "macro_f1": 0.2, "log_loss": 1.1},
            {"fold": fold, "model": "candidate", "macro_f1": 0.3 + fold / 100, "log_loss": 1.0},
        ])
    metrics = pd.DataFrame(rows)
    shifts = pd.DataFrame({"fold": range(1, 10), "mean_feature_psi": np.arange(1, 10)})
    inference, association = build_inference(metrics, shifts, n_bootstrap=1_000)
    assert set(inference.metric) == {"macro_f1", "log_loss"}
    assert (inference.mean_delta_favoring_model > 0).all()
    assert (inference.test_days == 9).all()
    assert association.iloc[0].spearman_psi_macro_f1 == pytest.approx(1.0)
