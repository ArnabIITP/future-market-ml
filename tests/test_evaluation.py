import numpy as np

from src.evaluation import _expected_calibration_error, walk_forward_splits


def test_walk_forward_splits_are_ordered_and_non_overlapping():
    folds = list(walk_forward_splits(n=11, min_train_size=5, test_size=3))
    assert [test.tolist() for _, test in folds] == [[5, 6, 7], [8, 9, 10]]
    assert all(train.max() < test.min() for train, test in folds)


def test_calibration_error_is_zero_for_perfectly_calibrated_events():
    truth = np.array([0, 0, 1, 1])
    probability = np.array([0.0, 0.0, 1.0, 1.0])
    assert _expected_calibration_error(truth, probability) == 0
