import numpy as np
import pytest

from ml_pipeline.validation import out_of_time_holdout_split, purged_walk_forward_splits


def _label_ends(n: int, horizon: int) -> np.ndarray:
    """Every sample's label depends on the next `horizon` bars."""
    ends = np.arange(n) + horizon
    ends[ends >= n] = -1  # no full forward window -> unlabelable
    return ends


def test_splits_are_chronological_walk_forward():
    n = 100
    label_end_idx = _label_ends(n, horizon=5)
    folds = list(purged_walk_forward_splits(n, label_end_idx, n_splits=4))
    assert len(folds) >= 1
    for fold in folds:
        assert fold.train_idx.max() < fold.test_idx.min()


def test_purge_removes_train_samples_whose_label_overlaps_test():
    n = 50
    horizon = 5
    label_end_idx = _label_ends(n, horizon=horizon)
    folds = list(purged_walk_forward_splits(n, label_end_idx, n_splits=3, embargo_bars=0))

    for fold in folds:
        test_start = fold.test_idx.min()
        # No surviving training sample's label window may reach into the
        # test fold — that would leak the test future into training.
        ends = label_end_idx[fold.train_idx]
        assert (ends < test_start).all()


def test_embargo_shrinks_train_near_boundary():
    n = 60
    horizon = 3
    label_end_idx = _label_ends(n, horizon=horizon)

    folds_no_embargo = list(purged_walk_forward_splits(n, label_end_idx, n_splits=3, embargo_bars=0))
    folds_embargo = list(purged_walk_forward_splits(n, label_end_idx, n_splits=3, embargo_bars=5))

    for f0, f1 in zip(folds_no_embargo, folds_embargo):
        assert len(f1.train_idx) <= len(f0.train_idx)
        if len(f1.train_idx) > 0:
            assert f1.train_idx.max() <= f0.train_idx.max()


def test_unlabelable_samples_excluded_from_both_train_and_test():
    n = 30
    label_end_idx = _label_ends(n, horizon=5)  # last 5 samples are -1
    folds = list(purged_walk_forward_splits(n, label_end_idx, n_splits=3))
    all_used = np.concatenate([np.concatenate([f.train_idx, f.test_idx]) for f in folds])
    invalid = np.flatnonzero(label_end_idx < 0)
    assert not np.isin(invalid, all_used).any()


def test_holdout_is_strictly_after_tuning_set_and_purged():
    n = 100
    horizon = 4
    label_end_idx = _label_ends(n, horizon=horizon)
    fold = out_of_time_holdout_split(n, label_end_idx, holdout_frac=0.2, embargo_bars=2)

    assert fold.train_idx.max() < fold.test_idx.min()
    ends = label_end_idx[fold.train_idx]
    assert (ends < fold.test_idx.min()).all()


def test_holdout_raises_when_too_small():
    n = 5
    label_end_idx = np.array([-1, -1, -1, -1, -1])
    with pytest.raises(ValueError):
        out_of_time_holdout_split(n, label_end_idx, holdout_frac=0.5)
