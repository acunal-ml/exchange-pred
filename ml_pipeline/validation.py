"""Purged + embargoed walk-forward cross-validation (López de Prado style).

docs/03 calls this "the single most important fix" for this project:
financial time series must never use random KFold, shuffling, or a
default train_test_split — that leaks the future into the past and
produces backtests that look excellent and fail live.

Two leakage vectors are handled here:
- **Purge**: a training sample at index i is only "safe" if its label's
  forward window [i, label_end_idx[i]] does not overlap the test fold's
  index range. Overlapping samples are dropped from train.
- **Embargo**: even after purging, samples immediately after a test fold
  can be serially correlated with it (autocorrelation in returns/vol).
  Drop an additional `embargo_bars` of training samples right after
  each test fold's end.

Folds are walk-forward: fold k's test range is strictly after fold k-1's,
so chronological order is always respected.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def purged_walk_forward_splits(
    n_samples: int,
    label_end_idx: np.ndarray,
    n_splits: int = 5,
    embargo_bars: int = 0,
    min_train_size: int = 1,
) -> Iterator[Fold]:
    """Yield walk-forward folds with purging and embargo applied.

    `label_end_idx[i]` must be the index of the last bar each sample's
    label depends on (as returned by
    data_pipeline.labeling.triple_barrier_labels). Samples with no valid
    label (label_end_idx == -1) are excluded entirely — they carry no
    forward-window information to purge against and shouldn't be trained
    or tested on.

    Test folds are contiguous, equal-sized, non-overlapping chronological
    blocks. Each fold's training set is every other valid sample *before*
    the test fold's end, minus anything purged or embargoed.
    """
    if n_samples != len(label_end_idx):
        raise ValueError("label_end_idx must have one entry per sample")

    valid_mask = label_end_idx >= 0
    valid_idx = np.flatnonzero(valid_mask)
    if len(valid_idx) < n_splits:
        raise ValueError("Not enough labeled samples for the requested number of splits")

    fold_boundaries = np.array_split(valid_idx, n_splits)

    for k, test_idx in enumerate(fold_boundaries):
        if len(test_idx) == 0:
            continue

        test_start = test_idx[0]

        # Walk-forward: only samples chronologically before this fold are
        # eligible for training (no test-into-future leakage).
        candidate_train = valid_idx[valid_idx < test_start]
        if len(candidate_train) == 0:
            continue

        # Purge: drop any candidate whose label window [i, label_end_idx[i]]
        # overlaps the test fold's index range [test_start, test_end].
        candidate_ends = label_end_idx[candidate_train]
        overlaps_test = candidate_ends >= test_start
        purged = candidate_train[~overlaps_test]

        # Embargo: additionally drop the embargo_bars training samples
        # immediately adjacent to the test boundary. This guards against
        # feature-side serial correlation (e.g. overlapping rolling
        # windows) right at the train/test edge that label-purging alone
        # doesn't catch.
        embargo_boundary = test_start - embargo_bars
        train_idx = purged[purged < embargo_boundary]

        if len(train_idx) < min_train_size:
            continue

        yield Fold(train_idx=train_idx, test_idx=test_idx)


def out_of_time_holdout_split(
    n_samples: int,
    label_end_idx: np.ndarray,
    holdout_frac: float = 0.15,
    embargo_bars: int = 0,
) -> Fold:
    """Carve a final out-of-time holdout, embargoed from the tuning set.

    This holdout must never be touched during hyperparameter tuning or
    model selection (docs/03) — it's reserved to feed the UI backtest
    panel and the final champion-vs-baseline comparison.
    """
    valid_idx = np.flatnonzero(label_end_idx >= 0)
    split_point = int(len(valid_idx) * (1 - holdout_frac))
    holdout_idx = valid_idx[split_point:]
    if len(holdout_idx) == 0:
        raise ValueError("holdout_frac too small for the available samples")

    holdout_start = holdout_idx[0]
    tuning_candidates = valid_idx[valid_idx < holdout_start]

    candidate_ends = label_end_idx[tuning_candidates]
    purged = tuning_candidates[candidate_ends < holdout_start]

    embargo_cutoff = holdout_start - embargo_bars
    tuning_idx = purged[purged < embargo_cutoff]

    return Fold(train_idx=tuning_idx, test_idx=holdout_idx)
