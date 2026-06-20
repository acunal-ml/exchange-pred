"""Probability calibration: isotonic or Platt (sigmoid) scaling.

docs/03: "Raw LightGBM/LSTM probabilities are usually over/under-confident.
Calibrate ... on a held-out fold so the Tab-2 confidence threshold has
real meaning." The signal aggregator (docs/01) decides
`label = argmax(P) if max(P) >= threshold else Hold` — that rule is only
meaningful if P's magnitude actually reflects empirical correctness
likelihood, which raw tree/NN outputs generally don't.

This calibrates per-class via one-vs-rest (the same approach
scikit-learn's CalibratedClassifierCV uses internally for multiclass),
fitting one calibrator per class on a held-out calibration set, then
renormalizing the resulting per-class scores back into a probability
simplex. Implemented directly on raw probability arrays (not wrapped
around a specific model's Estimator interface) so the exact same code
calibrates both the LightGBM and the LSTM champion — neither has to be
coerced into the other's API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from utils.logging_config import get_logger

logger = get_logger(__name__)

CalibrationMethod = Literal["isotonic", "sigmoid"]

# Isotonic regression is non-parametric (a free step function per class) and
# overfits badly on small calibration sets — confirmed via real-data
# smoke testing, where isotonic on ~65 AAPL calibration samples made ECE
# *worse* (0.065 -> 0.41) than leaving probabilities raw. Sigmoid/Platt
# scaling fits only 2 parameters per class, so it degrades far more
# gracefully when calibration data is scarce. Below this many samples,
# fit_calibrators silently downgrades isotonic -> sigmoid.
MIN_ISOTONIC_SAMPLES = 200


@dataclass
class CalibrationResult:
    calibrators: list
    method: CalibrationMethod


def fit_calibrators(raw_proba: np.ndarray, y_true: np.ndarray, method: CalibrationMethod = "isotonic") -> CalibrationResult:
    """Fit one calibrator per class on a held-out calibration set.

    `raw_proba` must come from a fold the model never trained on (its
    own held-out calibration slice, distinct from both the CV tuning
    folds and the final out-of-time reporting holdout — see
    train_lightgbm.run_training / train_lstm.run_training for where
    that slice is carved).
    """
    if method == "isotonic" and len(y_true) < MIN_ISOTONIC_SAMPLES:
        logger.warning(
            "Only %d calibration samples (<%d) — isotonic regression overfits at this size; "
            "falling back to sigmoid/Platt scaling.",
            len(y_true),
            MIN_ISOTONIC_SAMPLES,
        )
        method = "sigmoid"

    n_classes = raw_proba.shape[1]
    calibrators = []
    for class_id in range(n_classes):
        binary_target = (y_true == class_id).astype(int)
        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(raw_proba[:, class_id], binary_target)
        elif method == "sigmoid":
            calibrator = LogisticRegression()
            calibrator.fit(raw_proba[:, class_id].reshape(-1, 1), binary_target)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        calibrators.append(calibrator)
    return CalibrationResult(calibrators=calibrators, method=method)


def apply_calibration(result: CalibrationResult, raw_proba: np.ndarray) -> np.ndarray:
    """Transform raw per-class scores through their fitted calibrators,
    then renormalize rows back to a valid probability distribution
    (calibrating each class independently can break the sum-to-1
    constraint)."""
    calibrated = np.zeros_like(raw_proba, dtype=float)
    for class_id, calibrator in enumerate(result.calibrators):
        column = raw_proba[:, class_id]
        if result.method == "isotonic":
            calibrated[:, class_id] = calibrator.predict(column)
        else:
            calibrated[:, class_id] = calibrator.predict_proba(column.reshape(-1, 1))[:, 1]

    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return calibrated / row_sums


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: bin samples by predicted confidence (max class
    probability), compare each bin's mean confidence to its actual
    accuracy. Used to quantify whether calibration helped — log both
    the raw and calibrated ECE so a regression is visible, not assumed."""
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bin_edges) - 1, 0, n_bins - 1)

    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)
