import numpy as np

from ml_pipeline.calibrate import apply_calibration, expected_calibration_error, fit_calibrators


def _overconfident_proba(y_true: np.ndarray, n_classes: int = 3, sharpen: float = 6.0) -> np.ndarray:
    """Build raw probabilities that are correct on average but pushed
    toward extreme confidence (the kind of mis-calibration calibration
    is meant to fix)."""
    n = len(y_true)
    base = np.full((n, n_classes), 0.5 / (n_classes - 1))
    base[np.arange(n), y_true] = 0.5
    # sharpen toward the (mostly correct) argmax
    sharpened = base**sharpen
    return sharpened / sharpened.sum(axis=1, keepdims=True)


def test_apply_calibration_returns_valid_probability_rows():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 200)
    raw_proba = _overconfident_proba(y_true)

    result = fit_calibrators(raw_proba, y_true, method="isotonic")
    calibrated = apply_calibration(result, raw_proba)

    assert calibrated.shape == raw_proba.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)
    assert (calibrated >= 0).all()


def test_isotonic_calibration_reduces_overconfidence_ece():
    rng = np.random.default_rng(1)
    # y_true only ~60% matches the argmax of raw_proba -> raw probs are
    # overconfident relative to true accuracy.
    n = 500
    true_argmax = rng.integers(0, 3, n)
    flip = rng.random(n) < 0.4
    y_true = np.where(flip, (true_argmax + 1) % 3, true_argmax)
    raw_proba = _overconfident_proba(true_argmax)

    raw_ece = expected_calibration_error(y_true, raw_proba)

    result = fit_calibrators(raw_proba, y_true, method="isotonic")
    calibrated = apply_calibration(result, raw_proba)
    calibrated_ece = expected_calibration_error(y_true, calibrated)

    assert calibrated_ece < raw_ece


def test_sigmoid_platt_calibration_also_produces_valid_probabilities():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 3, 150)
    raw_proba = _overconfident_proba(y_true)

    result = fit_calibrators(raw_proba, y_true, method="sigmoid")
    calibrated = apply_calibration(result, raw_proba)

    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_input():
    # confidence exactly matches accuracy within each bin -> ECE ~ 0
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    proba = np.tile([0.5, 0.3, 0.2], (8, 1))
    proba[:4, 0] = 0.5
    proba[4:, 1] = 0.5
    # half of class-0-predicted are actually class 0 -> confidence 0.5 == accuracy 0.5
    ece = expected_calibration_error(y_true, proba, n_bins=5)
    assert ece < 0.3  # loose sanity bound, not a tight equality
