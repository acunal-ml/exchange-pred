import numpy as np
import pandas as pd

from data_pipeline.feature_engine import atr, bollinger_bands, compute_features, drop_warmup, macd, rsi


def _synthetic_ohlc(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_rsi_bounded_0_100():
    df = _synthetic_ohlc()
    values = rsi(df["close"]).dropna()
    assert (values >= 0).all() and (values <= 100).all()


def test_macd_has_no_lookahead_in_warmup():
    df = _synthetic_ohlc()
    out = macd(df["close"])
    # slow EMA (span=26) warm-up: first 25 rows must be NaN, never imputed.
    assert out["macd"].iloc[:25].isna().all()


def test_bollinger_percent_b_relationship():
    df = _synthetic_ohlc()
    bb = bollinger_bands(df["close"])
    valid = bb.dropna()
    # close at upper band -> %B ~ 1, at lower band -> %B ~ 0 (sanity bounds only)
    assert valid["bb_percent_b"].between(-3, 4).all()  # loose bound, no hard clipping expected


def test_atr_non_negative():
    df = _synthetic_ohlc()
    values = atr(df).dropna()
    assert (values >= 0).all()


def test_compute_features_warmup_dropped_not_imputed():
    df = _synthetic_ohlc()
    feats = compute_features(df)
    assert feats["rsi_14"].iloc[:13].isna().all()  # warm-up still NaN
    clean = drop_warmup(feats)
    assert clean.isna().sum().sum() == 0
    assert len(clean) < len(feats)
