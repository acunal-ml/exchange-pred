import numpy as np
import pandas as pd

from data_pipeline.feature_engine import (
    atr,
    bollinger_bands,
    compute_features,
    drop_warmup,
    macd,
    moving_average_ratios,
    price_channel_position,
    returns,
    rsi,
    volatility_regime,
    volume_zscore,
)


def _synthetic_ohlc(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1e5, 1e6, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


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
    assert len(clean) > 0  # regression: a missing/NaN volume feature must not drop every row
    assert clean.isna().sum().sum() == 0
    assert len(clean) < len(feats)


def test_returns_is_pct_change_over_each_period():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    out = returns(close, periods=(1, 2))
    assert np.isclose(out["ret_1"].iloc[1], 0.10)
    assert np.isclose(out["ret_2"].iloc[2], 0.21)


def test_moving_average_ratio_zero_when_price_equals_sma():
    close = pd.Series([100.0] * 25)
    out = moving_average_ratios(close, windows=(20,))
    assert np.isclose(out["close_sma20_ratio"].iloc[-1], 0.0)


def test_volume_zscore_zero_for_constant_volume():
    volume = pd.Series([1000.0] * 30)
    z = volume_zscore(volume, window=20)
    assert np.isclose(z.iloc[-1], 0.0)  # std==0 -> defined as 0, not NaN/inf


def test_volatility_regime_above_one_when_recent_vol_higher():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.001, 60)
    choppy = rng.normal(0, 0.05, 10)
    close = pd.Series(100 * np.cumprod(1 + np.concatenate([calm, choppy])))
    regime = volatility_regime(close, short=10, long=50)
    assert regime.iloc[-1] > 1.0


def test_price_channel_position_near_one_for_a_strict_uptrend():
    n = 70
    close = pd.Series(np.linspace(100, 200, n))  # strictly rising -> the latest bar is its own window's high
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    pos = price_channel_position(df, window=60)
    assert pos.iloc[-1] > 0.9
    assert pos.iloc[-1] <= 1.0


def test_compute_features_includes_all_new_normalized_columns():
    df = _synthetic_ohlc(n=150)
    feats = drop_warmup(compute_features(df))
    for col in (
        "macd_norm",
        "macd_signal_norm",
        "macd_hist_norm",
        "bb_upper_ratio",
        "bb_lower_ratio",
        "atr_pct",
        "ret_5",
        "ret_10",
        "ret_20",
        "close_sma20_ratio",
        "close_sma50_ratio",
        "volume_zscore_20",
        "vol_regime",
        "price_channel_pos",
    ):
        assert col in feats.columns
        assert feats[col].notna().all()
        assert np.isfinite(feats[col]).all()
