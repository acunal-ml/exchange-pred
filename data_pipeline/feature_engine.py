"""Feature engineering: RSI, MACD, Bollinger Bands, ATR.

docs/02 originally called for TA-Lib/pandas-ta. Both are rejected here:
TA-Lib needs a compiled C extension that is awkward on HF Spaces, and
pandas-ta's pinned numba dependency caps numpy<2.3, conflicting with the
rest of this project's stack and itself barely maintained. Per docs'
explicit instruction to "decide one canonical implementation per
indicator to avoid train/serve skew", these four indicators are
implemented directly here in vectorized pandas/numpy — one code path,
no extra dependency, identical at train and serve time.

All functions operate on a DataFrame indexed by UTC timestamp with at
least open/high/low/close columns. Indicator warm-up rows (leading NaN
from rolling windows) are left as NaN — callers must drop them, never
impute, per the "Indicator warm-up" rule in docs/02 (imputing injects
look-ahead).
"""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing == an EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out.where(avg_loss != 0, 100.0)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram})


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    # %B: where close sits within the band, 0 = lower band, 1 = upper band.
    percent_b = (close - lower) / (upper - lower)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_percent_b": percent_b})


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def returns(close: pd.Series, periods: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    """Lagged momentum. Already scale-invariant (a % change), unlike a raw
    price level — usable across assets/regimes without renormalizing."""
    return pd.DataFrame({f"ret_{p}": close.pct_change(p) for p in periods})


def moving_average_ratios(close: pd.Series, windows: tuple[int, ...] = (20, 50)) -> pd.DataFrame:
    """close/SMA - 1: how far price sits above/below its trend, as a
    fraction — stationary across price levels, unlike the SMA itself."""
    out = {}
    for w in windows:
        sma = close.rolling(window=w, min_periods=w).mean()
        out[f"close_sma{w}_ratio"] = close / sma - 1.0
    return pd.DataFrame(out)


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """How unusual today's volume is relative to its own recent history —
    a volume spike/drought, not the (asset-dependent, non-stationary)
    absolute share count."""
    mean = volume.rolling(window=window, min_periods=window).mean()
    std = volume.rolling(window=window, min_periods=window).std(ddof=0)
    z = (volume - mean) / std
    return z.where(std != 0, 0.0)


def volatility_regime(close: pd.Series, short: int = 10, long: int = 50) -> pd.Series:
    """Ratio of short- to long-run realized volatility of returns: >1
    means the market has gotten choppier recently than its own baseline
    — a regime signal triple-barrier width alone doesn't capture."""
    ret = close.pct_change()
    vol_short = ret.rolling(window=short, min_periods=short).std(ddof=0)
    vol_long = ret.rolling(window=long, min_periods=long).std(ddof=0)
    return vol_short / vol_long


def price_channel_position(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """Where close sits within its own N-bar high/low range: 0 = at the
    range low, 1 = at the range high — a longer-horizon analogue of
    Bollinger %B, using the actual trading range rather than a
    std-deviation band."""
    rolling_high = df["high"].rolling(window=window, min_periods=window).max()
    rolling_low = df["low"].rolling(window=window, min_periods=window).min()
    return (df["close"] - rolling_low) / (rolling_high - rolling_low)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all canonical features on closed candles only.

    Caller is responsible for ensuring the input contains only closed
    candles (i.e. drop or exclude the still-forming current bar before
    calling this) — see docs/01 execution flow and docs/02 data hygiene.

    Most ML-facing features here are deliberately normalized to be
    scale-invariant (ratios, %, z-scores) rather than the raw indicator
    levels — a raw price-unit feature (e.g. MACD's line, in $) drifts
    with the asset's price level over time and doesn't generalize across
    assets, which is bad practice for a model meant to stay valid as
    price drifts away from its training range. `atr_14` is the one
    exception: it's kept in raw price units because data_pipeline.labeling
    and inference.signal_aggregator.compute_levels both do entry +/- k*ATR
    arithmetic directly in price space — `atr_pct` below is the
    ML-feature-friendly version of the same information.
    """
    out = df.copy()
    out["rsi_14"] = rsi(out["close"])

    macd_df = macd(out["close"])
    out = out.join(macd_df)

    bb_df = bollinger_bands(out["close"])
    out = out.join(bb_df)

    out["atr_14"] = atr(out)

    # Normalized re-expressions of the above, for ML use (see docstring).
    out["macd_norm"] = out["macd"] / out["atr_14"]
    out["macd_signal_norm"] = out["macd_signal"] / out["atr_14"]
    out["macd_hist_norm"] = out["macd_hist"] / out["atr_14"]
    out["bb_upper_ratio"] = out["bb_upper"] / out["close"] - 1.0
    out["bb_lower_ratio"] = out["bb_lower"] / out["close"] - 1.0
    out["atr_pct"] = out["atr_14"] / out["close"]

    out = out.join(returns(out["close"]))
    out = out.join(moving_average_ratios(out["close"]))
    # "volume" is a required input column (every real OHLCV source in this
    # project provides it) — silently NaN-ing this feature when it's
    # missing would make drop_warmup() discard every row (volume_zscore_20
    # would be NaN everywhere), corrupting the whole pipeline instead of
    # failing loudly at the actual problem (bad input data).
    out["volume_zscore_20"] = volume_zscore(out["volume"])
    out["vol_regime"] = volatility_regime(out["close"])
    out["price_channel_pos"] = price_channel_position(out)

    return out


def drop_warmup(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Drop leading rows where rolling indicators are still NaN.

    Never forward-fill or impute these — that injects look-ahead bias.
    """
    cols = feature_columns or [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    return df.dropna(subset=cols)
