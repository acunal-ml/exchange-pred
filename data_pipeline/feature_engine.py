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


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all canonical features on closed candles only.

    Caller is responsible for ensuring the input contains only closed
    candles (i.e. drop or exclude the still-forming current bar before
    calling this) — see docs/01 execution flow and docs/02 data hygiene.
    """
    out = df.copy()
    out["rsi_14"] = rsi(out["close"])

    macd_df = macd(out["close"])
    out = out.join(macd_df)

    bb_df = bollinger_bands(out["close"])
    out = out.join(bb_df)

    out["atr_14"] = atr(out)

    return out


def drop_warmup(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Drop leading rows where rolling indicators are still NaN.

    Never forward-fill or impute these — that injects look-ahead bias.
    """
    cols = feature_columns or [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    return df.dropna(subset=cols)
