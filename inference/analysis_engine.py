"""The single Analysis Engine (docs/01 DRY requirement): every timeframe
and both UI tabs go through this one code path — no per-timeframe
branching of business logic.

    1. Resolve cache -> DB -> API data fetch (docs/02 lookup order).
    2. Compute features on closed candles only (no look-ahead).
    3. Run inference (ONNX champions via inference.model_loader).
    4. Call the Signal Aggregation Engine.
    5. Return the standardized SignalResult.

This module never trains and never touches mlflow — it only consumes
artifacts ml_pipeline/export_onnx.py already produced, via
inference/model_loader.py. That boundary is deliberate (docs/03).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from core.cache import cache
from core.db_setup import fetch_ohlcv, upsert_ohlcv_rows
from data_pipeline.feature_engine import compute_features, drop_warmup
from data_pipeline.sources.base import to_ohlcv_rows
from data_pipeline.sources.ingest_tvdatafeed import TVDatafeedSource
from data_pipeline.sources.ingest_yfinance import YFinanceSource
from inference.model_loader import ModelBundle, predict_proba
from inference.signal_aggregator import SignalResult, combine_indicator_votes, fuse_signals
from utils.logging_config import get_logger, utc_now

logger = get_logger(__name__)

# Approximate candle duration per timeframe — used only to decide
# whether the most recent fetched bar has actually closed yet (1M is a
# deliberately loose approximation; precision beyond "is this bar still
# forming" doesn't matter here).
TIMEFRAME_DURATIONS = {
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1H": timedelta(hours=1),
    "4H": timedelta(hours=4),
    "1D": timedelta(days=1),
    "1W": timedelta(weeks=1),
    "1M": timedelta(days=30),
}

CACHE_TTL_BY_TIMEFRAME = {
    "5m": 60,
    "15m": 300,
    "1H": 900,
    "4H": 1800,
    "1D": 3600,
    "1W": 86400,
    "1M": 86400,
}


def _serialize_df(df: pd.DataFrame) -> list[dict]:
    out = df.reset_index()
    out.columns = ["timestamp", *out.columns[1:]]
    out["timestamp"] = out["timestamp"].apply(lambda ts: ts.isoformat())
    return out.to_dict(orient="records")


def _deserialize_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def drop_unclosed_candle(df: pd.DataFrame, timeframe: str, now: datetime | None = None) -> pd.DataFrame:
    """Drop the trailing bar if it hasn't closed yet — features must
    never be computed on a still-forming candle (look-ahead)."""
    if df.empty:
        return df
    now = now or utc_now()
    duration = TIMEFRAME_DURATIONS[timeframe]
    last_close_time = df.index[-1] + duration
    if now < last_close_time:
        return df.iloc[:-1]
    return df


def fetch_ohlcv_cached(
    symbol: str,
    market: str,
    timeframe: str,
    lookback_days: int,
) -> pd.DataFrame:
    """Cache -> SQLite -> API lookup order (docs/02)."""
    cache_key = f"ohlcv:{market}:{symbol}:{timeframe}"
    cached = cache.get(cache_key)
    if cached is not None:
        return _deserialize_df(cached)

    end = utc_now()
    start = end - timedelta(days=lookback_days)

    rows = fetch_ohlcv(symbol, timeframe, int(start.timestamp()), int(end.timestamp()))
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
    else:
        df = pd.DataFrame()

    is_stale = df.empty or (end - df.index[-1]) > TIMEFRAME_DURATIONS[timeframe] * 2
    if is_stale:
        source = YFinanceSource() if market in ("NASDAQ", "COMMODITY") else TVDatafeedSource()
        fetched = source.fetch_ohlcv(symbol, timeframe, start, end)
        upsert_ohlcv_rows(to_ohlcv_rows(fetched, asset_id=f"{market}:{symbol}", timeframe=timeframe))
        df = fetched

    cache.set(cache_key, _serialize_df(df), ttl_seconds=CACHE_TTL_BY_TIMEFRAME[timeframe])
    return df


def analyze(
    symbol: str,
    market: str,
    timeframe: str,
    indicator_weights: dict[str, float] | None = None,
    w_ind: float = 1.0,
    w_lgbm: float = 1.0,
    w_lstm: float = 1.0,
    confidence_threshold: float = 0.4,
    horizon_bucket: str = "medium",
    lookback_days: int = 365,
    lgbm_bundle: ModelBundle | None = None,
    lstm_bundle: ModelBundle | None = None,
    feature_columns: list[str] | None = None,
    ohlcv_df: pd.DataFrame | None = None,
) -> SignalResult:
    """Run the full cache->features->inference->fusion pipeline for one
    (symbol, market, timeframe). `ohlcv_df` is a testing hook to bypass
    the live cache/DB/API path with pre-built data."""
    from ml_pipeline.common import FEATURE_COLUMNS

    feature_columns = feature_columns or FEATURE_COLUMNS

    df = ohlcv_df if ohlcv_df is not None else fetch_ohlcv_cached(symbol, market, timeframe, lookback_days)
    df = drop_unclosed_candle(df, timeframe)
    if df.empty:
        raise ValueError(f"No closed-candle data available for {symbol} ({timeframe})")

    feats = drop_warmup(compute_features(df))
    if feats.empty:
        raise ValueError(f"Not enough history to compute features for {symbol} ({timeframe})")

    latest = feats.iloc[-1]

    P_indicators = combine_indicator_votes(
        rsi=latest["rsi_14"],
        macd_hist=latest["macd_hist"],
        atr=latest["atr_14"],
        percent_b=latest["bb_percent_b"],
        indicator_weights=indicator_weights,
    )

    P_lgbm = None
    if lgbm_bundle is not None:
        X = latest[feature_columns].to_numpy(dtype=np.float32).reshape(1, -1)
        P_lgbm = predict_proba(lgbm_bundle, X)[0]

    P_lstm = None
    if lstm_bundle is not None:
        seq_len = lstm_bundle.seq_len
        if len(feats) >= seq_len:
            window = feats[feature_columns].to_numpy(dtype=np.float32)[-seq_len:]
            scaled = lstm_bundle.scaler.transform(window).astype(np.float32) if lstm_bundle.scaler else window
            P_lstm = predict_proba(lstm_bundle, scaled.reshape(1, seq_len, -1))[0]
        else:
            logger.warning("Not enough bars (%d) for LSTM seq_len=%d — dropping LSTM from fusion", len(feats), seq_len)

    return fuse_signals(
        P_indicators=P_indicators,
        P_lgbm=P_lgbm,
        P_lstm=P_lstm,
        w_ind=w_ind,
        w_lgbm=w_lgbm if P_lgbm is not None else 0.0,
        w_lstm=w_lstm if P_lstm is not None else 0.0,
        confidence_threshold=confidence_threshold,
        timeframe=timeframe,
        close=float(latest["close"]),
        atr=float(latest["atr_14"]),
        horizon_bucket=horizon_bucket,
    )
