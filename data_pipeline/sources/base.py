"""Common interface every market-data source implements.

docs/02_data_architecture_and_features.md calls out tvDatafeed as
unofficial/scraping-based and prone to breaking — it must sit behind this
interface with retry/backoff so a flaky source can't take down ingestion,
and so a fallback source can be swapped in without touching callers.

All sources return UTC-indexed OHLCV rows shaped for
core.db_setup.upsert_ohlcv_rows: asset_id, timeframe, timestamp (unix
epoch seconds UTC), open, high, low, close, volume, adjusted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataSourceError(Exception):
    """Raised when a source fails after exhausting its retry policy."""


class DataSource(ABC):
    name: str

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by UTC timestamp with columns
        open, high, low, close, volume. Must raise DataSourceError (not a
        raw exception) on unrecoverable failure so callers can fall back.
        """
        raise NotImplementedError


def session_aligned_resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample intraday OHLCV bars to a coarser rule, anchored to the
    session's actual opening time-of-day rather than midnight UTC.

    A fixed midnight-UTC origin (pandas resample's default) can split a
    single trading session across two buckets if the session straddles
    a bucket boundary — e.g. BIST's ~06:00-15:00 UTC session straddles
    the default 12:00 UTC half-day boundary, silently degrading "12H"
    bars to the same bar count as "6H" (found via real-data testing,
    not assumed). The session's opening time-of-day is derived from the
    data itself (the minimum minutes-since-midnight seen) rather than
    hardcoded per exchange, so this works for any source/exchange
    without per-market special-casing.
    """
    time_of_day_minutes = df.index.hour * 60 + df.index.minute
    origin = df.index[0].normalize() + pd.Timedelta(minutes=int(time_of_day_minutes.min()))
    return (
        df.resample(rule, origin=origin)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
    )


def to_ohlcv_rows(df: pd.DataFrame, asset_id: str, timeframe: str, adjusted: bool = True) -> list[dict]:
    """Convert a UTC-indexed OHLCV DataFrame into upsert-ready row dicts."""
    rows = []
    for ts, row in df.iterrows():
        rows.append(
            {
                "asset_id": asset_id,
                "timeframe": timeframe,
                "timestamp": int(ts.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if pd.notna(row.get("volume")) else None,
                "adjusted": int(adjusted),
            }
        )
    return rows
