"""BIST (Turkey) OHLCV via tvDatafeed.

docs/02 flags tvDatafeed as unofficial/scraping-based and prone to
breaking — wrapped here behind the DataSource interface with retry/backoff
so callers (analysis_engine) never see a raw tvDatafeed exception, only
DataSourceError, and can fall back to cached/SQLite data instead.

Source timestamps from TradingView are Europe/Istanbul; converted to UTC.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from data_pipeline.sources.base import DataSource, DataSourceError
from utils.logging_config import get_logger

logger = get_logger(__name__)

_TIMEFRAME_TO_TV_INTERVAL = {
    "5m": "5",
    "15m": "15",
    "1H": "1H",
    "4H": "4H",
    "1D": "1D",
    "1W": "1W",
    "1M": "1M",
}


class TVDatafeedSource(DataSource):
    name = "tvdatafeed"

    def __init__(self) -> None:
        self._tv = None  # lazy: avoid network/login at import time

    def _client(self):
        if self._tv is None:
            from tvDatafeed import TvDatafeed

            self._tv = TvDatafeed(
                username=settings.tvdatafeed_username,
                password=settings.tvdatafeed_password,
            )
        return self._tv

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _get_hist(self, symbol: str, interval, n_bars: int) -> pd.DataFrame:
        df = self._client().get_hist(
            symbol=symbol,
            exchange="BIST",
            interval=interval,
            n_bars=n_bars,
        )
        if df is None or df.empty:
            raise DataSourceError(f"tvDatafeed returned no data for {symbol}")
        return df

    def fetch_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        from tvDatafeed import Interval as TVInterval

        if timeframe not in _TIMEFRAME_TO_TV_INTERVAL:
            raise DataSourceError(f"Unsupported timeframe: {timeframe}")

        interval_map = {
            "5m": TVInterval.in_5_minute,
            "15m": TVInterval.in_15_minute,
            "1H": TVInterval.in_1_hour,
            "4H": TVInterval.in_4_hour,
            "1D": TVInterval.in_daily,
            "1W": TVInterval.in_weekly,
            "1M": TVInterval.in_monthly,
        }

        # TradingView's API takes a bar count, not a date range; over-fetch
        # and trim to [start, end] after converting to UTC.
        n_bars = 5000
        try:
            df = self._get_hist(symbol, interval_map[timeframe], n_bars)
        except Exception as exc:
            raise DataSourceError(f"tvDatafeed fetch failed for {symbol}: {exc}") from exc

        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]

        if df.index.tz is None:
            df.index = df.index.tz_localize("Europe/Istanbul")
        df.index = df.index.tz_convert("UTC")

        start_utc = pd.Timestamp(start).tz_localize("UTC") if start.tzinfo is None else pd.Timestamp(start).tz_convert("UTC")
        end_utc = pd.Timestamp(end).tz_localize("UTC") if end.tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
        return df.loc[(df.index >= start_utc) & (df.index <= end_utc)]
