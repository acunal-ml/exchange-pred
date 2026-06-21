"""NASDAQ + commodity OHLCV via yfinance.

docs/02: always use split/dividend-adjusted prices (auto_adjust=True) for
features. Source timestamps are America/New_York for US equities; we
convert to UTC immediately so nothing downstream has to think about tz.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from data_pipeline.sources.base import DataSource, DataSourceError
from utils.logging_config import get_logger

logger = get_logger(__name__)

# yfinance has no native "4h"; build it by resampling 1h candles.
_TIMEFRAME_TO_YF_INTERVAL = {
    "5m": "5m",
    "15m": "15m",
    "1H": "60m",
    "4H": "60m",  # resampled below
    "1D": "1d",
    "1W": "1wk",
    "1M": "1mo",
}

_RESAMPLE_RULE = {"4H": "4h"}


class YFinanceSource(DataSource):
    name = "yfinance"

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def _download(self, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if df.empty:
            raise DataSourceError(f"yfinance returned no data for {symbol} ({interval})")
        return df

    def fetch_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        if timeframe not in _TIMEFRAME_TO_YF_INTERVAL:
            raise DataSourceError(f"Unsupported timeframe: {timeframe}")

        interval = _TIMEFRAME_TO_YF_INTERVAL[timeframe]
        try:
            df = self._download(symbol, interval, start, end)
        except Exception as exc:
            raise DataSourceError(f"yfinance fetch failed for {symbol}: {exc}") from exc

        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]

        # yfinance index is tz-aware in the exchange's local tz once a
        # period/interval is set; normalize to UTC.
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        df.index = df.index.tz_convert("UTC")

        if timeframe in _RESAMPLE_RULE:
            df = (
                df.resample(_RESAMPLE_RULE[timeframe])
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna(subset=["open", "high", "low", "close"])
            )

        return df
