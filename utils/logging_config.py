"""Shared logging setup and UTC timezone helpers.

All timestamps in this project are stored and compared in UTC — see
docs/02_data_architecture_and_features.md ("Data hygiene"). Mixing
America/New_York (NASDAQ) and Europe/Istanbul (BIST) session times
without normalizing to UTC corrupts intraday alignment.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from config.settings import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """Normalize a naive or aware datetime to UTC.

    Naive datetimes are assumed to already be UTC (callers must localize
    exchange-local timestamps to their own tz *before* calling this).
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
