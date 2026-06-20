"""SQLite persistence layer for OHLCV time series.

Design constraints (docs/02_data_architecture_and_features.md):
- Composite PRIMARY KEY (asset_id, timeframe, timestamp) -> ingestion is an
  idempotent UPSERT, never duplicate-insert.
- WAL mode so concurrent reads don't block on writes (multi-user UI).
- On HF Spaces the disk is ephemeral: this file is a *cache*, rebuilt at
  startup from an HF Dataset repo or treated as a pure read-cache over the
  live API path. See docs/04_deployment_and_environment.md.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    asset_id   TEXT    NOT NULL,
    timeframe  TEXT    NOT NULL,
    timestamp  INTEGER NOT NULL,  -- unix epoch seconds, UTC
    open       REAL    NOT NULL,
    high       REAL    NOT NULL,
    low        REAL    NOT NULL,
    close      REAL    NOT NULL,
    volume     REAL,
    adjusted   INTEGER NOT NULL DEFAULT 1,  -- 1 = split/dividend-adjusted
    PRIMARY KEY (asset_id, timeframe, timestamp)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_ohlcv_asset_tf_ts
    ON ohlcv (asset_id, timeframe, timestamp);
"""

UPSERT_SQL = """
INSERT INTO ohlcv (asset_id, timeframe, timestamp, open, high, low, close, volume, adjusted)
VALUES (:asset_id, :timeframe, :timestamp, :open, :high, :low, :close, :volume, :adjusted)
ON CONFLICT (asset_id, timeframe, timestamp) DO UPDATE SET
    open = excluded.open,
    high = excluded.high,
    low = excluded.low,
    close = excluded.close,
    volume = excluded.volume,
    adjusted = excluded.adjusted;
"""


def get_db_path() -> Path:
    path = settings.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
    logger.info("SQLite schema ready at %s (WAL mode)", get_db_path())


def upsert_ohlcv_rows(rows: Iterable[dict]) -> int:
    """Idempotent bulk UPSERT. Each row must have the keys in UPSERT_SQL."""
    rows = list(rows)
    if not rows:
        return 0
    with get_connection() as conn:
        conn.execute("BEGIN;")
        try:
            conn.executemany(UPSERT_SQL, rows)
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise
    logger.debug("Upserted %d ohlcv rows", len(rows))
    return len(rows)


def fetch_ohlcv(asset_id: str, timeframe: str, start_ts: int, end_ts: int) -> list[sqlite3.Row]:
    query = """
        SELECT * FROM ohlcv
        WHERE asset_id = ? AND timeframe = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC;
    """
    with get_connection() as conn:
        return conn.execute(query, (asset_id, timeframe, start_ts, end_ts)).fetchall()
