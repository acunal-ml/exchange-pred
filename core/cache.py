"""Abstract cache layer.

docs/02 and docs/04 are explicit: do NOT hard-code Redis. Redis isn't
available on the HF free tier, so a Redis-only design can't deploy. This
module defines a backend-agnostic `CacheBackend` interface with:
- RedisBackend   -> local dev (Docker/daemon), shared, supports TTL.
- TTLCacheBackend -> HF / fallback, in-process cachetools.TTLCache.

Selection is driven by `settings.cache_backend` ("redis" | "ttl").
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from cachetools import TTLCache

from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)


class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str) -> Any | None: ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class TTLCacheBackend(CacheBackend):
    """In-process fallback. Safe on HF Spaces (no external service needed).

    Caveat: not shared across worker processes/replicas, and lost on
    restart — acceptable per docs/04, since the live API + SQLite path
    remains the source of truth.
    """

    def __init__(self, maxsize: int = 2048, default_ttl: int | None = None) -> None:
        self._default_ttl = default_ttl or settings.cache_default_ttl_seconds
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=self._default_ttl)

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        # cachetools.TTLCache has a single TTL for the whole cache instance;
        # a per-key TTL would need a second cache or manual expiry tracking.
        # Not needed yet — every caller today uses the default TTL.
        self._cache[key] = value

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)


class RedisBackend(CacheBackend):
    """Local dev backend. Requires a reachable Redis (Docker/daemon)."""

    def __init__(self, url: str | None = None) -> None:
        import redis  # local import: keep redis optional on HF

        self._client = redis.Redis.from_url(url or settings.redis_url, decode_responses=True)
        self._default_ttl = settings.cache_default_ttl_seconds

    def get(self, key: str) -> Any | None:
        raw = self._client.get(key)
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        self._client.set(key, json.dumps(value), ex=ttl_seconds or self._default_ttl)

    def delete(self, key: str) -> None:
        self._client.delete(key)


def build_cache_backend() -> CacheBackend:
    backend = settings.cache_backend.lower()
    if backend == "redis":
        try:
            client = RedisBackend()
            client._client.ping()
            logger.info("Using RedisBackend at %s", settings.redis_url)
            return client
        except Exception as exc:
            logger.warning("Redis unavailable (%s); falling back to TTLCacheBackend", exc)
            return TTLCacheBackend()
    logger.info("Using TTLCacheBackend (in-process)")
    return TTLCacheBackend()


cache: CacheBackend = build_cache_backend()
