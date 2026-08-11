"""
Repository for cached inference results.

The service layer talks to :class:`IInferenceRepository`, not to Redis or a
dict, which keeps caching backend-agnostic.  Two implementations ship out of
the box:

* :class:`InMemoryInferenceRepository` - process-local LRU-ish dict; the
  default for local development and unit tests. Zero external dependencies.
* :class:`RedisInferenceRepository` - production cache used when
  ``REDIS_URL`` is configured. TTL is per-image-hash so identical requests
  are served instantly for a bounded window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Optional
import hashlib
import json
import logging
import time

logger = logging.getLogger(__name__)


# ─── Helpers ───────────────────────────────────────────────────────────────


def compute_image_hash(image_bytes: bytes) -> str:
    """SHA-256 of raw image bytes - used as the cache key."""
    return hashlib.sha256(image_bytes).hexdigest()


# ─── Interface ─────────────────────────────────────────────────────────────


class IInferenceRepository(ABC):
    """Contract for any backend that caches inference results by image hash."""

    @abstractmethod
    async def get(self, image_hash: str, model: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    async def set(self, image_hash: str, model: str, payload: Dict[str, Any]) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @abstractmethod
    async def size(self) -> int: ...


# ─── In-memory implementation ──────────────────────────────────────────────


class InMemoryInferenceRepository(IInferenceRepository):
    """LRU-style dict cache with TTL. Suitable for single-process demos."""

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 512) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: "OrderedDict[str, tuple[float, Dict[str, Any]]]" = OrderedDict()

    @staticmethod
    def _key(image_hash: str, model: str) -> str:
        return f"{model}:{image_hash}"

    async def get(self, image_hash: str, model: str) -> Optional[Dict[str, Any]]:
        key = self._key(image_hash, model)
        entry = self._store.get(key)
        if entry is None:
            return None
        stored_at, payload = entry
        if time.time() - stored_at > self._ttl:
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)  # LRU touch
        return payload

    async def set(self, image_hash: str, model: str, payload: Dict[str, Any]) -> None:
        key = self._key(image_hash, model)
        self._store[key] = (time.time(), payload)
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    async def clear(self) -> None:
        self._store.clear()

    async def size(self) -> int:
        return len(self._store)


# ─── Redis implementation (optional) ───────────────────────────────────────


class RedisInferenceRepository(IInferenceRepository):
    """Redis-backed cache. Requires the ``redis`` package."""

    def __init__(self, redis_client: Any, ttl_seconds: int = 3600) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds

    @staticmethod
    def _key(image_hash: str, model: str) -> str:
        return f"inference:{model}:{image_hash}"

    async def get(self, image_hash: str, model: str) -> Optional[Dict[str, Any]]:
        try:
            raw = await self._redis.get(self._key(image_hash, model))
        except Exception:  # pragma: no cover - defensive against Redis outages
            logger.exception("Redis GET failed - falling back to cache miss")
            return None
        return json.loads(raw) if raw else None

    async def set(self, image_hash: str, model: str, payload: Dict[str, Any]) -> None:
        try:
            await self._redis.setex(
                self._key(image_hash, model),
                self._ttl,
                json.dumps(payload, default=str),
            )
        except Exception:  # pragma: no cover
            logger.exception("Redis SETEX failed - continuing without cache")

    async def clear(self) -> None:
        try:
            await self._redis.flushdb()
        except Exception:  # pragma: no cover
            logger.exception("Redis FLUSHDB failed")

    async def size(self) -> int:
        try:
            return int(await self._redis.dbsize())
        except Exception:  # pragma: no cover
            return -1
