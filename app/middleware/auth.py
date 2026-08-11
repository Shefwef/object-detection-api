"""
API-key authentication + tiny in-memory rate limiter.

Both features are opt-in - toggled by ``AUTH_ENABLED`` in settings.  The
motivation is that the demo build should stay open (no key required), but
the same container can be locked down in production by flipping a single
environment variable.

Rate limiting is a fixed-window counter over 60 seconds keyed by ``X-API-Key``
(or client IP when auth is disabled).  For horizontally scaled deployments
plug a Redis-backed limiter in via :func:`register_middleware`.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings


API_KEY_HEADER = "X-API-Key"

# Paths that never require authentication - health/probes/docs must stay open
_PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}


def _is_public(path: str) -> bool:
    if path in _PUBLIC_PATHS:
        return True
    return path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/openapi")


class RateLimiter:
    """In-process fixed-window counter."""

    def __init__(self, per_minute: int) -> None:
        self._max = per_minute
        self._window = 60.0
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, key: str) -> bool:
        now = time.time()
        bucket = self._hits[key]
        while bucket and now - bucket[0] > self._window:
            bucket.popleft()
        if len(bucket) >= self._max:
            return False
        bucket.append(now)
        return True


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)
        settings = get_settings()
        self._enabled = settings.AUTH_ENABLED
        self._valid_keys = set(settings.api_keys)
        self._limiter = RateLimiter(settings.RATE_LIMIT_PER_MINUTE)

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        path = request.url.path
        if _is_public(path):
            return await call_next(request)

        client_key = request.headers.get(API_KEY_HEADER) or (request.client.host if request.client else "unknown")

        if self._enabled:
            supplied = request.headers.get(API_KEY_HEADER)
            if not supplied or supplied not in self._valid_keys:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": f"Missing or invalid {API_KEY_HEADER} header."},
                    headers={"WWW-Authenticate": f'ApiKey realm="{API_KEY_HEADER}"'},
                )

        if not self._limiter.check(client_key):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Try again in a minute."},
                headers={"Retry-After": "60"},
            )

        return await call_next(request)


def register_middleware(app: FastAPI) -> None:
    """Attach the auth + rate limit middleware to the FastAPI application."""
    app.add_middleware(AuthMiddleware)


# ─── FastAPI dependency alternative ────────────────────────────────────────
#
# Prefer the middleware above so every route is protected uniformly.  The
# dependency form below is exported for endpoints that want to enforce auth
# even when the global middleware is disabled (e.g. an admin endpoint).


async def require_api_key(request: Request) -> str:
    settings = get_settings()
    if not settings.AUTH_ENABLED:
        return "auth-disabled"
    supplied = request.headers.get(API_KEY_HEADER)
    if not supplied or supplied not in set(settings.api_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid {API_KEY_HEADER} header.",
        )
    return supplied
