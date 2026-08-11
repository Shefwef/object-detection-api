"""
Structured logging bootstrap.

When ``LOG_JSON`` is True and ``structlog`` is installed, every log line is
emitted as a JSON document that downstream aggregators (CloudWatch, Datadog,
Loki, ...) can parse without regex.  Otherwise we fall back to the standard
library formatter so this project has no hard runtime dependency on structlog.
"""

from __future__ import annotations

import logging
import sys

from app.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    if settings.LOG_JSON:
        try:
            _configure_structlog(level)
            return
        except Exception:  # pragma: no cover - structlog missing / broken
            logging.warning("structlog unavailable - using plain text logging")

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _configure_structlog(level: int) -> None:
    import structlog

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
