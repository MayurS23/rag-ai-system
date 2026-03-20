"""
utils/logger.py
---------------
Structured logging setup using structlog.
Produces JSON logs in production (machine-parseable for Datadog / CloudWatch),
and pretty console output in development mode.
Every log entry automatically includes:
  - timestamp, level, module, function
  - request_id (if set via context var)
  - Any extra fields passed as kwargs
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional

import structlog
from structlog.types import EventDict

from utils.config import settings

# ── Context variable to carry request_id through async call chains ────────────
_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> str:
    return _request_id_ctx.get() or "no-request"


def set_request_id(request_id: Optional[str] = None) -> str:
    rid = request_id or str(uuid.uuid4())[:8]
    _request_id_ctx.set(rid)
    return rid


# ── Custom processors ─────────────────────────────────────────────────────────

def add_request_id(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    """Inject the current request_id into every log entry."""
    event_dict["request_id"] = get_request_id()
    return event_dict


def add_app_info(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    """Inject static app metadata."""
    event_dict["app"] = settings.APP_NAME
    event_dict["version"] = settings.APP_VERSION
    return event_dict


# ── Logger factory ────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    """
    Configure structlog with shared processors.
    Call once at application startup.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Standard library logging — captures third-party library logs too
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=handlers,
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        add_request_id,
        add_app_info,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_FORMAT == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    for handler in handlers:
        handler.setFormatter(formatter)


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """
    Get a named logger instance.

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("ingesting document", file="report.pdf", chunks=42)
    """
    return structlog.get_logger(name)


# Configure on import
_configure_logging()

# Module-level logger for utils itself
logger = get_logger(__name__)
