"""Structured logging for LLM-ServeForge.

Provides lightweight context-aware logging without external dependencies.
Log level is controlled by the LLM_SERVEFORGE_LOG environment variable
(default: WARNING).  Set to DEBUG for detailed request tracing.

Usage:
    from utils.logging import get_logger
    log = get_logger(__name__)
    log.info("schedule step", extra={"batch_size": 4, "step_id": 12})
"""

from __future__ import annotations

import logging
import os
import time
import sys
from typing import Any

_LOG_LEVEL = os.environ.get("LLM_SERVEFORGE_LOG", "WARNING").upper()

_root_handler: logging.Handler | None = None


def _init_root_handler() -> logging.Handler:
    global _root_handler
    if _root_handler is not None:
        return _root_handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, _LOG_LEVEL, logging.WARNING))
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-5s] %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    _root_handler = handler
    return handler


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for the given module name."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, _LOG_LEVEL, logging.WARNING))
    logger.propagate = False
    handler = _init_root_handler()
    if handler not in logger.handlers:
        logger.addHandler(handler)
    return logger


def log_step_begin(
    logger: logging.Logger,
    step_id: int,
    waiting: int,
    running: int,
) -> None:
    logger.info(
        "step %d begin | waiting=%d running=%d",
        step_id, waiting, running,
    )


def log_step_end(
    logger: logging.Logger,
    step_id: int,
    duration_ms: float,
    batch_size: int,
    total_tokens: int,
    results: int,
) -> None:
    logger.info(
        "step %d end | %.1fms batch=%d tokens=%d results=%d",
        step_id, duration_ms, batch_size, total_tokens, results,
    )


def log_request_lifecycle(
    logger: logging.Logger,
    request_id: str,
    event: str,
    **extra: Any,
) -> None:
    parts = f"req={request_id} {event}"
    if extra:
        parts += " | " + " ".join(f"{k}={v}" for k, v in extra.items())
    logger.debug(parts)


class StepTimer:
    """Context manager that logs step duration."""

    def __init__(self, logger: logging.Logger, step_id: int) -> None:
        self._logger = logger
        self._step_id = step_id
        self._start: float = 0.0

    def __enter__(self) -> StepTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._logger.debug("step %d duration %.2fms", self._step_id, elapsed_ms)
