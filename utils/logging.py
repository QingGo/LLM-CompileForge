"""Structured logging for LLM-ServeForge.

Provides lightweight context-aware logging without external dependencies.
Log level is controlled by the LLM_SERVEFORGE_LOG environment variable
(default: WARNING).  Set to DEBUG for detailed request tracing.

This module configures the **root logger** so that ALL modules (including
``compiler/*.py`` which use ``import logging; _log = logging.getLogger(...)``)
respond to ``LLM_SERVEFORGE_LOG``.  Engine-level convenience functions
(``get_logger``, ``StepTimer``, ``LogSession``) are provided on top.

Usage:
    from utils.logging import init_logging
    init_logging()  # call once at program start

    # Then in any module:
    import logging
    _log = logging.getLogger(__name__)  # works everywhere

    # Or use the convenience wrapper:
    from utils.logging import get_logger
    log = get_logger(__name__)

LogSession provides structured logging with timestamped directories:
    session = LogSession("pipeline", "opt_125m")
    session.save_ir("snapshot.mlir", mlir_text)
    session.save_report("timing.txt", timing_data)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_LOG_LEVEL_STR = os.environ.get("LLM_SERVEFORGE_LOG", "WARNING").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_STR, logging.WARNING)

_initialized: bool = False


def init_logging() -> None:
    """Initialize the root logger so all modules respond to ``LLM_SERVEFORGE_LOG``.

    Call this once at program startup (in `main()`, `scripts/`, or the
    server entry point).  After this, any module using
    ``logging.getLogger(__name__)`` will output at the configured level.
    """
    global _initialized
    if _initialized:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(_LOG_LEVEL)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)-5s] %(name)-24s %(message)s",
        datefmt="%H:%M:%S",
    ))

    root = logging.getLogger()
    root.setLevel(_LOG_LEVEL)
    # Remove default handlers to avoid duplicate output
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger at the configured level.

    Prefer calling ``init_logging()`` at startup and then using
    ``logging.getLogger(name)`` directly in each module.
    """
    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL)
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


class LogSession:
    """A compile/run log session with timestamped directory and semantic naming.

    Creates ``logs/<category>/<model>_<timestamp>/`` and provides
    methods to save IR snapshots and diagnostic reports.

    Usage:
        session = LogSession("pipeline", "opt_125m_fresh")
        path = session.save_ir(module, "stage_07_bufferize")
        session.save_report("timeout_report.txt", "Stage 7 timed out after 30s")
    """

    def __init__(
        self,
        category: str,
        model: str,
        base_dir: str = "logs",
    ) -> None:
        self.category = category
        self.model = model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(base_dir) / category / f"{model}_{timestamp}"
        self.path.mkdir(parents=True, exist_ok=True)

    def save_ir(self, ir_module: Any, stage_name: str) -> Path:
        """Save an IR module snapshot to the log directory.

        Accepts either an ``ir.Module`` (calls ``str()`` on it) or a plain string.
        Returns the path to the saved file.
        """
        text = str(ir_module) if not isinstance(ir_module, str) else ir_module
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in stage_name)
        path = self.path / f"{safe_name}.mlir"
        path.write_text(text)
        return path

    def save_report(self, name: str, content: str) -> Path:
        """Save a diagnostic report to the log directory."""
        path = self.path / name
        if not path.suffix:
            path = path.with_suffix(".txt")
        path.write_text(content)
        return path

    def __repr__(self) -> str:
        return f"LogSession({self.path})"
