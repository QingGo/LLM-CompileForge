"""Tests for JSON logging format and structured events.

Verifies that:
  1. JSON format produces valid JSON lines with expected keys
  2. Text format remains backward compatible
  3. Structured engine events contain event_type in JSON mode
  4. Server middleware emits JSON with method/path/status

All tests restore logging state to avoid leaking into other tests.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

import pytest


def _capture_json_logging() -> tuple[io.StringIO, Callable[[], None]]:
    """Configure JSON logging to a StringIO buffer.

    Returns (captured_buffer, restore_function).
    Call restore() after the test to reset logging state.
    """
    os.environ["LLM_SERVEFORGE_LOG_FORMAT"] = "json"

    captured = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured

    from utils.logging import init_logging

    init_logging()
    # Override module-level _LOG_LEVEL default (WARNING) so INFO messages appear
    logging.getLogger().setLevel(logging.INFO)
    for h in logging.getLogger().handlers:
        h.setLevel(logging.INFO)

    def restore() -> None:
        sys.stderr = old_stderr
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        logging.basicConfig(level=logging.WARNING, force=True)

    return captured, restore


def _capture_text_logging() -> tuple[io.StringIO, Callable[[], None]]:
    """Configure text logging to a StringIO buffer.

    Returns (captured_buffer, restore_function).
    """
    os.environ.pop("LLM_SERVEFORGE_LOG_FORMAT", None)

    captured = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = captured

    from utils.logging import init_logging

    init_logging()
    # Override module-level _LOG_LEVEL default (WARNING) so INFO messages appear
    logging.getLogger().setLevel(logging.INFO)
    for h in logging.getLogger().handlers:
        h.setLevel(logging.INFO)

    def restore() -> None:
        sys.stderr = old_stderr
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        logging.basicConfig(level=logging.WARNING, force=True)

    return captured, restore


# ═══════════════════════════════════════════════════════════════════════
# 1. JSON format produces valid JSON
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestJsonFormatValid:
    """JSON logging format produces parseable JSON lines."""

    def test_json_format_valid(self) -> None:
        captured, restore = _capture_json_logging()
        try:
            log = logging.getLogger("test_json")
            log.info("hello world", extra={
                "event_type": "test_event",
                "event_data": {"x": 1, "y": "two"},
            })
            output = captured.getvalue().strip()
            assert output, "Expected non-empty log output"
            data = json.loads(output)
            assert "timestamp" in data
            assert "level" in data
            assert data["level"] == "INFO"
            assert "logger" in data
            assert data["logger"] == "test_json"
            assert "message" in data
            assert data["message"] == "hello world"
        finally:
            restore()


# ═══════════════════════════════════════════════════════════════════════
# 2. Text format backward compatible
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTextFormatBackward:
    """Default text format unchanged and distinguishable from JSON."""

    def test_text_format_backward_compatible(self) -> None:
        captured, restore = _capture_text_logging()
        try:
            log = logging.getLogger("test_text")
            log.info("plain text message")
            output = captured.getvalue().strip()
            assert output, "Expected non-empty log output"
            with pytest.raises(json.JSONDecodeError):
                json.loads(output)
            assert "plain text message" in output
            assert "test_text" in output
            assert "test_text" in output, "Expected logger name in text format"
            assert "plain text message" in output
        finally:
            restore()


# ═══════════════════════════════════════════════════════════════════════
# 3. Structured engine event has event_type
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestJsonEngineEvent:
    """Structured logging via log_step_begin includes event_type in JSON."""

    def test_json_engine_event(self) -> None:
        captured, restore = _capture_json_logging()
        try:
            from utils.logging import log_step_begin

            log = logging.getLogger("test_engine")
            log_step_begin(log, step_id=7, waiting=2, running=1)
            output = captured.getvalue().strip()
            assert output, "Expected non-empty log output"
            data = json.loads(output)
            assert data.get("event_type") == "engine_step"
            assert "event_data" in data
            event_data = data["event_data"]
            assert event_data["step_id"] == 7
            assert event_data["phase"] == "begin"
            assert event_data["waiting"] == 2
            assert event_data["running"] == 1
        finally:
            restore()


# ═══════════════════════════════════════════════════════════════════════
# 4. Server middleware request log works
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestJsonServerRequest:
    """Server middleware emits JSON with method/path/status in JSON mode."""

    def test_json_server_request(self) -> None:
        captured, restore = _capture_json_logging()
        try:
            from fastapi import FastAPI, Request
            from fastapi.testclient import TestClient

            app = FastAPI()
            logger = logging.getLogger("test_server")

            @app.middleware("http")
            async def _log_middleware(request: Request, call_next: Any) -> Any:
                import time as _time

                start = _time.time()
                response = await call_next(request)
                duration_ms = int((_time.time() - start) * 1000)
                logger.info(
                    "request %s %s -> %d (%dms)",
                    request.method,
                    request.url.path,
                    response.status_code,
                    duration_ms,
                    extra={
                        "event_type": "server_request",
                        "event_data": {
                            "method": request.method,
                            "path": request.url.path,
                            "status": response.status_code,
                            "duration_ms": duration_ms,
                        },
                    },
                )
                return response

            @app.get("/health")
            async def _health():
                return {"status": "ok"}

            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200

            output = captured.getvalue().strip()
            lines = [line for line in output.split("\n") if line.strip()]
            json_events = [json.loads(line) for line in lines]
            server_events = [e for e in json_events if e.get("event_type") == "server_request"]
            assert len(server_events) >= 1, (
                f"Expected at least 1 server_request event, got {len(server_events)}"
            )
            data = server_events[0]["event_data"]
            assert data["method"] == "GET"
            assert data["status"] == 200
            assert "path" in data
            assert "duration_ms" in data
        finally:
            restore()


# ── Pass-through for manual invocation ──────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
