"""API Server (FastAPI) — Phase 1 MVP.

Lazy-loads all sub-modules via __getattr__ to avoid importing
torch (~5.7s on macOS) at package import time.

Usage:
    from server import create_app, create_engine

    engine = create_engine()       # Slow: imports torch
    app = create_app(engine)       # Fast: configures FastAPI
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "create_app",
    "create_engine",
]

_LAZY_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import server.app as _app

        _globals: dict[str, Any] = {
            "create_app": _app.create_app,
            "create_engine": _app.create_engine,
        }
        if name in _globals:
            value = _globals[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module 'server' has no attribute '{name}'")
