"""Pytest configuration for engine unit tests.

All tests here are self-contained unit tests that only depend on engine/*
modules. No compiler or runtime dependencies.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

# ── Force offline HF loading: local cache only, no HTTP revision checks ──
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def pytest_runtest_setup(item: Any) -> None:
    if item.get_closest_marker("unit"):
        item.own_markers = list(item.own_markers)
        item.own_markers.append(pytest.mark.timeout(1))
