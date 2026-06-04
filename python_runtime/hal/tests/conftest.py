"""Shared pytest configuration for HAL unit tests."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by HAL tests."""
    config.addinivalue_line("markers", "unit: fast unit tests")
