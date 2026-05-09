"""Smoke test — verifies core environment is functional.

"What are you?" pattern: rapid sanity check that the project
can import its own modules and Python environment is intact.
"""
import pytest


@pytest.mark.smoke
def test_python_environment():
    """Verify Python >= 3.10 and basic imports work."""
    import sys
    assert sys.version_info >= (3, 10), f"Need Python >= 3.10, got {sys.version}"


@pytest.mark.smoke
def test_package_imports():
    """All core modules must be importable."""
    import compiler  # noqa: F401
    import engine  # noqa: F401
    import hal  # noqa: F401
    import server  # noqa: F401


@pytest.mark.smoke
@pytest.mark.timeout(30)
def test_dependencies_available():
    """Key runtime dependencies must be installed."""
    import fastapi  # noqa: F401
    import numpy  # noqa: F401
    import pydantic  # noqa: F401
    import torch  # noqa: F401
