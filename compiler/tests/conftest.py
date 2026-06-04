"""Shared test configuration for compiler/tests/.

Provides common fixtures and project-root sys.path setup so individual test
files can focus on test logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# Ensure the project root is on sys.path so all compiler imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Common fixtures ──────────────────────────────────────────────────


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root directory."""
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def dummy_tensor() -> torch.Tensor:
    """A small reproducible tensor for shape-invariant tests."""
    torch.manual_seed(42)
    return torch.randn(2, 4)


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory scoped to a single test."""
    d = tmp_path / "output"
    d.mkdir()
    return d


# ── MLIR fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def mlir_context() -> Any:
    """Session-scoped MLIR Context shared across all compiler tests.

    Shared to avoid creating/destroying many Context objects, which can
    trigger nanobind type registry instability in LLVM 22.x.
    """
    import mlir.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    return ctx


# MLIR bindings may not be available in all environments.
# Markers are defined in pyproject.toml — no re-registration needed here.


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Skip MLIR-dependent tests when bindings are unavailable."""
    _has_mlir = _check_mlir_bindings()
    if not _has_mlir:
        skip_mlir = pytest.mark.skip(reason="MLIR Python bindings not available")
        for item in items:
            _nid = item.nodeid.lower()
            _pipeline_keys = (
                "mlir", "llvm", "pipeline_ir", "pipeline_stage",
                "pipeline_lowering", "pipeline_validation", "pipeline_bugs"
            )
            if any(k in _nid for k in _pipeline_keys):
                item.add_marker(skip_mlir)


def _check_mlir_bindings() -> bool:
    """Check if MLIR Python bindings are importable."""
    try:
        import mlir.ir  # noqa: F401
        return True
    except ImportError:
        return False
