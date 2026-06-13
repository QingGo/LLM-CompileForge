"""Per-position and per-layer cosine similarity tests for compiled dylib.

Tests compare ctypes (dylib) output vs Python executor reference.
Every position must have cos > 0.99 — per-layer is supplementary.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

from scripts._cos import cosine_similarity

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from scripts.ctypes_forward import run_ctypes, run_python_executor  # noqa: E402

DYLIB_DIR = "outputs/compiled/opt_125m_fresh"


def _dylib_exists() -> bool:
    try:
        return os.path.isdir(DYLIB_DIR) and any(f.endswith(".dylib") for f in os.listdir(DYLIB_DIR))
    except OSError:
        return False


requires_dylib = pytest.mark.skipif(
    not _dylib_exists(),
    reason="compiled opt_125m_fresh dylib not found at " + DYLIB_DIR,
)


# ── Module-scoped fixtures (run once per module) ──────────────


def _run_ctypes():
    if not _dylib_exists():
        pytest.skip("dylib not found")
    result = run_ctypes(artifact_dir=DYLIB_DIR)
    if len(result) == 0:
        pytest.skip("dylib compiled without compute graph (skip_compute_graph=True) — recompile to enable this test")
    return result


def _run_python():
    if not _dylib_exists():
        pytest.skip("dylib not found")
    return run_python_executor(artifact_dir=DYLIB_DIR)


@pytest.fixture(scope="module")
def dylib_result():
    return _run_ctypes()


@pytest.fixture(scope="module")
def python_result():
    return _run_python()


# ── Per-position cosine test — the primary signal ─────────────


@pytest.mark.integration
@pytest.mark.timeout(120)
@requires_dylib
def test_per_position_cosine(
    dylib_result: Any,
    python_result: Any,
) -> None:
    """Every (batch, pos) must have cos > 0.99."""
    dylib = dylib_result
    python = python_result
    for batch in range(2):
        for pos in range(4):
            dylib_out = dylib[batch, pos]
            python_out = python[batch, pos]
            cos = cosine_similarity(dylib_out, python_out)
            assert cos > 0.99, f"batch={batch}, pos={pos}: cos={cos}"


# ── Per-layer cosine test — supplementary signal ──────────────


@pytest.mark.integration
@pytest.mark.timeout(120)
@requires_dylib
def test_per_layer_cosine(
    dylib_result: Any,
    python_result: Any,
) -> None:
    """Per-function cosine must be > 0.99 for all functions."""
    dylib = dylib_result
    python = python_result
    for fi in range(1, len(dylib)):
        dylib_layer = dylib[fi]
        python_layer = python[fi]
        cos = cosine_similarity(dylib_layer, python_layer)
        assert cos > 0.99, f"func[{fi}]: cos={cos}"


# ── Standalone execution ──────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
