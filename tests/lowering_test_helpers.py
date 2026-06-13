# ruff: noqa: E501
"""Shared test helpers for lowering pattern tests.

Extracted from test_lowering_patterns.py and test_mlir_lowering.py to
avoid duplication after the merge+split refactor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_mlir_pkg = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
    sys.path.insert(0, str(_mlir_pkg))

from compiler.pipeline import _apply_sf_to_linalg  # noqa: E402

try:
    from compiler.backend.llvm_backend import _has_bindings as _has_vec_bindings  # noqa: F811
except ImportError:

    def _has_vec_bindings() -> bool:
        return False


MLIR_BINDINGS = False
try:
    import mlir.ir as ir  # noqa: F401

    MLIR_BINDINGS = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not MLIR_BINDINGS, reason="mlir-core not available")


def lower(sf_text: str) -> str:
    """Lower a single-op module and return the lowered text."""
    lowered = _apply_sf_to_linalg(sf_text)
    assert lowered is not None, "Lowering returned None"
    assert len(lowered) > 0, "Lowering returned empty string"
    return lowered


def check_lowered(
    lowered: str,
    expected: str = "linalg.",
    not_expected: str = "sf.",
) -> None:
    """Verify lowered output contains expected and lacks not_expected."""
    if not_expected and not_expected in lowered:
        raise AssertionError(f"Lowered IR still contains '{not_expected}'. First 500 chars: {lowered[:500]}")
    if expected and expected not in lowered:
        # Some ops lower to tensor.dialect ops, not linalg
        if "tensor." in lowered or "arith." in lowered:
            return
    if expected:
        assert expected in lowered, f"Expected '{expected}' not found in lowered output"


def check_op_count(text: str, op_name: str, min_count: int = 1) -> None:
    """Assert op_name appears at least min_count times in text."""
    count = text.count(op_name)
    assert count >= min_count, f"Expected >= {min_count} '{op_name}', got {count}"


def check_absent(text: str, op_name: str) -> None:
    """Assert op_name does not appear in text."""
    assert op_name not in text, f"'{op_name}' should not be in lowered output"


# ── Template modules ─────────────────────────────────────────────


BINARY_MODULE = (
    "module {{\n"
    "  func.func @test(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xf32> {{\n"
    '    %0 = "{op}"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>\n'
    "    return %0 : tensor<2x4xf32>\n"
    "  }}\n"
    "}}"
)

ACTIVATION_MODULE = (
    "module {{\n"
    "  func.func @test(%a: tensor<2x4xf32>) -> tensor<2x4xf32> {{\n"
    '    %0 = "{op}"(%a) : (tensor<2x4xf32>) -> tensor<2x4xf32>\n'
    "    return %0 : tensor<2x4xf32>\n"
    "  }}\n"
    "}}"
)


# ── Bufferization validation helper ──────────────────────────────


def lower_and_bufferize(mlir_text: str) -> bool:
    """Lower sf ops, parse result, run one-shot-bufferize to validate."""
    import mlir.ir as ir
    import mlir.passmanager as pm

    lowered_text = _apply_sf_to_linalg(mlir_text)
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(lowered_text, ctx)
        pman = pm.PassManager.parse("builtin.module(one-shot-bufferize{bufferize-function-boundaries})", ctx)
        pman.run(module.operation)
    return True
