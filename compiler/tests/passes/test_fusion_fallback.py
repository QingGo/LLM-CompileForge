"""Tests for _run_pattern fallback when sf dialect is unavailable.

The fix at lines 67-69 of compiler/mlir_passes/fusion.py adds a
try/except ImportError around the sf dialect registration, so _run_pattern
gracefully handles missing sf bindings.

Also verifies that allow_unregistered_dialects=True lets the pipeline
proceed with ops from unrecognized dialects.
"""

from __future__ import annotations

import builtins

import pytest

from compiler.passes.fusion import _run_pattern


@pytest.mark.unit
class TestRunPatternFallback:
    def test_matches_sf_mul_among_sf_ops(self):
        """_run_pattern finds sf.mul ops in MLIR with multiple sf ops."""
        mlir = """module {
  func.func @test(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "sf.add"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %1 = "sf.mul"(%0, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}"""
        found_mul = False

        def callback(op, rewriter):
            nonlocal found_mul
            if op.name == "sf.mul":
                found_mul = True
            return True

        result = _run_pattern(mlir, "sf.mul", callback)
        assert found_mul, "Callback should have matched sf.mul"
        assert "sf.mul" in result
        assert "sf.add" in result

    def test_unregistered_dialect_does_not_crash(self):
        """_run_pattern handles unregistered ops via allow_unregistered_dialects=True."""
        mlir = """module {
  func.func @test(%a: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "custom_dialect.foo"(%a) : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}"""

        def callback(op, rewriter):
            return True

        result = _run_pattern(mlir, "custom_dialect.foo", callback)
        assert "custom_dialect.foo" in result

    def test_noop_callback_unchanged(self):
        """_run_pattern returns unchanged IR when callback always returns True."""
        mlir = """module {
  func.func @test(%a: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "sf.relu"(%a) : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}"""

        def callback(op, rewriter):
            return True

        result = _run_pattern(mlir, "sf.relu", callback)
        assert "sf.relu" in result
        assert "func.func @test" in result

    def test_fallback_when_sf_import_fails(self):
        """_run_pattern should not crash when sf dialect import raises ImportError."""
        mlir = """module {
  func.func @test(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "sf.mul"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}"""
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "mlir_sf" in name or "_sfDialects" in name:
                msg = f"Simulated: cannot import {name}"
                raise ImportError(msg)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:

            def callback(op, rewriter):
                return True

            result = _run_pattern(mlir, "sf.mul", callback)
            assert "sf.mul" in result
        finally:
            builtins.__import__ = original_import
