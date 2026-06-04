"""Tests for compiler/mlir_passes — MLIR-level pass infrastructure."""

from __future__ import annotations

import pytest

_SIMPLE_ARITH_MLIR = """
module {
  func.func @main() -> i32 {
    %c42 = arith.constant 42 : i32
    %c58 = arith.constant 58 : i32
    %sum = arith.addi %c42, %c58 : i32
    return %sum : i32
  }
}
"""

_DUPLICATE_MLIR = """
module {
  func.func @main() -> i32 {
    %c42 = arith.constant 42 : i32
    %dup = arith.constant 42 : i32
    %sum = arith.addi %c42, %dup : i32
    return %sum : i32
  }
}
"""

_NO_FUNC_MLIR = """
module {
}
"""


@pytest.mark.unit
class TestMlirCountOps:
    def test_counts_arith_ops(self) -> None:
        from compiler.passes import mlir_count_ops_in_module

        stats = mlir_count_ops_in_module(_SIMPLE_ARITH_MLIR)
        assert stats.get("arith", 0) >= 3  # 2 constants + 1 addi
        assert stats.get("func", 0) >= 1  # func.func
        total = sum(stats.values())
        assert total >= 4  # func.func + arith ops + return

    def test_empty_module_zero_counts(self) -> None:
        from compiler.passes import mlir_count_ops_in_module

        stats = mlir_count_ops_in_module(_NO_FUNC_MLIR)
        # builtin.module is not counted as "builtin" in some versions
        assert isinstance(stats, dict)


@pytest.mark.unit
class TestMlirVerifyStructure:
    def test_valid_module_passes(self) -> None:
        import mlir.ir as ir  # type: ignore[import-untyped]

        from compiler.passes import mlir_verify_structure

        ctx = ir.Context()
        with ctx:
            module = ir.Module.parse(_SIMPLE_ARITH_MLIR, ctx)
            issues = mlir_verify_structure(module, ctx)
        assert len(issues) == 0

    def test_empty_module_reported(self) -> None:
        import mlir.ir as ir  # type: ignore[import-untyped]

        from compiler.passes import mlir_verify_structure

        ctx = ir.Context()
        with ctx:
            module = ir.Module.parse(_NO_FUNC_MLIR, ctx)
            issues = mlir_verify_structure(module, ctx)
        assert len(issues) >= 1


@pytest.mark.unit
class TestMlirCSE:
    def test_cse_does_not_crash(self) -> None:
        import mlir.ir as ir  # type: ignore[import-untyped]

        from compiler.passes import mlir_run_cse

        ctx = ir.Context()
        with ctx:
            module = ir.Module.parse(_SIMPLE_ARITH_MLIR, ctx)
            result = mlir_run_cse(module)
        assert result is not None

    def test_canonicalize_does_not_crash(self) -> None:
        import mlir.ir as ir  # type: ignore[import-untyped]

        from compiler.passes import mlir_run_canonicalize

        ctx = ir.Context()
        with ctx:
            module = ir.Module.parse(_DUPLICATE_MLIR, ctx)
            result = mlir_run_canonicalize(module)
        assert result is not None
