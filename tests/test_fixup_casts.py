# ruff: noqa: E501
"""Unit tests for _fixup_unrealized_casts_pass (MLIR bindings version).

Tests parse MLIR text into ir.Module, run the pass, and verify:
- Direct struct casts (bare ptrs → !llvm.struct) are replaced with undef + insertvalue
- Empty IR and no-cast IR pass through unchanged
- Multiple casts in one function all eliminated
- Rank-2 casts with 7 bare args handled correctly
- Fixed IR is valid and re-parseable
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestMLIRFixupPass:
    """Tests for _fixup_unrealized_casts_pass (MLIR bindings version)."""


    @staticmethod
    def _run_pass(mlir_text: str) -> str:
        import mlir.ir as ir
        from compiler.mlir_dialect.fixups import _fixup_unrealized_casts_pass

        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            module = ir.Module.parse(mlir_text, ctx)
            _fixup_unrealized_casts_pass(module)
            return str(module)

    DIRECT_STRUCT_MLIR = """
module {
  llvm.func @test(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64) {
    %0 = builtin.unrealized_conversion_cast %arg0, %arg1, %arg2, %arg3, %arg4 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return
  }
}"""

    def test_direct_struct_cast_eliminated(self):
        fixed = self._run_pass(self.DIRECT_STRUCT_MLIR)
        assert "unrealized_conversion_cast" not in fixed
        assert "llvm.mlir.undef" in fixed
        assert "llvm.insertvalue" in fixed

    def test_output_reparseable(self):
        import mlir.ir as ir

        fixed_text = self._run_pass(self.DIRECT_STRUCT_MLIR)
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ir.Location.unknown(ctx):
            reparsed = ir.Module.parse(fixed_text, ctx)
            assert reparsed is not None

    def test_empty_ir_no_crash(self):
        mlir = """
module {
  llvm.func @empty() {
    llvm.return
  }
}"""
        fixed = self._run_pass(mlir)
        assert "llvm.func @empty" in fixed

    def test_multiple_casts_all_eliminated(self):
        mlir = """
module {
  llvm.func @test(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: i64, %a3: i64, %a4: i64,
                   %b0: !llvm.ptr, %b1: !llvm.ptr, %b2: i64, %b3: i64, %b4: i64) {
    %0 = builtin.unrealized_conversion_cast %a0, %a1, %a2, %a3, %a4 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = builtin.unrealized_conversion_cast %b0, %b1, %b2, %b3, %b4 : !llvm.ptr, !llvm.ptr, i64, i64, i64 to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.return
  }
}"""
        fixed = self._run_pass(mlir)
        assert "unrealized_conversion_cast" not in fixed

    def test_rank2_direct_cast(self):
        mlir = """
module {
  llvm.func @test(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: i64, %a3: i64, %a4: i64,
                   %a5: i64, %a6: i64) {
    %0 = builtin.unrealized_conversion_cast %a0, %a1, %a2, %a3, %a4, %a5, %a6 : !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64 to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.return
  }
}"""
        fixed = self._run_pass(mlir)
        assert "unrealized_conversion_cast" not in fixed
        assert "llvm.insertvalue" in fixed

    def test_no_matching_patterns_unchanged(self):
        mlir = """
module {
  llvm.func @add(%a: i64, %b: i64) -> i64 {
    %0 = llvm.add %a, %b : i64
    llvm.return %0 : i64
  }
}"""
        fixed = self._run_pass(mlir)
        assert "llvm.add" in fixed
        assert "unrealized_conversion_cast" not in fixed
