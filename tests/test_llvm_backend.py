"""Tests for compiler.mlir_dialect.llvm_backend — LLVM IR emission and llc compilation."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# ── Helpers ──────────────────────────────────────────────────────────

_MLIR_TOOLS_OK: bool | None = None


def _tools_available() -> bool:
    global _MLIR_TOOLS_OK
    if _MLIR_TOOLS_OK is not None:
        return _MLIR_TOOLS_OK

    from compiler.mlir_dialect.llvm_backend import _find_llc, _find_mlir_translate

    try:
        _find_llc()
        _find_mlir_translate()
        _MLIR_TOOLS_OK = True
    except RuntimeError:
        _MLIR_TOOLS_OK = False
    return _MLIR_TOOLS_OK


@pytest.fixture(scope="session")
def _mlir_path_setup() -> None:
    _mlir_pkg = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
    if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
        sys.path.insert(0, str(_mlir_pkg))


# ── Unit tests ───────────────────────────────────────────────────────

@pytest.mark.unit
class TestFindTools:

    def test_find_llc(self) -> None:
        from compiler.mlir_dialect.llvm_backend import _find_llc
        llc = _find_llc()
        assert os.path.isfile(llc)
        assert os.access(llc, os.X_OK)

    def test_find_mlir_translate(self) -> None:
        from compiler.mlir_dialect.llvm_backend import _find_mlir_translate
        mt = _find_mlir_translate()
        assert os.path.isfile(mt)
        assert os.access(mt, os.X_OK)

    def test_find_tools_returns_different_binaries(self) -> None:
        from compiler.mlir_dialect.llvm_backend import _find_llc, _find_mlir_translate
        llc = _find_llc()
        mt = _find_mlir_translate()
        assert Path(llc).name != Path(mt).name


@pytest.mark.unit
class TestFixupMLIRForTranslate:

    def test_strips_nuw_from_getelementptr(self) -> None:
        from compiler.mlir_dialect.llvm_backend import _fixup_mlir_for_translate

        mlir = "%0 = llvm.getelementptr inbounds|nuw %ptr[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32\n"
        fixed = _fixup_mlir_for_translate(mlir)
        assert "inbounds|nuw" not in fixed
        assert "inbounds %ptr[%idx]" in fixed

    def test_preserves_overflow_on_add(self) -> None:
        from compiler.mlir_dialect.llvm_backend import _fixup_mlir_for_translate

        mlir = '%0 = llvm.add %a, %b overflow<nsw, nuw> : i64\n'
        fixed = _fixup_mlir_for_translate(mlir)
        assert 'overflow<nsw, nuw>' in fixed

    def test_idempotent(self) -> None:
        from compiler.mlir_dialect.llvm_backend import _fixup_mlir_for_translate

        mlir = "%0 = llvm.getelementptr inbounds %ptr[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, f32\n"
        fixed = _fixup_mlir_for_translate(mlir)
        assert fixed == mlir


@pytest.mark.unit
class TestLLCCompile:

    def test_compile_minimal_ll(self, tmp_path: Path) -> None:
        if not _tools_available():
            pytest.skip("llc/mlir-translate not available")

        from compiler.mlir_dialect.llvm_backend import llc_compile

        minimal_ll = """; ModuleID = 'minimal'
target triple = "x86_64-apple-macosx10.15.0"
define i32 @answer() {
  ret i32 42
}
"""
        ll_path = str(tmp_path / "minimal.ll")
        Path(ll_path).write_text(minimal_ll)

        obj_path = llc_compile(ll_path)
        assert os.path.isfile(obj_path)
        assert obj_path.endswith(".o")
        assert os.path.getsize(obj_path) > 0

    def test_custom_output_path(self, tmp_path: Path) -> None:
        if not _tools_available():
            pytest.skip("llc/mlir-translate not available")

        from compiler.mlir_dialect.llvm_backend import llc_compile

        minimal_ll = """; ModuleID = 'minimal'
target triple = "x86_64-apple-macosx10.15.0"
define i32 @answer() {
  ret i32 42
}
"""
        ll_path = str(tmp_path / "minimal.ll")
        Path(ll_path).write_text(minimal_ll)

        custom = str(tmp_path / "custom.out")
        result = llc_compile(ll_path, output=custom)
        assert result == custom
        assert os.path.getsize(custom) > 0

    def test_bad_ir_raises(self, tmp_path: Path) -> None:
        if not _tools_available():
            pytest.skip("llc/mlir-translate not available")

        from compiler.mlir_dialect.llvm_backend import llc_compile

        bad_path = str(tmp_path / "not.ll")
        Path(bad_path).write_text("this is not valid LLVM IR")

        from compiler.exceptions import LLCError

        with pytest.raises(LLCError, match="llc compilation failed"):
            llc_compile(bad_path)


@pytest.mark.unit
class TestLinkDylib:

    def test_link_minimal_o_to_dylib(self, tmp_path: Path) -> None:
        if not _tools_available():
            pytest.skip("llc/mlir-translate not available")

        from compiler.mlir_dialect.llvm_backend import link_dylib, llc_compile

        minimal_ll = """; ModuleID = 'minimal'
target triple = "x86_64-apple-macosx10.15.0"
define i32 @answer() {
  ret i32 42
}
"""
        ll_path = str(tmp_path / "minimal.ll")
        Path(ll_path).write_text(minimal_ll)
        obj_path = llc_compile(ll_path)

        dylib_path = str(tmp_path / "libtest.dylib")
        result = link_dylib([obj_path], dylib_path)
        assert result == dylib_path
        assert os.path.getsize(dylib_path) > 0

    def test_link_multiple_o_files(self, tmp_path: Path) -> None:
        if not _tools_available():
            pytest.skip("llc/mlir-translate not available")

        from compiler.mlir_dialect.llvm_backend import link_dylib, llc_compile

        ll1 = """; ModuleID = 'a'
target triple = "x86_64-apple-macosx10.15.0"
define i32 @a() { ret i32 1 }
"""
        ll2 = """; ModuleID = 'b'
target triple = "x86_64-apple-macosx10.15.0"
define i32 @b() { ret i32 2 }
"""
        for i, ll in enumerate([ll1, ll2]):
            Path(str(tmp_path / f"part{i}.ll")).write_text(ll)
        o1 = llc_compile(str(tmp_path / "part0.ll"))
        o2 = llc_compile(str(tmp_path / "part1.ll"))

        dylib_path = str(tmp_path / "libmulti.dylib")
        result = link_dylib([o1, o2], dylib_path)
        assert os.path.getsize(result) > 0


# ── Integration tests ────────────────────────────────────────────────

@pytest.mark.integration
class TestQwenMain2FullPipeline:

    def _run_pipeline_on_main2(self, ir_ctx: Any) -> str:
        """Run sf→linalg→LLVM pipeline on Qwen main_2, return LLVM IR text."""
        from compiler.mlir_artifact import MlirModule, mlir_module_to_ir_module
        from compiler.mlir_dialect.llvm_backend import lower_linalg_to_llvm_ir, mlir_module_to_llvm_ir
        from compiler.pipeline import _apply_sf_to_linalg as sf_to_linalg_pass_on_module
        from compiler.serialize import load_artifact

        compiled_path = Path("compiled/qwen3_0.8b/model.mlir")
        if not compiled_path.is_file():
            pytest.skip("Qwen model not compiled — run: python scripts/compile.py qwen")

        mod = load_artifact(str(Path("compiled/qwen3_0.8b")))
        func = mod.functions[2]  # main_2: 500 compute ops, 0 weights
        ir_mod = mlir_module_to_ir_module(MlirModule(functions=[func]), ctx=ir_ctx)
        sf_to_linalg_pass_on_module(ir_mod)
        lower_linalg_to_llvm_ir(ir_mod)
        return mlir_module_to_llvm_ir(ir_mod)

    def test_pipeline_produces_valid_llvm_ir(self, _mlir_path_setup: None, mlir_context: Any) -> None:
        llvm_ir = self._run_pipeline_on_main2(mlir_context)
        assert llvm_ir, "Expected non-empty LLVM IR"
        assert "llvm.func" not in llvm_ir, "LLVM IR should not contain MLIR syntax"
        assert re.search(r"define\s.*@main_2", llvm_ir), "Expected main_2 function"

    def test_emit_ll_file_and_compile_to_object(
        self, _mlir_path_setup: None, mlir_context: Any, tmp_path: Path
    ) -> None:
        if not _tools_available():
            pytest.skip("llc/mlir-translate not available")

        from compiler.mlir_dialect.llvm_backend import llc_compile

        llvm_ir = self._run_pipeline_on_main2(mlir_context)
        ll_path = str(tmp_path / "main2.ll")
        Path(ll_path).write_text(llvm_ir)

        obj_path = llc_compile(ll_path)
        assert os.path.isfile(obj_path)
        assert os.path.getsize(obj_path) > 0, f".o file is empty: {obj_path}"

    def test_llvm_ir_contains_expected_instructions(self, _mlir_path_setup: None, mlir_context: Any) -> None:
        llvm_ir = self._run_pipeline_on_main2(mlir_context)
        assert "alloca" in llvm_ir or "load" in llvm_ir or "getelementptr" in llvm_ir
        assert "ret" in llvm_ir

    def test_compile_to_dylib(self, _mlir_path_setup: None, mlir_context: Any, tmp_path: Path) -> None:
        if not _tools_available():
            pytest.skip("llc/mlir-translate not available")

        import mlir.ir as ir

        from compiler.mlir_artifact import MlirModule, mlir_module_to_ir_module
        from compiler.mlir_dialect.llvm_backend import compile_mlir_to_dylib, lower_linalg_to_llvm_ir
        from compiler.pipeline import _apply_sf_to_linalg as sf_to_linalg_pass_on_module
        from compiler.serialize import load_artifact

        compiled_path = Path("compiled/qwen3_0.8b/model.mlir")
        if not compiled_path.is_file():
            pytest.skip("Qwen model not compiled")

        mod = load_artifact(str(Path("compiled/qwen3_0.8b")))
        func = mod.functions[2]
        ir_mod = mlir_module_to_ir_module(MlirModule(functions=[func]), ctx=mlir_context)

        def _add_emit_c_interface(op):
            if hasattr(op, "name") and op.name == "func.func":
                with mlir_context:
                    op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            return ir.WalkResult.ADVANCE

        ir_mod.operation.walk(_add_emit_c_interface)
        sf_to_linalg_pass_on_module(ir_mod)
        lower_linalg_to_llvm_ir(ir_mod)

        dylib_path = str(tmp_path / "libmain2.dylib")
        compile_mlir_to_dylib(ir_mod, dylib_path)
        assert os.path.isfile(dylib_path)
        assert os.path.getsize(dylib_path) > 0

        # Verify symbol exists
        result = subprocess.run(
            ["nm", "-g", dylib_path], capture_output=True, text=True
        )
        assert "_main_2" in result.stdout or "__mlir_ciface_main_2" in result.stdout
