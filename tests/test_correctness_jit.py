"""Tests for correctness: LLVM JIT output verification.

Verifies the full AOT compilation pipeline (SF→linalg→LLVM) produces
numerically correct results (cosine similarity > 0.999) when executed
via the MLIR ExecutionEngine JIT.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import pytest

from compiler.mlir_dialect.compile_utils import _setup_mlir_path
from scripts._cos import cosine_similarity as _cosine_similarity


def _lower_to_llvm(module: Any) -> Any:
    """Run MLIR bufferize → LLVM passes."""
    import mlir.passmanager as pm

    pipeline = (
        "builtin.module("
        "func.func(linalg-fuse-elementwise-ops),"
        "canonicalize,"
        "cse,"
        "one-shot-bufferize{bufferize-function-boundaries},"
        "convert-linalg-to-loops,"
        "lower-affine,"
        "convert-scf-to-cf,"
        "expand-strided-metadata,"
        "finalize-memref-to-llvm,"
        "convert-cf-to-llvm,"
        "convert-math-to-llvm,"
        "convert-arith-to-llvm,"
        "convert-func-to-llvm,"
        "reconcile-unrealized-casts"
        ")"
    )
    ctx = module.operation.context
    ctx.allow_unregistered_dialects = True
    pman = pm.PassManager.parse(pipeline, ctx)
    pman.run(module.operation)
    return module


# ── Memref wrappers ──────────────────────────────────────────────────


def _make_memref_arg(arr: np.ndarray) -> tuple[Any, Any]:
    """Build a double-pointer memref argument for ExecutionEngine.invoke().

    Returns ``(inner_ptr, outer_ptr)``:
    * ``inner_ptr``  — ``ctypes.pointer(descriptor)``, kept alive for the call.
    * ``outer_ptr``  — ``ctypes.pointer(inner_ptr)``, passed to invoke().
    """
    from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

    desc = get_ranked_memref_descriptor(arr)
    inner = ctypes.pointer(desc)
    outer = ctypes.pointer(inner)
    return inner, outer


def _read_memref_output(inner_ptr: Any, arr_dummy: np.ndarray) -> np.ndarray:
    """Read output from a memref descriptor that was written by the JIT."""
    from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

    clone = get_ranked_memref_descriptor(arr_dummy)
    ctypes.memmove(
        ctypes.addressof(clone),
        ctypes.cast(inner_ptr, ctypes.c_void_p).value,
        ctypes.sizeof(clone),
    )
    return np.ctypeslib.as_array(clone.aligned, shape=tuple(clone.shape)).copy()


# ── Pytest fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mlir_ctx() -> Any:
    import mlir.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    return ctx


@pytest.fixture(scope="session", autouse=True)
def _mlir_path_setup_fixture() -> None:
    _setup_mlir_path()


# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestSFSmokeJIT:

    @pytest.mark.skip(reason="#46: JIT path outdated after C++ lowering migration")
    def test_add_relu_mul_via_jit(self, mlir_ctx: Any) -> None:
        import mlir.ir as ir
        from mlir.execution_engine import ExecutionEngine

        from compiler.pipeline import _apply_sf_to_linalg as sf_to_linalg_pass_on_module

        with ir.Location.unknown(mlir_ctx):
            module = ir.Module.parse(
                """module {
  func.func @f(%a: tensor<2x64xf32>, %b: tensor<2x64xf32>, %c: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = "sf.add"(%a, %b) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
    %1 = "sf.relu"(%0) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    %2 = "sf.mul"(%1, %c) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
    return %2 : tensor<2x64xf32>
  }
}""",
                mlir_ctx,
            )
        _add_emit_c_interface(module, mlir_ctx)
        sf_to_linalg_pass_on_module(module)
        _lower_to_llvm(module)

        engine = ExecutionEngine(module, opt_level=0)

        rng = np.random.RandomState(42)
        a = rng.randn(2, 64).astype(np.float32)
        b = rng.randn(2, 64).astype(np.float32)
        c = rng.randn(2, 64).astype(np.float32)
        expected = np.maximum(a + b, 0.0) * c

        a_inner, a_outer = _make_memref_arg(a)
        b_inner, b_outer = _make_memref_arg(b)
        c_inner, c_outer = _make_memref_arg(c)
        r_inner, r_outer = _make_memref_arg(np.zeros((2, 64), dtype=np.float32))

        engine.invoke("f", r_outer, a_outer, b_outer, c_outer)
        out = _read_memref_output(r_inner, np.zeros((2, 64), dtype=np.float32))

        assert _cosine_similarity(out, expected) > 0.9999

    @pytest.mark.skip(reason="#46: JIT path outdated after C++ lowering migration")
    def test_matmul_jit_matches_numpy(self, mlir_ctx: Any) -> None:
        import mlir.ir as ir
        from mlir.execution_engine import ExecutionEngine

        from compiler.pipeline import _apply_sf_to_linalg as sf_to_linalg_pass_on_module

        with ir.Location.unknown(mlir_ctx):
            module = ir.Module.parse(
                """module {
  func.func @f(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %0 = "sf.matmul"(%a, %b) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}""",
                mlir_ctx,
            )
        _add_emit_c_interface(module, mlir_ctx)
        sf_to_linalg_pass_on_module(module)
        _lower_to_llvm(module)

        engine = ExecutionEngine(module, opt_level=0)

        rng = np.random.RandomState(3)
        a = rng.randn(4, 8).astype(np.float32)
        b = rng.randn(8, 4).astype(np.float32)
        expected = a @ b

        a_inner, a_outer = _make_memref_arg(a)
        b_inner, b_outer = _make_memref_arg(b)
        r_inner, r_outer = _make_memref_arg(np.zeros((4, 4), dtype=np.float32))

        engine.invoke("f", r_outer, a_outer, b_outer)
        out = _read_memref_output(r_inner, np.zeros((4, 4), dtype=np.float32))

        assert out.shape == (4, 4)
        assert np.max(np.abs(out - expected)) < 1e-5
        assert _cosine_similarity(out, expected) > 0.9999

    @pytest.mark.skip(reason="#46: JIT path outdated after C++ lowering migration")
    def test_multiple_outputs_via_jit(self, mlir_ctx: Any) -> None:
        """JIT a function returning 2 tensors — the packed result struct.

        Uses only ops that do NOT require ``memrefCopy`` (no linalg.copy).
        """
        import mlir.ir as ir
        from mlir.execution_engine import ExecutionEngine

        from compiler.pipeline import _apply_sf_to_linalg as sf_to_linalg_pass_on_module

        with ir.Location.unknown(mlir_ctx):
            module = ir.Module.parse(
                """module {
  func.func @f(%a: tensor<2x64xf32>) -> (tensor<2x64xf32>, tensor<2x64xf32>) {
    %0 = "sf.add"(%a, %a) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
    %1 = "sf.mul"(%0, %a) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
    %2 = "sf.relu"(%1) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0, %2 : tensor<2x64xf32>, tensor<2x64xf32>
  }
}""",
                mlir_ctx,
            )
        _add_emit_c_interface(module, mlir_ctx)
        sf_to_linalg_pass_on_module(module)
        _lower_to_llvm(module)

        engine = ExecutionEngine(module, opt_level=0)

        rng = np.random.RandomState(11)
        a = rng.randn(2, 64).astype(np.float32)
        expected0 = a + a

        a_inner, a_outer = _make_memref_arg(a)
        r_inner, r_outer = _make_memref_arg(np.zeros((2, 64), dtype=np.float32))

        engine.invoke("f", r_outer, a_outer)
        # The result struct packs both outputs; the first output occupies
        # the first MemRefDescriptor slot.
        out = _read_memref_output(r_inner, np.zeros((2, 64), dtype=np.float32))

        assert out.shape == expected0.shape
        assert _cosine_similarity(out, expected0) > 0.9999


# ── Helper ───────────────────────────────────────────────────────────


def _add_emit_c_interface(module: Any, ctx: Any) -> None:
    import mlir.ir as ir

    def _cb(op: Any) -> Any:
        if hasattr(op, "name") and op.name == "func.func":
            with ctx:
                op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        return ir.WalkResult.ADVANCE

    module.operation.walk(_cb)
