"""Exploration: Test MLIR ExecutionEngine JIT viability.

This script tests whether MLIR's ExecutionEngine can be used to
bypass llc for output comparison — helping isolate whether the
cos=0.850 degradation comes from llc or earlier MLIR lowering.

Goal: JIT-compile the lowered model (or a simplified version) via
ExecutionEngine instead of going through llc + ld → .dylib.

Method:
  1. Test with a tiny matmul module (known to work)
  2. Test with model.lowered.mlir through the full lowering pipeline
  3. If JIT works, compare output against ctypes dylib oracle
  4. Document blockers if any

Usage:
  source .venv/bin/activate
  python scripts/test_jit_exploration.py
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Environment setup ────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
COMPILED = REPO / "compiled" / "opt_125m_fresh"

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.pop("CONDA_PREFIX", None)


def _setup_mlir_path() -> None:
    _mlir_pkg = REPO / "mlir_binding" / "mlir_package"
    if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
        sys.path.insert(0, str(_mlir_pkg))


_setup_mlir_path()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    return float(
        np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12)
    )


# ── Memref helpers ──────────────────────────────────────────────────


def _make_memref_arg(arr: np.ndarray) -> tuple[Any, Any]:
    """Build a double-pointer memref argument for ExecutionEngine.invoke()."""
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


def add_emit_c_interface(module, ctx):
    """Add llvm.emit_c_interface to all func.func ops."""
    import mlir.ir as ir

    def _cb(op):
        if hasattr(op, "name") and op.name == "func.func":
            with ctx:
                op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        return ir.WalkResult.ADVANCE

    module.operation.walk(_cb)


# ── Test 1: Tiny matmul (full pipeline through ExecutionEngine) ─────


def test_tiny_matmul_jit():
    """Test JIT compilation of a tiny matmul via ExecutionEngine."""
    print("\n" + "=" * 70)
    print("TEST 1: Tiny matmul via ExecutionEngine JIT")
    print("=" * 70)

    import mlir.ir as ir
    import mlir.passmanager as pm
    from mlir._mlir_libs import _mlirRegisterEverything
    from mlir.execution_engine import ExecutionEngine

    # Create context with all dialects
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx.append_dialect_registry(reg)

    # Use minimal matmul WITHOUT fill (pure linalg.matmul with init tensor arg)
    # fill requires extra constant definitions that complicate parsing
    print("  Creating minimal matmul module...")
    with ir.Location.unknown(ctx):
        module = ir.Module.parse(
            """module {
  func.func @main(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>,"""
            """ %init: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>,"""
            """ tensor<8x4xf32>) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}""",
            ctx,
        )

    # Add emit_c_interface for JIT compatibility
    add_emit_c_interface(module, ctx)

    # Run the same lowering pipeline as the model
    print("  Running lowering pipeline...")
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
    try:
        pman = pm.PassManager.parse(pipeline, ctx)
        pman.run(module.operation)
        print(f"    Lowering succeeded. Output:\n{str(module)[:500]}...")
    except Exception as e:
        print(f"    ❌ Lowering failed: {e}")
        return False

    # JIT compile with ExecutionEngine
    print("\n  Creating ExecutionEngine...")
    try:
        t0 = time.time()
        engine = ExecutionEngine(module, opt_level=0)
        dt = time.time() - t0
        print(f"    ✅ ExecutionEngine created in {dt:.3f}s")
    except Exception as e:
        print(f"    ❌ ExecutionEngine creation failed: {e}")
        return False

    # Test with random data
    print("\n  Running JIT matmul...")
    rng = np.random.RandomState(42)
    a = rng.randn(4, 8).astype(np.float32)
    b = rng.randn(8, 4).astype(np.float32)
    expected = a @ b

    try:
        a_inner, a_outer = _make_memref_arg(a)
        b_inner, b_outer = _make_memref_arg(b)
        r_inner, r_outer = _make_memref_arg(np.zeros((4, 4), dtype=np.float32))

        engine.invoke("main", r_outer, a_outer, b_outer)
        out = _read_memref_output(r_inner, np.zeros((4, 4), dtype=np.float32))

        cos = cosine_similarity(out, expected)
        max_diff = np.max(np.abs(out - expected))
        print(f"    cos(JIT, numpy): {cos:.10f}")
        print(f"    max diff:        {max_diff:.8f}")
        print(f"    out shape:       {out.shape}")

        if cos > 0.9999:
            print("    ✅ JIT matmul matches numpy reference")
            return True
        else:
            print(f"    ❌ JIT output differs (cos={cos:.6f})")
            return False
    except Exception as e:
        print(f"    ❌ JIT invoke failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── Test 2: Load model.lowered.mlir and JIT ─────────────────────────


def test_model_lowered_jit():
    """Try to load model.lowered.mlir through pipeline to ExecutionEngine."""
    print("\n" + "=" * 70)
    print("TEST 2: Load model.lowered.mlir → lower → ExecutionEngine")
    print("=" * 70)

    import mlir.ir as ir
    from mlir._mlir_libs import _mlirRegisterEverything
    from mlir.execution_engine import ExecutionEngine

    lowered_path = COMPILED / "model.lowered.mlir"
    if not lowered_path.exists():
        print(f"  ❌ {lowered_path} not found")
        return False

    lowered_text = lowered_path.read_text()
    n_funcs = lowered_text.count("func.func")
    n_ops = len(lowered_text)
    print(f"  Loaded model.lowered.mlir: {n_funcs} functions, {n_ops} chars")

    # Create context
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx.append_dialect_registry(reg)

    # Parse the lowered module
    with ir.Location.unknown(ctx):
        module = ir.Module.parse(lowered_text, ctx)
        print(f"  Parsed {len(list(module.body))} blocks")

    # Run the full lower_linalg_to_llvm_ir pipeline
    print("  Running lower_linalg_to_llvm_ir pipeline...")
    try:
        from compiler.mlir_dialect.llvm_backend import lower_linalg_to_llvm_ir
        t0 = time.time()
        llvm_text = lower_linalg_to_llvm_ir(module)
        dt = time.time() - t0
        print(f"    Pipeline completed in {dt:.1f}s")
        print(f"    Output LLVM IR: {len(llvm_text)} chars")
    except Exception as e:
        print(f"    ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Re-parse the LLVM dialect module
    llvm_ctx = ir.Context()
    llvm_ctx.allow_unregistered_dialects = True
    with ir.Location.unknown(llvm_ctx):
        llvm_mod = ir.Module.parse(llvm_text, llvm_ctx)
        add_emit_c_interface(llvm_mod, llvm_ctx)

        # Run func-to-llvm + reconcile-unrealized-casts
        import mlir.passmanager as pm
        try:
            pm.PassManager.parse(
                "builtin.module(convert-func-to-llvm,reconcile-unrealized-casts)",
                llvm_ctx,
            ).run(llvm_mod.operation)
            print("    ✅ func-to-llvm + casts reconciled")
        except Exception as e:
            print(f"    ❌ func-to-llvm failed: {e}")
            return False

        # Count functions
        mod_str = str(llvm_mod)
        n_funcs_llvm = mod_str.count("llvm.func @")
        n_funcs_total = mod_str.count("func @")
        print(f"    LLVM functions: {n_funcs_llvm}")
        print(f"    Total functions: {n_funcs_total}")

        # Try creating ExecutionEngine
        print("  Creating ExecutionEngine (this may take a while)...")
        try:
            t0 = time.time()
            ExecutionEngine(llvm_mod, opt_level=0)
            dt = time.time() - t0
            print(f"    ✅ ExecutionEngine created in {dt:.1f}s")
            print("    JIT compilation of full model WORKS")
            print()
            print("    NOTE: The model's main_0 function has ~215 memref args.")
            print("    Calling it via JIT requires constructing all those args")
            print("    as ctypes memref descriptors — extremely complex.")
            print()
            print("    But the JIT CAN compile and is available.")
            print("    Next step would be layer-by-layer JIT testing.")
            return True
        except Exception as e:
            print(f"    ❌ ExecutionEngine creation failed: {e}")
            return False


# ── Test 3: Compare JIT vs dylib for a single matmul op ─────────────


def test_single_matmul_jit_vs_dylib():
    """Compare JIT output vs dylib output for a single matmul function."""
    print("\n" + "=" * 70)
    print("TEST 3: JIT vs dylib for single matmul (if dylib exists)")
    print("=" * 70)

    dylib_path = COMPILED / "libopt_125m.dylib"
    if not dylib_path.exists():
        print(f"  ⚠️  No dylib found at {dylib_path}, skipping")
        print("  To create one: python scripts/compile_dylib.py compiled/opt_125m_fresh")
        return None

    # We can't easily extract a single function from the dylib for comparison
    # since the dylib has a single monolithic main_0 function.
    # This test would require building a small standalone matmul dylib.
    print("  The .dylib contains a single monolithic main_0 function.")
    print("  A fair JIT vs dylib comparison requires either:")
    print("    a) A standalone matmul → dylib compilation, OR")
    print("    b) Splitting the model into per-layer dylibs")
    print()
    print("  Building a standalone matmul dylib for comparison...")

    # Build a standalone matmul through the full compile_pipeline
    import mlir.ir as ir
    import mlir.passmanager as pm
    from mlir._mlir_libs import _mlirRegisterEverything

    from compiler.mlir_dialect.compile_utils import (
        compile_mlir_to_dylib,
    )

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx.append_dialect_registry(reg)

    with ir.Location.unknown(ctx):
        # Same matmul as Test 1, but as a separate function
        module = ir.Module.parse(
            """module {
  func.func @matmul_4x8(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %cst = arith.constant 0.0 : f32
    %0 = tensor.empty() : tensor<4x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x8xf32>,"""
            """ tensor<8x4xf32>) outs(%1 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
  }
}""",
            ctx,
        )

    # Lower to LLVM dialect
    pipeline = (
        "builtin.module("
        "canonicalize,cse,"
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
    try:
        pman = pm.PassManager.parse(pipeline, ctx)
        pman.run(module.operation)
        print("    ✅ LLVM lowering succeeded")
    except Exception as e:
        print(f"    ❌ LLVM lowering failed: {e}")
        return None

    # JIT path: ExecutionEngine
    print("  --- JIT path ---")
    add_emit_c_interface(module, ctx)
    try:
        from mlir.execution_engine import ExecutionEngine

        t0 = time.time()
        engine = ExecutionEngine(module, opt_level=0)
        jit_time = time.time() - t0
        print(f"    ✅ ExecutionEngine created in {jit_time:.3f}s")
    except Exception as e:
        print(f"    ❌ ExecutionEngine creation failed: {e}")
        return None

    # Run JIT
    rng = np.random.RandomState(42)
    a = rng.randn(4, 8).astype(np.float32)
    b = rng.randn(8, 4).astype(np.float32)
    expected = a @ b

    try:
        a_inner, a_outer = _make_memref_arg(a)
        b_inner, b_outer = _make_memref_arg(b)
        r_inner, r_outer = _make_memref_arg(np.zeros((4, 4), dtype=np.float32))

        t0 = time.time()
        engine.invoke("matmul_4x8", r_outer, a_outer, b_outer)
        jit_run_time = time.time() - t0
        jit_out = _read_memref_output(r_inner, np.zeros((4, 4), dtype=np.float32))
        jit_cos = cosine_similarity(jit_out, expected)
        print(f"    JIT run time: {jit_run_time*1000:.2f}ms")
        print(f"    JIT cos vs numpy: {jit_cos:.10f}")
    except Exception as e:
        print(f"    ❌ JIT invoke failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # dylib path: compile to .dylib and compare
    print("  --- dylib path ---")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        dylib_out = os.path.join(td, "libmatmul_test.dylib")
        try:
            t0 = time.time()
            compile_mlir_to_dylib(module, dylib_out, opt_level=0)
            dylib_compile_time = time.time() - t0
            print(f"    ✅ dylib compiled in {dylib_compile_time:.1f}s")
        except Exception as e:
            print(f"    ❌ dylib compilation failed: {e}")
            return None

        # Load dylib via ctypes and run
        try:
            lib = ctypes.CDLL(dylib_out)

            # Define the matmul_4x8 function signature
            # After memref lowering, it takes bare pointers
            # This is tricky — let's try invoking the ciface wrapper
            func = getattr(lib, "_mlir_ciface_matmul_4x8", None)
            if func is None:
                print("    ❌ No _mlir_ciface_matmul_4x8 function found")
                # Try raw function
                func = getattr(lib, "matmul_4x8", None)
                if func is None:
                    print("    ❌ No matmul_4x8 function found either")
                    print(f"    Available symbols: {len(dir(lib))} total")
                    return None

            # Build memref args for dylib
            a_inner2, a_outer2 = _make_memref_arg(a)
            b_inner2, b_outer2 = _make_memref_arg(b)
            r_inner2, r_outer2 = _make_memref_arg(np.zeros((4, 4), dtype=np.float32))

            # Invoke via ciface
            t0 = time.time()
            func(r_outer2, a_outer2, b_outer2)
            dylib_run_time = time.time() - t0
            dylib_out_arr = _read_memref_output(
                r_inner2, np.zeros((4, 4), dtype=np.float32)
            )
            dylib_cos = cosine_similarity(dylib_out_arr, expected)
            print(f"    dylib run time: {dylib_run_time*1000:.2f}ms")
            print(f"    dylib cos vs numpy: {dylib_cos:.10f}")

            # Compare JIT vs dylib
            cos_jv = cosine_similarity(jit_out, dylib_out_arr)
            print("\n  🔍 JIT vs dylib comparison:")
            print(f"     cos(JIT, dylib): {cos_jv:.10f}")
            diff = np.max(np.abs(jit_out - dylib_out_arr))
            print(f"     max abs diff:    {diff:.8f}")

            if cos_jv > 0.9999:
                print("  ✅ JIT == dylib — llc does NOT introduce error for matmul")
            elif cos_jv > 0.99:
                print(f"  ⚠️  JIT ≈ dylib — small difference (cos={cos_jv:.6f})")
            else:
                print("  ❌ JIT ≠ dylib — llc introduces significant error")

        except Exception as e:
            print(f"    ❌ dylib run failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    return True


# ── Main ─────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("MLIR ExecutionEngine JIT Exploration")
    print("=" * 70)
    print(f"Repo: {REPO}")
    print(f"Compiled artifacts: {COMPILED}")

    results = {}

    # Test 1: Tiny matmul via ExecutionEngine
    r1 = test_tiny_matmul_jit()
    results["tiny_matmul_jit"] = r1

    # Test 2: model.lowered.mlir through pipeline to ExecutionEngine
    if r1:  # only try if basic JIT works
        r2 = test_model_lowered_jit()
        results["model_lowered_jit"] = r2
    else:
        print("\n  ⏭️  Skipping Test 2 (Test 1 failed)")
        results["model_lowered_jit"] = None

    # Test 3: JIT vs dylib comparison
    if r1:
        r3 = test_single_matmul_jit_vs_dylib()
        results["jit_vs_dylib"] = r3
    else:
        print("\n  ⏭️  Skipping Test 3 (Test 1 failed)")
        results["jit_vs_dylib"] = None

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test, result in results.items():
        status = "✅" if result is True else ("❌" if result is False else "⏭️")
        print(f"  {status} {test}")
    print()


if __name__ == "__main__":
    main()
