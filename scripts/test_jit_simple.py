"""Minimal ExecutionEngine test — isolate JIT issues from pipeline issues."""
import ctypes
import os
import time
import traceback

import numpy as np

MLIR_RUNNER_UTILS = os.getenv("MLIR_RUNNER_UTILS", "")
MLIR_C_RUNNER_UTILS = os.getenv("MLIR_C_RUNNER_UTILS", "")

from mlir.execution_engine import ExecutionEngine  # noqa: E402
from mlir.ir import Context, Location, Module, UnitAttr, WalkResult  # noqa: E402
from mlir.passmanager import PassManager  # noqa: E402

ctx = Context()
ctx.allow_unregistered_dialects = True

def lower_to_llvm(module):
    pipeline = "builtin.module(" \
        "finalize-memref-to-llvm," \
        "convert-func-to-llvm," \
        "convert-arith-to-llvm," \
        "convert-cf-to-llvm," \
        "reconcile-unrealized-casts)"
    pm = PassManager.parse(pipeline, ctx)
    pm.run(module.operation)
    return module

# Test 1: Simple void function with llvm.emit_c_interface
print("=" * 60)
print("TEST 1: Void function — basic ExecutionEngine smoke test")
print("=" * 60)
with Location.unknown(ctx):
    module = Module.parse(r"""
func.func @void_func() attributes { llvm.emit_c_interface } {
  return
}
""")
    lower_to_llvm(module)
    print(f"  Lowered module ({len(str(module))} chars)")
    try:
        ee = ExecutionEngine(module)
        ee.invoke("void_func")
        print("  ✅ ExecutionEngine created and invoked")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

# Test 2: Simple float add (scalar, not tensor)
print("\n" + "=" * 60)
print("TEST 2: Scalar float add — basic argument passing")
print("=" * 60)
with Location.unknown(ctx):
    module = Module.parse(r"""
func.func @add(%arg0: f32, %arg1: f32) -> f32 attributes { llvm.emit_c_interface } {
  %add = arith.addf %arg0, %arg1 : f32
  return %add : f32
}
""")
    lower_to_llvm(module)
    print(f"  Lowered module ({len(str(module))} chars)")
    try:
        ee = ExecutionEngine(module)
        c_float_p = ctypes.c_float * 1
        arg0 = c_float_p(42.0)
        arg1 = c_float_p(2.0)
        res = c_float_p(-1.0)
        ee.invoke("add", arg0, arg1, res)
        print(f"  ✅ {arg0[0]} + {arg1[0]} = {res[0]}")
        assert abs(res[0] - 44.0) < 1e-6
    except Exception as e:
        print(f"  ❌ Failed: {e}")

# Test 3: Memref-based matmul (closest to our model's JIT needs)
print("\n" + "=" * 60)
print("TEST 3: Memref-based matmul via ExecutionEngine")
print("=" * 60)
with Location.unknown(ctx):
    module = Module.parse(r"""
func.func @matmul(%arg0: memref<4x8xf32>, %arg1: memref<8x4xf32>,"""
    r""" %arg2: memref<4x4xf32>) attributes { llvm.emit_c_interface } {
  linalg.matmul ins(%arg0, %arg1 : memref<4x8xf32>, memref<8x4xf32>) outs(%arg2 : memref<4x4xf32>) -> ()
  return
}
""")
    # For linalg on memref, need convert-linalg-to-loops + lower-affine + scf-to-cf first
    pipeline = "builtin.module(" \
        "convert-linalg-to-loops," \
        "lower-affine," \
        "convert-scf-to-cf," \
        "expand-strided-metadata," \
        "finalize-memref-to-llvm," \
        "convert-func-to-llvm," \
        "convert-arith-to-llvm," \
        "convert-cf-to-llvm," \
        "reconcile-unrealized-casts)"
    pm = PassManager.parse(pipeline, ctx)
    pm.run(module.operation)
    print(f"  Lowered module ({len(str(module))} chars)")
    try:
        ee = ExecutionEngine(module)
        print("  ✅ ExecutionEngine created")

        # Build memref args
        from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
        rng = np.random.RandomState(42)
        a = rng.randn(4, 8).astype(np.float32)
        b = rng.randn(8, 4).astype(np.float32)
        init = np.zeros((4, 4), dtype=np.float32)

        a_desc = get_ranked_memref_descriptor(a)
        b_desc = get_ranked_memref_descriptor(b)
        r_desc = get_ranked_memref_descriptor(init)

        a_inner = ctypes.pointer(a_desc)
        b_inner = ctypes.pointer(b_desc)
        r_inner = ctypes.pointer(r_desc)

        a_outer = ctypes.pointer(a_inner)
        b_outer = ctypes.pointer(b_inner)
        r_outer = ctypes.pointer(r_inner)

        ee.invoke("matmul", r_outer, a_outer, b_outer)

        # Read output
        r_clone = get_ranked_memref_descriptor(init)
        ctypes.memmove(
            ctypes.addressof(r_clone),
            ctypes.cast(r_inner, ctypes.c_void_p).value,
            ctypes.sizeof(r_clone),
        )
        out = np.ctypeslib.as_array(r_clone.aligned, shape=(4, 4)).copy()

        expected = a @ b
        def cos_sim(x, y):
            return float(np.dot(x.ravel(), y.ravel()) / (np.linalg.norm(x.ravel()) * np.linalg.norm(y.ravel()) + 1e-12))

        cos = cos_sim(out, expected)
        max_diff = np.max(np.abs(out - expected))
        print(f"  cos(JIT, numpy): {cos:.10f}")
        print(f"  max diff:        {max_diff:.8f}")
        if cos > 0.9999:
            print("  ✅ Matmul JIT matches numpy")
        else:
            print(f"  ❌ Mismatch (cos={cos:.6f})")

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

# Test 4: model.lowered.mlir through lower_linalg_to_llvm_ir
print("\n" + "=" * 60)
print("TEST 4: model.lowered.mlir → ExecutionEngine JIT")
print("=" * 60)
lowered_path = os.path.join("compiled", "opt_125m_fresh", "model.lowered.mlir")
if os.path.exists(lowered_path):
    lowered_text = open(lowered_path).read()
    print(f"  Loaded model.lowered.mlir: {len(lowered_text)} chars")

    from mlir._mlir_libs import _mlirRegisterEverything
    ctx2 = Context()
    ctx2.allow_unregistered_dialects = True
    from mlir.ir import DialectRegistry
    reg = DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx2.append_dialect_registry(reg)

    with Location.unknown(ctx2):
        module = Module.parse(lowered_text, ctx2)
        print(f"  Parsed: {len(list(module.body))} blocks")

    # Use the same pipeline as compile_utils: lower_linalg_to_llvm_ir
    try:
        from compiler.mlir_dialect.llvm_backend import lower_linalg_to_llvm_ir
        t0 = time.time()
        llvm_text = lower_linalg_to_llvm_ir(module)
        dt = time.time() - t0
        print(f"  Pipeline: {dt:.1f}s, output: {len(llvm_text)} chars")
    except Exception as e:
        print(f"  ❌ Pipeline failed: {e}")
        traceback.print_exc()
        llvm_text = None

    if llvm_text:
        ctx3 = Context()
        ctx3.allow_unregistered_dialects = True
        with Location.unknown(ctx3):
            llvm_mod = Module.parse(llvm_text, ctx3)

            # Add llvm.emit_c_interface
            def add_ciface(op):
                if hasattr(op, "name") and op.name == "func.func":
                    with ctx3:
                        op.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get()
                return WalkResult.ADVANCE
            llvm_mod.operation.walk(add_ciface)

            # Run func-to-llvm + reconcile-casts
            try:
                pm = PassManager.parse(
                    "builtin.module(convert-func-to-llvm,reconcile-unrealized-casts)",
                    ctx3)
                pm.run(llvm_mod.operation)
                print("  ✅ func-to-llvm + casts resolved")
            except Exception as e:
                print(f"  ❌ Post-processing failed: {e}")

            # Count functions
            mod_str = str(llvm_mod)
            n_llvm_funcs = mod_str.count("llvm.func @")
            print(f"  LLVM functions: {n_llvm_funcs}")
            has_main0 = "main_0" in mod_str
            print(f"  Has main_0: {has_main0}")

            # Try ExecutionEngine
            print("  Creating ExecutionEngine...")
            try:
                t0 = time.time()
                ee = ExecutionEngine(llvm_mod, opt_level=0)
                dt = time.time() - t0
                print(f"  ✅ ExecutionEngine created in {dt:.1f}s")
                print("  JIT compilation WORKS for the full model!")
                print()
                print("  LIMITATION: main_0 has ~215 memref args,")
                print("  making invocation from Python extremely complex.")
                print("  Layer-by-layer JIT testing would be needed.")
            except Exception as e:
                print(f"  ❌ ExecutionEngine creation failed: {e}")
                traceback.print_exc()
else:
    print(f"  ❌ {lowered_path} not found")

print("\nDone.")
