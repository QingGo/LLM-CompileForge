"""Precision bisect: test individual ops through the lowered+JIT path.

Goal: isolate whether accuracy gap is in C++ lowering or LLVM codegen.
Strategy: test one op at a time, comparing JIT output with numpy/torch.
"""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_mlir_path():
    _pkg = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
    if _pkg.is_dir() and str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))


def cosine_similarity(a, b):
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))


def make_memref_f32(arr):
    from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
    desc = get_ranked_memref_descriptor(arr)
    inner = ctypes.pointer(desc)
    outer = ctypes.pointer(inner)
    return inner, outer


def read_memref_output(inner_ptr, arr_dummy):
    from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
    clone = get_ranked_memref_descriptor(arr_dummy)
    ctypes.memmove(
        ctypes.addressof(clone),
        ctypes.cast(inner_ptr, ctypes.c_void_p).value,
        ctypes.sizeof(clone),
    )
    return np.ctypeslib.as_array(clone.aligned, shape=tuple(clone.shape)).copy()


def add_emit_c_interface(module, ctx):
    import mlir.ir as ir
    def _cb(op):
        if hasattr(op, "name") and op.name == "func.func":
            with ctx:
                op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        return ir.WalkResult.ADVANCE
    module.operation.walk(_cb)


def test_op(sf_mlir: str, numpy_fn, shapes, desc: str) -> float:
    """Test a single op: compile sf MLIR → JIT run → compare with numpy."""
    import mlir.ir as ir
    import mlir.passmanager as pm
    from mlir.execution_engine import ExecutionEngine
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)

    rng = np.random.RandomState(42)
    inputs = [rng.randn(*s).astype(np.float32) for s in shapes]
    expected = numpy_fn(*inputs)

    with ir.Location.unknown(ctx):
        # Parse + lower
        module = ir.Module.parse(sf_mlir, ctx)
        try:
            pm.PassManager.parse(
                "builtin.module("
                "sf-promote-weights,canonicalize,cse,"
                "sf-lower-to-linalg"
                ")",
                ctx
            ).run(module.operation)
        except Exception as e:
            print(f"  Lowering FAILED: {e}")
            return -1.0

        # LLVM pipeline
        try:
            pipeline = (
                "builtin.module("
                "canonicalize,cse,"
                "one-shot-bufferize{bufferize-function-boundaries},"
                "convert-linalg-to-loops,"
                "lower-affine,"
                "convert-scf-to-cf,"
                "expand-strided-metadata,"
                "lower-affine,"
                "finalize-memref-to-llvm,"
                "convert-cf-to-llvm,"
                "convert-math-to-llvm,"
                "convert-arith-to-llvm,"
                ")"
            )
            pm.PassManager.parse(pipeline, ctx).run(module.operation)
        except Exception as e:
            print(f"  LLVM pipeline FAILED: {e}")
            return -1.0

        add_emit_c_interface(module, ctx)
        try:
            pm.PassManager.parse(
                "builtin.module(convert-func-to-llvm,reconcile-unrealized-casts)",
                ctx
            ).run(module.operation)
        except Exception as e:
            print(f"  func-to-llvm FAILED: {e}")
            return -1.0

        # JIT
        try:
            engine = ExecutionEngine(module, opt_level=0)
        except Exception as e:
            print(f"  ExecutionEngine FAILED: {e}")
            return -1.0

        # Build args
        args = []
        inner_list = []
        for inp in inputs:
            inner, outer = make_memref_f32(inp)
            inner_list.append(inner)
            args.append(outer)

        out_shape = expected.shape
        result = np.zeros(out_shape, dtype=np.float32)
        r_inner, r_outer = make_memref_f32(result)
        inner_list.append(r_inner)
        args.append(r_outer)

        engine.invoke("f", *args)
        out = read_memref_output(r_inner, np.zeros(out_shape, dtype=np.float32))

        sim = cosine_similarity(out, expected)
        max_diff = np.max(np.abs(out - expected))
        print(f"  {desc}: cos={sim:.10f}, max_diff={max_diff:.8e}")
        return sim


def main():
    _setup_mlir_path()

    tests = []

    # 1. Matmul 4x8 @ 8x4
    tests.append((
        """module {
  func.func @f(%a: tensor<4x8xf32>, %b: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %0 = "sf.matmul"(%a, %b) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}""",
        lambda a, b: a @ b,
        [(4, 8), (8, 4)],
        "matmul 4x4"
    ))

    # 2. Matmul 64x768 @ 768x64 (typical sizes)
    tests.append((
        """module {
  func.func @f(%a: tensor<64x768xf32>, %b: tensor<768x64xf32>) -> tensor<64x64xf32> {
    %0 = "sf.matmul"(%a, %b) : (tensor<64x768xf32>, tensor<768x64xf32>) -> tensor<64x64xf32>
    return %0 : tensor<64x64xf32>
  }
}""",
        lambda a, b: a @ b,
        [(64, 768), (768, 64)],
        "matmul 64x64"
    ))

    # 3. BatchMatmul 2x4x768 @ 2x768x4 (small)
    tests.append((
        """module {
  func.func @f(%a: tensor<2x4x768xf32>, %b: tensor<2x768x4xf32>) -> tensor<2x4x4xf32> {
    %0 = "sf.matmul"(%a, %b) : (tensor<2x4x768xf32>, tensor<2x768x4xf32>) -> tensor<2x4x4xf32>
    return %0 : tensor<2x4x4xf32>
  }
}""",
        lambda a, b: a @ b,
        [(2, 4, 768), (2, 768, 4)],
        "batch_matmul 2x4x4"
    ))

    # 4. Relu
    tests.append((
        """module {
  func.func @f(%a: tensor<64xf32>) -> tensor<64xf32> {
    %0 = "sf.relu"(%a) : (tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}""",
        lambda a: np.maximum(a, 0.0),
        [(64,)],
        "relu 64"
    ))

    # 5. Add
    tests.append((
        """module {
  func.func @f(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
    %0 = "sf.add"(%a, %b) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
    return %0 : tensor<64xf32>
  }
}""",
        lambda a, b: a + b,
        [(64,), (64,)],
        "add 64"
    ))

    # 6. Embedding (reduce over vocab = no reduction, just lookup)
    tests.append((
        """module {
  func.func @f(%a: tensor<1x4xi64>, %b: tensor<10x64xf32>) -> tensor<1x4x64xf32> {
    %0 = "sf.embedding"(%a, %b) : (tensor<1x4xi64>, tensor<10x64xf32>) -> tensor<1x4x64xf32>
    return %0 : tensor<1x4x64xf32>
  }
}""",
        lambda a, b: b[a[0].astype(int)],
        [(1, 4), (10, 64)],
        "embedding 1x4"
    ))

    # 7. LayerNorm (mean + var)
    tests.append((
        """module {
  func.func @f(%a: tensor<2x64xf32>, %w: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<2x64xf32> {
    %0 = "sf.layer_norm"(%a, %w, %b) {eps = 1.000000e-05 : f32}"""
        """ : (tensor<2x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""",
        lambda a, w, b: (a - a.mean(axis=1, keepdims=True)) / np.sqrt(a.var(axis=1, keepdims=True) + 1e-5) * w + b,
        [(2, 64), (64,), (64,)],
        "layer_norm 2x64"
    ))

    print("=== Testing individual ops through lowered+JIT path ===\n")
    all_pass = True
    for mlir, fn, shapes, desc in tests:
        try:
            sim = test_op(mlir, fn, shapes, desc)
            if sim < 0.999:
                print(f"  ⚠  cos={sim:.6f} < 0.999 for {desc}")
                all_pass = False
        except Exception as e:
            print(f"  FAILED: {desc}: {e}")
            all_pass = False

    print(f"\n{'✅ All ops pass' if all_pass else '❌ Some ops failed'}")


if __name__ == "__main__":
    main()
