#!/usr/bin/env python3
"""RED phase: test sf.embedding through full production pipeline."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _find_tool(name: str) -> str:
    candidates = [
        name,
        str(ROOT / "llvm-project" / "build" / "bin" / name),
    ]
    for c in candidates:
        p = Path(c)
        if p.is_file():
            return str(p)
        try:
            r = subprocess.run([c, "--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return c
        except FileNotFoundError:
            continue
    raise RuntimeError(f"{name} not found")


def test_embedding_dylib() -> float:
    import mlir.ir as ir
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf

    from compiler.mlir_dialect.fixups import _fixup_unrealized_casts_pass
    from compiler.mlir_dialect.llvm_backend import lower_linalg_to_llvm_ir
    from compiler.pipeline import _apply_sf_to_linalg

    sf_mlir = """module {
  func.func @f(%w: tensor<8x4xf32>, %idx: tensor<2x3xi64>) -> tensor<2x3x4xf32> {
    %0 = "sf.embedding"(%w, %idx) : (tensor<8x4xf32>, tensor<2x3xi64>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}"""

    weight = np.array([[i*4+1, i*4+2, i*4+3, i*4+4] for i in range(8)], dtype=np.float32)
    indices = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64)
    expected = weight[indices]

    print("  [1/7] sf -> linalg ...")
    lowered = _apply_sf_to_linalg(sf_mlir)
    assert "linalg.generic" in lowered
    print("  [2/7] linalg -> LLVM ...")
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)
    from mlir._mlir_libs import _mlirRegisterEverything
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx.append_dialect_registry(reg)
    with ir.Location.unknown(ctx):
        module = ir.Module.parse(lowered, ctx)
        lower_linalg_to_llvm_ir(module)
    llvm_mlir = str(module)
    print("  [3/7] Fix residual unrealized_conversion_cast ...")
    _fixup_unrealized_casts_pass(module)
    fixed = str(module)

    with tempfile.TemporaryDirectory() as td:
        mlir_path = os.path.join(td, "fixed.mlir")
        with open(mlir_path, "w") as f:
            f.write(fixed)
        print("  [4/7] mlir-translate -> .ll ...")
        subprocess.run(
            [_find_tool("mlir-translate"), "--mlir-to-llvmir", mlir_path, "-o",
             os.path.join(td, "module.ll")],
            capture_output=True, text=True, check=True, timeout=60,
        )
        print("  [5/7] clang -c -> .o (via LLVM IR) ...")
        subprocess.run(
            [_find_tool("cc"), "-c", os.path.join(td, "module.ll"), "-o",
             os.path.join(td, "module.o"), "-O0"],
            capture_output=True, text=True, check=True, timeout=60,
        )
        print("  [6/7] clang -shared -> .dylib ...")
        subprocess.run(
            [_find_tool("cc"), "-shared", "-o", os.path.join(td, "libembed.dylib"),
             os.path.join(td, "module.o")],
            capture_output=True, text=True, check=True, timeout=60,
        )
        print("  [7/7] Load and run ...")
        lib = ctypes.CDLL(os.path.join(td, "libembed.dylib"))

        def make_memref_struct(ptr, ndim, shape):
            strides = tuple(int(np.prod(shape[i+1:])) for i in range(ndim))
            class MemRef(ctypes.Structure):
                _fields_ = [
                    ("allocated", ctypes.c_void_p),
                    ("aligned", ctypes.c_void_p),
                    ("offset", ctypes.c_int64),
                    ("sizes", ctypes.c_int64 * ndim),
                    ("strides", ctypes.c_int64 * ndim),
                ]
            return MemRef(
                ctypes.c_void_p(ptr),
                ctypes.c_void_p(ptr),
                0,
                (ctypes.c_int64 * ndim)(*shape),
                (ctypes.c_int64 * ndim)(*strides),
            )

        wd = make_memref_struct(weight.ctypes.data, 2, weight.shape)
        kid = make_memref_struct(indices.ctypes.data, 2, indices.shape)

        sret = (ctypes.c_uint8 * 1024)()
        lib._mlir_ciface_f(
            ctypes.byref(sret),
            ctypes.byref(wd),
            ctypes.byref(kid),
        )

        # Parse sret struct: {ptr allocated, ptr aligned, i64 offset, [3xi64] sizes, [3xi64] strides}
        aligned_ptr = ctypes.c_void_p.from_buffer(sret, 8)  # offset 8 = aligned ptr
        out_data_ptr = ctypes.cast(aligned_ptr, ctypes.POINTER(ctypes.c_float))
        num_elts = int(np.prod(expected.shape))
        out_vals = [out_data_ptr[i] for i in range(num_elts)]
        out = np.array(out_vals, dtype=np.float32).reshape(expected.shape)

        cos = float(np.dot(out.ravel(), expected.ravel()) /
                    (np.linalg.norm(out.ravel()) * np.linalg.norm(expected.ravel()) + 1e-12))
        max_diff = float(np.max(np.abs(out - expected)))
        print(f"  cos={cos:.10f} max_diff={max_diff:.8e}")

        if max_diff < 1e-4:
            print(f"  Expected: {expected.ravel()[:12].tolist()}...")
            print(f"  Got:      {out.ravel()[:12].tolist()}...")

        return cos


def main():
    print("RED: sf.embedding via .dylib" + "=" * 40)
    try:
        cos = test_embedding_dylib()
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    if cos >= 0.999:
        print(f"\nGREEN: cos={cos:.10f} >= 0.999")
    else:
        print(f"\nRED: cos={cos:.10f} < 0.999 - lowering bug CONFIRMED")


if __name__ == "__main__":
    main()
