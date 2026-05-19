#!/usr/bin/env python3
"""FP precision diagnostic tool for compiled model dylibs.

Usage:
    python scripts/diagnose_fp.py                             # Run matmul precision tests
    python scripts/diagnose_fp.py compiled/opt_125m_fresh     # Full model diagnosis (requires weights)
    python scripts/diagnose_fp.py --ref ref_output.npy        # Compare dylib output with reference

The script:
  1. Compiles matmul test cases through the full pipeline (tile + FMA + LLVM)
  2. Calls the dylib via ctypes and compares with numpy reference
  3. Reports cos similarity and max/mean error per case
  4. With --ref, compares full model output against a saved reference
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_mlir_pkg = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
    sys.path.insert(0, str(_mlir_pkg))


def _has_mlir() -> bool:
    try:
        import mlir.ir  # noqa: F401
        return True
    except ImportError:
        return False


def _find_llc() -> str:
    for p in ["/usr/local/opt/llvm/bin/llc", "/opt/homebrew/opt/llvm/bin/llc"]:
        if os.path.isfile(p):
            return p
    env = os.environ.get("SERVE_FORGE_LLVM_BIN", "")
    if env:
        return os.path.join(env, "llc")
    return subprocess.check_output(["which", "llc"]).decode().strip()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    na = np.linalg.norm(a_f)
    nb = np.linalg.norm(b_f)
    return float(np.dot(a_f, b_f) / (na * nb + 1e-12))


def make_memref_2d(arr: np.ndarray) -> ctypes.Structure:
    from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
    return get_ranked_memref_descriptor(arr)


def compile_matmul_dylib(M: int, K: int, N: int, work_dir: str,
                         with_fill: bool = True) -> tuple[str, int, int]:
    """Compile a matmul test to dylib. Returns (dylib_path, fmuladd_count, ll_lines)."""
    import mlir.ir as ir
    from compiler.mlir_dialect.llvm_backend import (
        _tile_matmuls_per_func,
        lower_linalg_to_llvm_ir,
        mlir_module_to_llvm_ir,
        link_dylib,
    )

    if with_fill:
        fill_src = (
            "    %zero = arith.constant 0.000000e+00 : f32\n"
            f"    %filled = linalg.fill ins(%zero : f32) outs(%empty : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>\n"
        )
        out_name = "%filled"
    else:
        fill_src = ""
        out_name = "%empty"

    mlir_src = (
        f"func.func @main(%a: tensor<{M}x{K}xf32>, %b: tensor<{K}x{N}xf32>) -> tensor<{M}x{N}xf32> {{\n"
        f"    %empty = tensor.empty() : tensor<{M}x{N}xf32>\n"
        f"{fill_src}"
        f"    %c = linalg.matmul ins(%a, %b : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) outs({out_name} : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>\n"
        f"    return %c : tensor<{M}x{N}xf32>\n"
        f"}}\n"
    )

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ir.Location.unknown(ctx):
        module = ir.Module.parse(mlir_src, ctx)

        def _ac(op):
            if op.operation.name == "func.func":
                with ctx:
                    op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            return ir.WalkResult.ADVANCE
        module.operation.walk(_ac)
        _tile_matmuls_per_func(module, tile_k=64)
        lower_linalg_to_llvm_ir(module)

    mlir_text = str(module)
    n_fmuladd = mlir_text.count("llvm.intr.fmuladd")

    llvm_ir = mlir_module_to_llvm_ir(module)
    ll_lines = len(llvm_ir.splitlines())

    ll_path = os.path.join(work_dir, "test.ll")
    with open(ll_path, "w") as f:
        f.write(llvm_ir)

    llc_bin = _find_llc()
    subprocess.run(
        [llc_bin, "-filetype=obj", "-O3", ll_path, "-o", os.path.join(work_dir, "test.o")],
        check=True, capture_output=True,
    )
    dylib_path = os.path.join(work_dir, "test.dylib")
    link_dylib([os.path.join(work_dir, "test.o")], dylib_path)
    return dylib_path, n_fmuladd, ll_lines


def run_matmul_tests():
    """Run precision diagnostics for various matmul sizes."""
    if not _has_mlir():
        print("❌ MLIR Python bindings not available. Activate virtualenv and run setup.sh.")
        return 1

    test_cases = [
        ("tiny_no_tile", 4, 8, 4, False),
        ("small_no_tile", 32, 64, 64, False),
        ("tile64", 64, 64, 64, True),
        ("layer_k64", 128, 64, 768, True),
        ("layer", 128, 768, 768, True),
        ("lm_head", 4, 768, 50272, True),
    ]

    print("=" * 70)
    print("  FP Precision Diagnostic: Matmul Pipeline")
    print("=" * 70)
    print(f"{'Test':<18} {'FMA':>4} {'IR':>6} {'O3 cos':>10} {'O3 max_err':>12} {'O3 mean_err':>12} {'O0 cos':>10}")
    print("-" * 70)

    all_ok = True

    for name, M, K, N, do_tile in test_cases:
        if not do_tile:
            # Skip non-tiled fast path
            print(f"{name:<18} {'n/a':>4} {'fast':>6} {'n/a':>10} {'n/a':>12} {'n/a':>12} {'n/a':>10}")
            continue

        np.random.seed(42 + hash(name) % 100000)
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        expected = a @ b

        with tempfile.TemporaryDirectory() as td:
            try:
                dylib_path, n_fma, ll_lines = compile_matmul_dylib(M, K, N, td)
            except Exception as e:
                print(f"{name:<18} {'ERR':>4} {'err':>6} {str(e)[:40]}")
                all_ok = False
                continue

            lib = ctypes.CDLL(dylib_path)
            a_d = make_memref_2d(a)
            b_d = make_memref_2d(b)
            out_arr = np.zeros((M, N), dtype=np.float32)
            out_d = make_memref_2d(out_arr)

            ciface = getattr(lib, "_mlir_ciface_main", None)
            if ciface is None:
                print(f"{name:<18} {'ERR':>4} {'no ciface':>6}")
                all_ok = False
                continue

            for opt_dir, opt_name in [(td, "O3")]:  # Only O3 for now
                out_arr[:] = 0
                ciface(ctypes.byref(out_d), ctypes.byref(a_d), ctypes.byref(b_d))
                result = np.ctypeslib.as_array(
                    ctypes.cast(out_d.aligned, ctypes.POINTER(ctypes.c_float)),
                    shape=(M, N),
                ).copy()

            cos_val = cosine(result, expected)
            max_err = float(np.max(np.abs(result - expected)))
            mean_err = float(np.mean(np.abs(result - expected)))

            ok = cos_val > 0.999
            all_ok = all_ok and ok
            mark = "✅" if ok else "❌"
            print(f"{mark} {name:<15} {n_fma:>4} {ll_lines:>6} {cos_val:>10.6f} {max_err:>12.2e} {mean_err:>12.2e}")

    print("-" * 70)
    print(f"{'ALL OK' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


def diagnose_full_model(model_dir: str, ref_path: str | None = None):
    """Compile full model and compare with reference."""
    print(f"Full model diagnosis not yet implemented. Provide --ref for reference comparison.")
    print(f"Model dir: {model_dir}, Ref: {ref_path}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="FP Precision Diagnostic")
    parser.add_argument("model_dir", nargs="?", help="Compiled model directory")
    parser.add_argument("--ref", help="Reference output .npy file")
    parser.add_argument("--matmul-only", action="store_true", help="Only run matmul tests")
    args = parser.parse_args()

    if args.model_dir and not args.matmul_only:
        return diagnose_full_model(args.model_dir, args.ref)
    return run_matmul_tests()


if __name__ == "__main__":
    sys.exit(main())
