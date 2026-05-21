# ruff: noqa: E501
"""Lowering diagnostic: test each sf op type individually."""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Templates use ${var} syntax — replaced via .replace() not .format()
# to avoid conflicts with MLIR's generic op format syntax { ... }.
_TEMPLATES: dict[str, str] = {
    "sf.add": """
func.func @test(%a: tensor<$shape>, %b: tensor<$shape>) -> tensor<$shape> {
    %0 = "sf.add"(%a, %b) : (tensor<$shape>, tensor<$shape>) -> tensor<$shape>
    return %0 : tensor<$shape>
}
""",
    "sf.mul": """
func.func @test(%a: tensor<$shape>, %b: tensor<$shape>) -> tensor<$shape> {
    %0 = "sf.mul"(%a, %b) : (tensor<$shape>, tensor<$shape>) -> tensor<$shape>
    return %0 : tensor<$shape>
}
""",
    "sf.relu": """
func.func @test(%a: tensor<$shape>) -> tensor<$shape> {
    %0 = "sf.relu"(%a) : (tensor<$shape>) -> tensor<$shape>
    return %0 : tensor<$shape>
}
""",
    "sf.linear_2d": """
func.func @test(%a: tensor<$shape>, %w: tensor<$dimx_dim>, %b: tensor<$dim>) -> tensor<$shape> {
    %0 = "sf.linear"(%a, %w, %b) : (tensor<$shape>, tensor<$dimx_dim>, tensor<$dim>) -> tensor<$shape>
    return %0 : tensor<$shape>
}
""",
    "sf.linear_3d": """
func.func @test(%a: tensor<$b_seq_dim>, %w: tensor<$dimx_dim>, %b: tensor<$dim>) -> tensor<$b_seq_dim> {
    %0 = "sf.linear"(%a, %w, %b) : (tensor<$b_seq_dim>, tensor<$dimx_dim>, tensor<$dim>) -> tensor<$b_seq_dim>
    return %0 : tensor<$b_seq_dim>
}
""",
    "sf.linear_3d_dynamic": """
func.func @test(%a: tensor<?x?x$dim>, %w: tensor<$dimx_dim>, %b: tensor<$dim>) -> tensor<?x?x$dim> {
    %0 = "sf.linear"(%a, %w, %b) : (tensor<?x?x$dim>, tensor<$dimx_dim>, tensor<$dim>) -> tensor<?x?x$dim>
    return %0 : tensor<?x?x$dim>
}
""",
    "sf.layer_norm": r"""
func.func @test(%a: tensor<$shape>, %w: tensor<$dim>, %b: tensor<$dim>) -> tensor<$shape> {
    %0 = "sf.layer_norm"(%a, %w, %b) {normalized_shape = [$dim_int], eps = 1.000000e-05 : f32} : (tensor<$shape>, tensor<$dim>, tensor<$dim>) -> tensor<$shape>
    return %0 : tensor<$shape>
}
""",
    "sf.layer_norm_dynamic": r"""
func.func @test(%a: tensor<?x$dim>, %w: tensor<$dim>, %b: tensor<$dim>) -> tensor<?x$dim> {
    %0 = "sf.layer_norm"(%a, %w, %b) {normalized_shape = [$dim_int], eps = 1.000000e-05 : f32} : (tensor<?x$dim>, tensor<$dim>, tensor<$dim>) -> tensor<?x$dim>
    return %0 : tensor<?x$dim>
}
""",
    "sf.scaled_dot_product_attention": r"""
func.func @test(%q: tensor<1x12x4x64xf32>, %k: tensor<1x12x4x64xf32>, %v: tensor<1x12x4x64xf32>) -> tensor<1x12x4x64xf32> {
    %0 = "sf.scaled_dot_product_attention"(%q, %k, %v) {scale = 1.000000e+00 : f64} : (tensor<1x12x4x64xf32>, tensor<1x12x4x64xf32>, tensor<1x12x4x64xf32>) -> tensor<1x12x4x64xf32>
    return %0 : tensor<1x12x4x64xf32>
}
""",
    "sf.view": r"""
func.func @test(%a: tensor<$shape>) -> tensor<$view_shape> {
    %0 = "sf.view"(%a) {shape = [$view_dims]} : (tensor<$shape>) -> tensor<$view_shape>
    return %0 : tensor<$view_shape>
}
""",
    "sf.transpose": r"""
func.func @test(%a: tensor<$shape>) -> tensor<$tpose_shape> {
    %0 = "sf.transpose"(%a) <{dim0 = 1 : i64, dim1 = 2 : i64}> : (tensor<$shape>) -> tensor<$tpose_shape>
    return %0 : tensor<$tpose_shape>
}
""",
    "sf.unsqueeze": r"""
func.func @test(%a: tensor<$unsq_in>) -> tensor<$unsq_out> {
    %0 = "sf.unsqueeze"(%a) {dim = 0 : i64} : (tensor<$unsq_in>) -> tensor<$unsq_out>
    return %0 : tensor<$unsq_out>
}
""",
    "sf.ones_like": r"""
func.func @test(%a: tensor<$shape>) -> tensor<$shape> {
    %0 = "sf.ones_like"(%a) {shape = [$dims], device = "cpu", pin_memory = false} : (tensor<$shape>) -> tensor<$shape>
    return %0 : tensor<$shape>
}
""",
    "sf.cumsum": r"""
func.func @test(%a: tensor<$shape>) -> tensor<$shape> {
    %0 = "sf.cumsum"(%a) {dim = 1 : i64} : (tensor<$shape>) -> tensor<$shape>
    return %0 : tensor<$shape>
}
""",
    "sf.arange": r"""
func.func @test(%a: tensor<1xi64>) -> tensor<i64> {
    %0 = "sf.arange"(%a) {device = "cpu", pin_memory = false} : (tensor<1xi64>) -> tensor<i64>
    return %0 : tensor<i64>
}
""",
    "sf.index": r"""
func.func @test(%data: tensor<1xf32>, %idx1: tensor<1x1x1xf32>) -> tensor<1x1x1xf32> {
    %0 = "sf.index"(%data, %idx1) {shape = ["%idx1"]} : (tensor<1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
}
""",
}


def _apply(template: str, vals: dict[str, str]) -> str:
    # Replace longer variable names first to avoid partial replacement
    for k, v in sorted(vals.items(), key=lambda x: -len(x[0])):
        template = template.replace(f"${k}", v)
    return template


def _setup_mlir_path():
    pkg = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
    if pkg.is_dir() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def run_test(name: str, mlir_text: str, timeout_s: int = 5) -> tuple[bool, str]:
    import mlir.ir as ir
    import mlir.passmanager as pm
    try:
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf
    except ImportError:
        return False, "mlir_sf not available"

    ctx = ir.Context()
    try:
        sf.register_dialects(ctx._CAPIPtr, load=True)
    except Exception as e:
        return False, f"sf registration failed: {e}"
    ctx.allow_unregistered_dialects = True

    with ir.Location.unknown(ctx):
        try:
            module = ir.Module.parse("module {\n" + mlir_text + "\n}", ctx)
        except Exception as e:
            return False, f"parse failed: {e}"

        try:
            pman = pm.PassManager.parse(
                "builtin.module("
                "sf-promote-weights,canonicalize,cse,sf-lower-to-linalg"
                ")", ctx)
            pman.enable_verifier(False)
            t0 = time.time()
            pman.run(module.operation)
            elapsed = time.time() - t0
        except Exception as e:
            elapsed = 0
            return False, f"pipeline failed: {str(e)[:200]}"

        text = str(module)
        sf_remaining = text.count('"sf.')
        if sf_remaining > 0:
            return False, f"incomplete ({sf_remaining} sf ops remain) {elapsed:.2f}s"

        linalg_count = text.count("linalg.")
        return True, f"OK {linalg_count} linalg ops in {elapsed:.2f}s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()

    _setup_mlir_path()

    defaults = {
        "b": "1", "seq": "4", "nh": "12", "dk": "64",
        "dim_int": "64",
        "dim": "64xf32",
        "shape": "2x64xf32",
        "ndim": "768",
        "view_shape": "1x?x12x64xf32",
        "view_dims": "-1, 12, 64",
        "tpose_shape": "1x12x?x64xf32",
        "unsq_in": "1xf32",
        "unsq_out": "1x1xf32",
        "dims": "1, 4",
        "dimx": "64x",
        "b_seq_dim": "1x4x64xf32",
        "dimx_dim": "64x64xf32",
        "shape32": "4x64xf32",
    }

    passed = 0
    failed = 0

    print(f"{'Op':<35} {'Status':<10} {'Detail':<50}")
    print("-" * 95)

    for op_name, template in sorted(_TEMPLATES.items()):
        if args.op and args.op not in op_name:
            continue

        mlir = _apply(template, defaults)
        ok, msg = run_test(op_name, mlir, args.timeout)

        if len(msg) > 47:
            msg = msg[:44] + "..."

        status = "PASS" if ok else "FAIL"
        print(f"{op_name:<35} {status:<10} {msg:<50}")

        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nPassed: {passed}, Failed: {failed}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
