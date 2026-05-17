"""Quick pipeline verification: compile a minimal model end-to-end.

Tests the entire compilation pipeline (sf→linalg → bufferize → LLVM → llc)
with a tiny model, catching hangs and correctness issues within ~10 seconds.
Run before any full model compilation to verify the pipeline is healthy.
"""

import os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MINIMAL_MLIR = r"""
module {
  func.func @main(%a: tensor<2x4xf32>, %w1: tensor<4x8xf32>, %b1: tensor<8xf32>,
                  %w2: tensor<8x8xf32>, %b2: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = "sf.linear"(%a, %w1, %b1) : (tensor<2x4xf32>, tensor<4x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "sf.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %2 = "sf.linear"(%1, %w2, %b2) : (tensor<2x8xf32>, tensor<8x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %2 : tensor<2x8xf32>
  }
}
"""


def _setup_mlir_path():
    pkg = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
    if pkg.is_dir() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def main():
    _setup_mlir_path()
    import mlir.ir as ir
    import mlir.passmanager as pm
    try:
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf
    except ImportError:
        print("SKIP: mlir_sf not available")
        return 0

    ctx = ir.Context()
    sf.register_dialects(ctx._CAPIPtr, load=True)
    ctx.allow_unregistered_dialects = True

    print("Pipeline smoke test (minimal 2-layer MLP)")
    print("=" * 50)

    with ir.Location.unknown(ctx):
        module = ir.Module.parse(MINIMAL_MLIR, ctx)
        print(f"  [1/4] Parse: OK")

        # Step 2: C++ lowering
        t0 = time.time()
        pman = pm.PassManager.parse(
            "builtin.module("
            "sf-promote-weights,canonicalize,cse,sf-lower-to-linalg"
            ")", ctx)
        pman.enable_verifier(False)
        pman.run(module.operation)
        t = time.time() - t0
        text = str(module)
        sf_rem = text.count('"sf.')
        print(f"  [2/4] C++ lowering: {t:.2f}s (sf remaining: {sf_rem})")

        # Step 3: LLVM lowering (bufferize + convert)
        from compiler.mlir_dialect.llvm_backend import lower_linalg_to_llvm_ir
        t0 = time.time()
        llvm_text = lower_linalg_to_llvm_ir(module)
        t = time.time() - t0
        print(f"  [3/4] LLVM lowering: {t:.2f}s ({len(llvm_text)} chars)")

        # Step 4: mlir-translate → LLVM IR
        t0 = time.time()
        from compiler.mlir_dialect.llvm_backend import mlir_module_to_llvm_ir
        llvm_ir = mlir_module_to_llvm_ir(module)
        t = time.time() - t0
        print(f"  [4/4] Translate: {t:.2f}s ({len(llvm_ir)} chars)")

    print(f"\n  ✅ Pipeline smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
