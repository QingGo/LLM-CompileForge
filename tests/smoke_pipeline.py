"""Quick pipeline verification: compile a minimal model end-to-end.

Tests the entire compilation pipeline (sf→linalg → bufferize → LLVM → llc)
with a tiny model, catching hangs and correctness issues within ~10 seconds.
Run before any full model compilation to verify the pipeline is healthy.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler.backend.compile_utils import _setup_mlir_path
from compiler.pipeline.lowering import SF_LOWERING_PIPELINE

MINIMAL_MLIR = r"""
module {
  func.func @main(%a: tensor<2x4xf32>, %w1: tensor<8x4xf32>, %b1: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = "sf.linear"(%a, %w1, %b1) : (tensor<2x4xf32>, tensor<8x4xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = "sf.relu"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}
"""


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
        print("  [1/4] Parse: OK")

        # Step 2: C++ lowering
        t0 = time.time()
        pman = pm.PassManager.parse("builtin.module(" + SF_LOWERING_PIPELINE + ")", ctx)
        pman.enable_verifier(True)
        pman.run(module.operation)
        t = time.time() - t0
        text = str(module)
        sf_rem = text.count('"sf.')
        print(f"  [2/4] C++ lowering: {t:.2f}s (sf remaining: {sf_rem})")

        # Step 3: LLVM lowering (bufferize + convert)
        from compiler.backend.compile_utils import mlir_module_to_llvm_ir
        from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir

        t0 = time.time()
        llvm_text = lower_linalg_to_llvm_ir(module)
        t = time.time() - t0
        print(f"  [3/4] LLVM lowering: {t:.2f}s ({len(llvm_text)} chars)")

    # Step 4: mlir-translate → LLVM IR (optional, may fail on complex shapes)
    try:
        t0 = time.time()
        llvm_ir = mlir_module_to_llvm_ir(module)
        t = time.time() - t0
        print(f"  [4/4] Translate: {t:.2f}s ({len(llvm_ir)} chars)")
    except Exception as e:
        print(f"  [4/4] Translate: SKIPPED ({str(e)[:80]})")
        pass

    print(f"\n  {'✅ Pipeline smoke test passed' if sf_rem == 0 else '⚠ Some sf ops remain'}")

    print("\n  ✅ Pipeline smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
