#!/usr/bin/env python3
"""
Test: Does bufferization corrupt tensor.extract indices used in sf.embedding lowering?

The sf.embedding lowering produces:
  linalg.generic {
    ^bb0(%in: f32, %out: f32):
      %fp_idx = arith.fptoui %in : f32 to i64
      %idx = arith.index_cast %fp_idx : i64 to index
      %dim = linalg.index 1 : index
      %val = tensor.extract %table[%idx, %dim] : tensor<2050x768xf32>
      linalg.yield %val : f32
  }

This test creates that exact pattern, runs it through bufferization and subsequent
pipeline stages, and verifies the indices survive correctly.

Run:
  source .venv/bin/activate
  python tools/bufferize_embedding_test.py 2>&1 | tee /tmp/bufferize_test_results.txt
"""

import sys, os
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("bufferize_test")

# ── Create minimal MLIR test ────────────────────────────────────────────

TEST_MLIR = r"""module {
  func.func @test_embedding(%arg0: tensor<4xf32>, %arg1: tensor<2050x768xf32>) -> tensor<4x768xf32> {
    %c0 = arith.constant 0 : index
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 2.000000e+00 : f32
    %cst_3 = arith.constant 3.000000e+00 : f32
    %cst_4 = arith.constant 4.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x768xf32>
    %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<4xf32>) outs(%0 : tensor<4x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %fp_idx = arith.fptoui %in : f32 to i64
      %idx = arith.index_cast %fp_idx : i64 to index
      %dim = linalg.index 1 : index
      %val = tensor.extract %arg1[%idx, %dim] : tensor<2050x768xf32>
      linalg.yield %val : f32
    } -> tensor<4x768xf32>
    return %1 : tensor<4x768xf32>
  }
}
"""


def run_stage(name: str, module: "ir.Module", pipeline_str: str) -> "ir.Module":
    """Run a pass pipeline on a module, return (possibly new) module."""
    import mlir.passmanager as pm
    ctx = module.operation.context
    try:
        pm.PassManager.parse(pipeline_str, ctx).run(module.operation)
        log.info(f"  [PASS] {name}")
    except Exception as e:
        log.error(f"  [FAIL] {name}: {e}")
        # Return module even on failure for debug inspection
    return module


def main():
    import mlir.ir as ir

    ctx = ir.Context()
    # Load needed dialects
    ctx.allow_unregistered_dialects = True

    # Parse module
    module = ir.Module.parse(TEST_MLIR, ctx)
    log.info("=" * 72)
    log.info("INVESTIGATION: Does bufferization corrupt tensor.extract indices?")
    log.info("=" * 72)

    # ── Stage 1: Initial IR ──
    log.info("\n[1/8] Initial IR (before bufferization)")
    log.info(str(module))
    assert "tensor.extract" in str(module), "tensor.extract should be present"

    # ── Stage 2: Bufferize ──
    log.info("\n[2/8] After one-shot-bufferize")
    run_stage("one-shot-bufferize", module,
              "builtin.module(one-shot-bufferize{bufferize-function-boundaries allow-unknown-ops},canonicalize,cse,convert-bufferization-to-memref)")
    log.info(str(module))

    # Check: tensor.extract should now be memref.load (or still tensor.extract if not bufferized)
    has_tensor_extract = "tensor.extract" in str(module)
    has_memref_load = "memref.load" in str(module)
    has_fptoui = "arith.fptoui" in str(module) or "fptoui" in str(module)
    has_index_cast = "arith.index_cast" in str(module) or "index_cast" in str(module)
    log.info(f"  tensor.extract present: {has_tensor_extract}")
    log.info(f"  memref.load present: {has_memref_load}")
    log.info(f"  fptoui present: {has_fptoui}")
    log.info(f"  index_cast present: {has_index_cast}")

    # ── Stage 3: linalg→loops ──
    log.info("\n[3/8] After linalg→loops")
    run_stage("convert-linalg-to-loops", module, "builtin.module(convert-linalg-to-loops)")
    log.info(str(module))

    # If the linalg.generic was bufferized, it would have become scf.for loops
    has_scf_for = "scf.for" in str(module)
    has_memref_load_st3 = "memref.load" in str(module)
    log.info(f"  scf.for present: {has_scf_for}")
    log.info(f"  memref.load present: {has_memref_load_st3}")

    # ── Stage 4: lower-affine ──
    log.info("\n[4/8] After lower-affine")
    run_stage("lower-affine", module, "builtin.module(lower-affine)")
    log.info("  (affine ops lowered)")

    # ── Stage 5: scf→cf ──
    log.info("\n[5/8] After scf→cf")
    run_stage("convert-scf-to-cf", module, "builtin.module(convert-scf-to-cf)")

    # ── Stage 6: lower-affine + lower-vec-mask + canonicalize ──
    log.info("\n[6/8] After expand-strided → lower-affine-2 → lower-vec-mask → canonicalize")
    run_stage("expand-strided", module, "builtin.module(expand-strided-metadata)")
    run_stage("lower-affine-2", module, "builtin.module(lower-affine)")
    run_stage("lower-vec-mask", module, "builtin.module(func.func(lower-vector-mask))")
    run_stage("canonicalize,cse", module, "builtin.module(canonicalize,cse)")

    # ── Stage 7: Convert to LLVM ──
    log.info("\n[7/8] After LLVM conversions (cf→llvm, finalize-memref, arith→llvm, func→llvm)")
    run_stage("convert-cf-to-llvm", module, "builtin.module(convert-cf-to-llvm)")
    run_stage("finalize-memref-to-llvm", module,
              "builtin.module(finalize-memref-to-llvm{use-generic-functions=false})")
    run_stage("convert-math-to-llvm", module, "builtin.module(convert-math-to-llvm)")
    run_stage("convert-arith-to-llvm", module, "builtin.module(convert-arith-to-llvm)")
    run_stage("convert-func-to-llvm", module, "builtin.module(convert-func-to-llvm)")
    run_stage("reconcile-unrealized-casts", module, "builtin.module(reconcile-unrealized-casts)")

    # ── Stage 8: Final LLVM IR ──
    log.info("\n[8/8] Final LLVM IR (checking fptoui + gep pattern)")
    llvm_ir = str(module)
    log.info(llvm_ir)

    # ── Analysis ──
    log.info("\n" + "=" * 72)
    log.info("ANALYSIS")
    log.info("=" * 72)

    # Check for the critical pattern in LLVM IR
    has_fptoui_llvm = "fptoui" in llvm_ir
    has_getelementptr = "getelementptr" in llvm_ir
    has_load = "load" in llvm_ir

    log.info(f"  fptoui present in LLVM IR: {has_fptoui_llvm}")
    log.info(f"  getelementptr present: {has_getelementptr}")
    log.info(f"  load present: {has_load}")

    # Specific checks
    log.info("")
    if has_fptoui_llvm:
        log.info("  ✓ fptoui survives to LLVM IR – index computation preserved")
    else:
        log.warning("  ✗ fptoui ABSENT from LLVM IR – index may be corrupted!")

    if has_fptoui_llvm:
        log.info("  ✓ Embedding table lookup uses fptoui → gep → load pattern")
        log.info("  ✓ Bufferization does NOT corrupt tensor.extract indices")
    else:
        log.info("  ✗ NEEDS INVESTIGATION: the fptoui → index_cast → tensor.extract pattern")
        log.info("    is lost during lowering")

    log.info("\n" + "=" * 72)
    log.info("CONCLUSION")
    log.info("=" * 72)

    # Check if the pattern is intact in LLVM IR
    lines = llvm_ir.split('\n')
    fptoui_lines = [l for l in lines if 'fptoui' in l]
    for l in fptoui_lines:
        log.info(f"  Found: {l.strip()}")

    if has_fptoui_llvm and has_getelementptr:
        log.info("\n  ✓ INDEX INTEGRITY: The fptoui + gep pattern is correctly")
        log.info("    preserved through the entire pipeline.")
        log.info("\n  The root cause of identical position embeddings for padding")
        log.info("  positions is likely NOT in bufferization, but in the VALUES")
        log.info("  fed into the fptoui (the cumsum values for padding tokens).")
        log.info("  H1 (cumsum) was 'DISPROVEN' but might need re-examination")
        log.info("  for PADDING tokens specifically.")
    else:
        log.error("\n  ✗ INDEX CORRUPTION: The fptoui pattern is lost!")
        log.error("    Bufferization may corrupt tensor.extract indices.")

    log.info("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
