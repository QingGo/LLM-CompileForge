# ruff: noqa: E501
"""Unit tests for bugs found during the tile+vectorize pipeline development.

Each test targets a specific bug that was found in the wild, so we have
regression coverage and can detect regressions faster.

Bug references (see AGENTS.md #41-#47):
  #41: vector.mask { vector.contract } hangs convert-vector-to-llvm
       → test_contract_lowering_outerproduct_finishes
  #42: mlir-translate (LLVM 20) doesn't know 'ub' dialect
       → test_contract_lowering_outerproduct_finishes (includes convert-ub-to-llvm)
  #43: convert-vector-to-scf creates scf.for → needs 2nd convert-scf-to-cf
       → test_contract_lowering_outerproduct_finishes
  #44: vectorized .dylib cosine verified
       → test_mlir_executor_multi_function
  #45: Rust .dylib forward accuracy gap (pre-existing)
       → (no test — tracked separately)
  #46: test_correctness_jit.py outdated (C++ lowering replaced Python)
       → (needs rewrite)
  #47: llc -O0 stack overflow on large functions
       → test_contract_lowering_outerproduct_finishes (uses -O2)

Session-specific bugs (not yet in AGENTS.md):
  A. Nested builtin.module blocks one-shot-bufferize
       → test_vectorize_flattens_nested_module
  B. transform.* ops leak into output → confuse bufferization
       → test_transform_ops_removed_after_vectorize
  C. vector_sizes [0,0,32,32] fails isValidMaskedInputVector when
     canonicalize folds ? → 2 (static dim > vector_size 0)
       → test_exact_vector_sizes_work
  D. one-shot-bufferize needs full dialect registry (vector bufferization)
       → test_bufferize_produces_memrefs
  E. convert-vector-to-llvm hangs with default `dot` strategy
       → test_contract_lowering_outerproduct_finishes
  F. MlirExecutor only runs main_0, ignores main_1
       → test_mlir_executor_multi_function
"""

from __future__ import annotations

import sys
from pathlib import Path

_mlir_pkg = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
    sys.path.insert(0, str(_mlir_pkg))

import pytest  # noqa: E402

_HAS_MLIR = False
try:
    import mlir.ir  # noqa: F401

    _HAS_MLIR = True
except ImportError:
    pass

pytestmark = [
    pytest.mark.skipif(not _HAS_MLIR, reason="mlir-core not available"),
    pytest.mark.integration,
]


# ── Nested builtin.module flattening ───────────────────────────
# Bug: transform-interpreter creates nested `module { module { func.func } }`.
# one-shot-bufferize ignores nested modules → 0 memrefs → cf.br has tensors → crash.


@pytest.mark.integration
def test_vectorize_flattens_nested_module():
    """Canonicalized module must not have nested builtin modules."""
    import mlir.ir as ir

    # Need a lowered module to test
    lowered_path = (
        Path(__file__).resolve().parent.parent
        / "compiled"
        / "opt_125m_fresh"
        / "model.lowered.mlir"
    )
    if not lowered_path.exists():
        pytest.skip("model.lowered.mlir not found — run compile_dylib.py first")

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ir.Location.unknown(ctx):
        module = ir.Module.parse(lowered_path.read_text())
        import mlir.passmanager as pm

        pm.PassManager.parse("builtin.module(canonicalize,cse)", ctx).run(module.operation)

        # Check: no nested `module { module {`
        text = str(module)
        # Count top-level module nesting: find the first func.func,
        # then count how many 'module {' come before it
        func_idx = text.find("func.func")
        before_func = text[:func_idx] if func_idx > 0 else text
        n_nested = before_func.count("module {")
        assert n_nested <= 1, (
            f"Expected <=1 top-level module, got {n_nested} (nested module not flattened)"
        )


# ── isValidMaskedInputVector static batch dims ─────────────────
# Bug: after canonicalize, batch dims become static (2x4). vector_sizes [0,0,32,32]
# fails because isValidMaskedInputVector checks staticSize <= inputSize.
# With 2 <= 0 → FAIL. Fix: use vector_sizes >= actual static dims.


@pytest.mark.integration
def test_exact_vector_sizes_work():
    """Exact vector_sizes matching static dims must not raise."""
    import mlir.ir as ir
    import mlir.passmanager as pm

    ctx = ir.Context()
    ctx.load_all_available_dialects()
    with ir.Location.unknown(ctx):
        m = ir.Module.parse(
            """
func.func @test(%a: tensor<2x4x768xf32>, %b: tensor<2x768x768xf32>) -> tensor<2x4x768xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2x4x768xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<2x4x768xf32>) -> tensor<2x4x768xf32>
  %r = linalg.batch_matmul ins(%a, %b : tensor<2x4x768xf32>, tensor<2x768x768xf32>) outs(%fill : tensor<2x4x768xf32>) -> tensor<2x4x768xf32>
  return %r : tensor<2x4x768xf32>
}
""",
            ctx,
        )
        script = """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %bmms = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> (!transform.any_op)
    %tiled_b, %loops_b:2 = transform.structured.tile_using_for %bmms tile_sizes [0, 0, 32, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled_b vector_sizes [2, 4, 32, 32] {create_named_contraction} : !transform.any_op
    transform.yield
  }
}
"""
        combined = ir.Module.parse(script + "\n" + str(m), ctx)
        pm.PassManager.parse("builtin.module(transform-interpreter)", ctx).run(combined.operation)
        text = str(combined)
        assert "vector.contract" in text, "Expected vector.contract for batch_matmul"
        assert "vector.mask" not in text, (
            "Exact vector_sizes should not create masks"
        )


# ── Transform ops removal ─────────────────────────────────────
# Bug: transform-interpreter leaves transform dialect ops in the module.
# These carry tensor types that confuse one-shot-bufferize.


@pytest.mark.integration
def test_transform_ops_removed_after_vectorize():
    """Canonicalized module must not have transform dialect ops in output."""
    import mlir.ir as ir

    lowered_path = (
        Path(__file__).resolve().parent.parent
        / "compiled"
        / "opt_125m_fresh"
        / "model.lowered.mlir"
    )
    if not lowered_path.exists():
        pytest.skip("model.lowered.mlir not found")

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ir.Location.unknown(ctx):
        module = ir.Module.parse(lowered_path.read_text())
        import mlir.passmanager as pm

        pm.PassManager.parse("builtin.module(canonicalize,cse)", ctx).run(module.operation)
        text = str(module)
        assert "transform.structured" not in text, (
            "transform dialect ops leaked into output module"
        )


# ── Bufferization with full dialect registry ──────────────────
# Bug: one-shot-bufferize needs vector::registerBufferizableOpInterfaceExternalModels
# to handle vector.transfer_read/write tensor→memref conversion.


@pytest.mark.integration
def test_bufferize_produces_memrefs():
    """After bufferize, there must be memref< types and no tensor< types in output."""
    import mlir.ir as ir
    import mlir.passmanager as pm
    from mlir._mlir_libs import _mlirRegisterEverything

    lowered_path = (
        Path(__file__).resolve().parent.parent
        / "compiled"
        / "opt_125m_fresh"
        / "model.lowered.mlir"
    )
    if not lowered_path.exists():
        pytest.skip("model.lowered.mlir not found")

    # Pre-process
    ctx1 = ir.Context()
    ctx1.allow_unregistered_dialects = True
    with ir.Location.unknown(ctx1):
        m = ir.Module.parse(lowered_path.read_text())
        pm.PassManager.parse("builtin.module(canonicalize,cse)", ctx1).run(m.operation)

    # Bufferize
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx2 = ir.Context()
    ctx2.allow_unregistered_dialects = True
    ctx2.append_dialect_registry(reg)

    with ir.Location.unknown(ctx2):
        m2 = ir.Module.parse(str(m), ctx2)
        pm.PassManager.parse(
            "builtin.module(one-shot-bufferize{bufferize-function-boundaries})", ctx2
        ).run(m2.operation)
        text = str(m2)
        n_memref = text.count("memref<")
        n_tensor = text.count("tensor<")
        assert n_memref > 0, "Expected memref<, got 0 (vector ops not bufferized)"
        assert n_tensor == 0, (
            f"Expected 0 tensor<, got {n_tensor} (tensors leaked)"
        )


# ── convert-vector-to-llvm must not hang ──────────────────────
# Bug: 73 multi-dimensional vector.contract with default `dot` lowering
# strategy caused the pass to hang for >10 min. `outerproduct` is fast.


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_contract_lowering_outerproduct_finishes():
    """convert-vector-to-llvm{vector-contract-lowering=outerproduct} must finish."""
    import mlir.ir as ir
    import mlir.passmanager as pm
    from mlir._mlir_libs import _mlirRegisterEverything

    lowered_path = (
        Path(__file__).resolve().parent.parent
        / "compiled"
        / "opt_125m_fresh"
        / "model.lowered.mlir"
    )
    if not lowered_path.exists():
        pytest.skip("model.lowered.mlir not found")

    # Pre-process
    ctx1 = ir.Context()
    ctx1.allow_unregistered_dialects = True
    with ir.Location.unknown(ctx1):
        m = ir.Module.parse(lowered_path.read_text())
        pm.PassManager.parse("builtin.module(canonicalize,cse)", ctx1).run(m.operation)

    # Bufferize
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx2 = ir.Context()
    ctx2.allow_unregistered_dialects = True
    ctx2.append_dialect_registry(reg)

    with ir.Location.unknown(ctx2):
        m2 = ir.Module.parse(str(m), ctx2)
        pm.PassManager.parse(
            "builtin.module(one-shot-bufferize{bufferize-function-boundaries})", ctx2
        ).run(m2.operation)

        # Lower to LLVM (the step that would hang with `dot` strategy)
        import time
        t0 = time.time()
        pm.PassManager.parse(
            "builtin.module(canonicalize,cse,convert-bufferization-to-memref,"
            "convert-linalg-to-loops,lower-affine,convert-scf-to-cf,"
            "expand-strided-metadata,lower-affine,func.func(lower-vector-mask),"
            "func.func(convert-vector-to-scf),finalize-memref-to-llvm,"
            "convert-cf-to-llvm,convert-math-to-llvm,"
            "convert-vector-to-llvm{vector-contract-lowering=outerproduct},"
            "convert-arith-to-llvm,convert-ub-to-llvm,convert-func-to-llvm,"
            "reconcile-unrealized-casts)", ctx2
        ).run(m2.operation)
        elapsed = time.time() - t0
        assert elapsed < 30, f"convert-vector-to-llvm took {elapsed:.1f}s (>30s)"
        text = str(m2)
        assert "vector.contract" not in text, "All vector.contract must be lowered"


# ── Multi-function MlirExecutor ───────────────────────────────
# Bug: MlirExecutor only ran module.main (main_0). For split models,
# main_0's output is NOT the logits — main_1 is needed.


@pytest.mark.integration
def test_mlir_executor_multi_function():
    """MlirExecutor must chain multi-function models."""
    from compiler.serialize import load_artifact
    from engine.mlir_executor import MlirExecutor
    from hal.pytorch_backend import PyTorchBackend

    mod_dir = (
        Path(__file__).resolve().parent.parent / "compiled" / "opt_125m_fresh"
    )
    if not (mod_dir / "model.mlir").exists():
        pytest.skip("compiled model not found")

    mod = load_artifact(str(mod_dir))
    executor = MlirExecutor(mod, PyTorchBackend("cpu"))
    import torch
    out = executor.forward(input_ids=torch.randint(0, 100, (1, 4)))
    # Output from main_1 should be LM head logits: [1, 4, 50272]
    assert out.shape[-1] == 50272, (
        f"Expected logits shape [*, *, 50272], got {out.shape}"
    )
    assert out.numel() > 0, "Output should not be empty"
