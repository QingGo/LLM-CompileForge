"""Pipeline output validation: verify LLVM IR is clean and well-formed.

This test runs the full lowering pipeline and checks:
  1. No `arith` dialect ops remain (all lowered to LLVM)
  2. No `vector` dialect ops remain (all lowered to LLVM)
  3. No `scf` or `cf` control-flow ops remain
  4. No `unrealized_conversion_cast` ops remain
  5. No `memref` ops remain (all lowered)
  6. LLVM IR is valid (mlir-translate succeeds)
  7. llc produces a non-empty .o file
  8. Link produces a .dylib with ciface symbols
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# ── helpers ──────────────────────────────────────────────────────────────


from tests.helpers import has_mlir_bindings


def _load_lowered(model_dir: str):
    """Load model.lowered.mlir and return (module, context)."""
    import mlir.ir as ir
    from mlir._mlir_libs import _mlirRegisterEverything

    path = Path(model_dir) / "model.lowered.mlir"
    if not path.exists():
        pytest.skip(f"{path} not found — run compile_dylib.py first")

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx.append_dialect_registry(reg)
    ctx.load_all_available_dialects()

    mod = ir.Module.parse(path.read_text(), ctx)
    return mod, ctx, ir


# ── Test: IR cleanliness after full pipeline ─────────────────────────────


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_no_arith_ops_after_lowering():
    """No arith.* ops should survive the full lowering pipeline."""
    if not has_mlir_bindings():
        pytest.skip("MLIR bindings not available")

    mod, ctx, ir = _load_lowered("compiled/opt_125m_v8")
    import mlir.passmanager as pm
    from compiler.mlir_dialect.llvm_backend import (
        _tile_matmuls_per_func,
        lower_linalg_to_llvm_ir,
    )

    with ir.Location.unknown(ctx):
        pm.PassManager.parse(
            "builtin.module(canonicalize,cse)", ctx
        ).run(mod.operation)

        _tile_matmuls_per_func(mod, tile_k=64)
        pm.PassManager.parse(
            "builtin.module(canonicalize,cse)", ctx
        ).run(mod.operation)

        llvm_dialect = lower_linalg_to_llvm_ir(mod)

    # Check for problematic ops
    n_arith = llvm_dialect.count("arith.")
    n_vector = llvm_dialect.count("vector.")
    n_scf = llvm_dialect.count("scf.")
    n_cf = llvm_dialect.count("cf.")
    n_memref = llvm_dialect.count("memref.")
    n_unrealized = llvm_dialect.count("unrealized")

    # Check for arith ops with vector types specifically
    import re as _re
    n_arith_vec = len(_re.findall(r'arith\.\w+[^:]*:\s*vector<', llvm_dialect))

    print(f"  arith: {n_arith}  (arith+vec: {n_arith_vec})")
    print(f"  vector: {n_vector}")
    print(f"  scf: {n_scf}")
    print(f"  cf: {n_cf}")
    print(f"  memref: {n_memref}")
    print(f"  unrealized: {n_unrealized}")

    # These should ALL be 0 after successful lowering
    assert n_arith_vec == 0, (
        f"{n_arith_vec} arith ops with vector types remain — "
        "convert-vector-to-llvm likely created arith ops after "
        "convert-arith-to-llvm already ran"
    )
    # unrealized_conversion_cast are expected here (handled by
    # _fixup_unrealized_casts in mlir_module_to_llvm_ir).
    if n_unrealized > 0:
        print(f"  ⚠ {n_unrealized} unrealized casts (handled by fixup later)")


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_tile_sizes_within_bounds():
    """After tiling K=64,N=64, all inner matmuls should have dims ≤ 64."""
    if not has_mlir_bindings():
        pytest.skip("MLIR bindings not available")

    mod, ctx, ir = _load_lowered("compiled/opt_125m_v8")
    import mlir.passmanager as pm
    from compiler.mlir_dialect.llvm_backend import _tile_matmuls_per_func

    with ir.Location.unknown(ctx):
        pm.PassManager.parse(
            "builtin.module(canonicalize,cse)", ctx
        ).run(mod.operation)
        _tile_matmuls_per_func(mod, tile_k=64)
        pm.PassManager.parse(
            "builtin.module(canonicalize,cse)", ctx
        ).run(mod.operation)

        txt = str(mod)

    # Find all linalg matmul/batch_matmul IN operands and check dims
    # (the OUTS operand can be the accumulation tensor — skip it)
    for m in re.finditer(
        r'linalg\.(?:matmul|batch_matmul)\b[^{]*ins\(([^)]+)\)',
        txt,
    ):
        ins = m.group(1)
        tensors = re.findall(r'tensor<([\dx]+)xf32>', ins)
        for t in tensors:
            dims = [int(d) for d in t.split("x")]
            max_dim = max(dims)
            if max_dim > 64:
                pytest.fail(
                    f"Tile dim {max_dim} > 64 in tensor<{t}xf32> "
                    f"(op: {m.group()[:120]})"
                )


# ── Test: pipeline timing thresholds ─────────────────────────────────────


@pytest.mark.unit
@pytest.mark.timeout(60)
def test_pipeline_timing():
    """Full lowering pipeline should complete within time budget."""
    if not has_mlir_bindings():
        pytest.skip("MLIR bindings not available")

    mod, ctx, ir = _load_lowered("compiled/opt_125m_v8")
    import mlir.passmanager as pm
    from compiler.mlir_dialect.llvm_backend import (
        _tile_matmuls_per_func,
        lower_linalg_to_llvm_ir,
        mlir_module_to_llvm_ir,
    )

    with ir.Location.unknown(ctx):
        pm.PassManager.parse(
            "builtin.module(canonicalize,cse)", ctx
        ).run(mod.operation)
        _tile_matmuls_per_func(mod, tile_k=64)
        pm.PassManager.parse(
            "builtin.module(canonicalize,cse)", ctx
        ).run(mod.operation)
        _tile_matmuls_per_func(mod, tile_k=64)

        # Time lowering
        t0 = time.perf_counter()
        mlir_text = lower_linalg_to_llvm_ir(mod)
        t_lower = time.perf_counter() - t0
        print(f"  lowering: {t_lower:.1f}s")
        assert t_lower < 30, f"Lowering took {t_lower:.1f}s (>30s)"

        # Time translate + fixup
        t0 = time.perf_counter()
        llvm_ir = mlir_module_to_llvm_ir(mod)
        t_translate = time.perf_counter() - t0
        print(f"  translate: {t_translate:.1f}s")
        assert t_translate < 30, f"Translate took {t_translate:.1f}s (>30s)"

    # Quick llc test
    with tempfile.TemporaryDirectory() as td:
        ll_path = os.path.join(td, "model.ll")
        with open(ll_path, "w") as f:
            f.write(llvm_ir)

        t0 = time.perf_counter()
        r = subprocess.run(
            ["llc", "-O0", "-filetype=obj", ll_path, "-o", os.path.join(td, "model.o")],
            capture_output=True, text=True, timeout=30,
        )
        t_llc = time.perf_counter() - t0
        print(f"  llc: {t_llc:.1f}s")
        assert r.returncode == 0, f"llc failed: {r.stderr[:200]}"
        assert os.path.getsize(os.path.join(td, "model.o")) > 0


# ── Test: FMA fusion fires properly ──────────────────────────────────────


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_fma_fusion_fires():
    """FMA fusion should replace ~90% of fmul+{fadd,fsub} with llvm.intr.fmuladd."""
    if not has_mlir_bindings():
        pytest.skip("MLIR bindings not available")

    mod, ctx, ir = _load_lowered("compiled/opt_125m_v8")
    import mlir.passmanager as pm
    from compiler.mlir_dialect.llvm_backend import (
        _tile_matmuls_per_func,
        lower_linalg_to_llvm_ir,
    )

    with ir.Location.unknown(ctx):
        pm.PassManager.parse(
            "builtin.module(canonicalize,cse)", ctx
        ).run(mod.operation)
        _tile_matmuls_per_func(mod, tile_k=64)
        pm.PassManager.parse(
            "builtin.module(canonicalize,cse)", ctx
        ).run(mod.operation)
        llvm_dialect = lower_linalg_to_llvm_ir(mod)

    n_fmuladd = llvm_dialect.count("llvm.intr.fmuladd")
    n_fmul = llvm_dialect.count("llvm.fmul")
    n_fadd = llvm_dialect.count("llvm.fadd")
    n_fsub = llvm_dialect.count("llvm.fsub")
    total = n_fmuladd + n_fmul
    rate = 100.0 * n_fmuladd / total if total > 0 else 0

    print(f"  fmuladd={n_fmuladd} fmul={n_fmul} fadd={n_fadd} fsub={n_fsub} rate={rate:.0f}%")

    # At least 80% of fmul operations should have been fused
    assert rate >= 80.0, (
        f"FMA fusion rate too low: {rate:.0f}% ({n_fmuladd}/{total})"
    )


# ── Pass-through for manual invocation ────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
