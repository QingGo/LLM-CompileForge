"""Test: rebuild .dylib WITHOUT FMA fusion to diagnose cos degradation.

Usage: source .venv/bin/activate && python scripts/test_no_fma_rebuild.py

Effect: Loads model.lowered.mlir (already sf→linalg lowered), runs the
LLVM lowering pipeline with FMA fusion disabled, and compiles to .dylib.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

logging.basicConfig(level=logging.INFO)

from compiler.mlir_dialect.lowering.llvm_backend import compile_module_to_dylib, lower_linalg_to_llvm_ir  # noqa: E402

COMPILED_DIR = "compiled/opt_125m_fresh"
MODEL_NAME = "opt_125m"


def main():
    import mlir.ir as ir

    lowered_path = Path(COMPILED_DIR) / "model.lowered.mlir"
    if not lowered_path.exists():
        print(f"ERROR: {lowered_path} not found. Run make rebuild-mlir first.")
        return 1

    lowered_text = lowered_path.read_text()
    print(f"Loaded model.lowered.mlir ({len(lowered_text)} chars)")

    # Step 5: LLVM lowering (no FMA — _make_fma_stage is no-op) + .dylib
    ctx_llvm = ir.Context()
    ctx_llvm.allow_unregistered_dialects = True
    with ctx_llvm:
        ir_mod = ir.Module.parse(lowered_text, ctx_llvm)
        print("Running LLVM lowering (FMA disabled)...")
        lower_linalg_to_llvm_ir(ir_mod)
        print("LLVM lowering succeeded")

    dylib_path = compile_module_to_dylib(
        ir_mod,
        str(Path(COMPILED_DIR).resolve()),
        model_name=MODEL_NAME,
    )

    print(f"\nCompilation complete: {dylib_path}")
    for fname in [f"lib{MODEL_NAME}.dylib", "constants.bin"]:
        fpath = Path(COMPILED_DIR) / fname
        if fpath.exists():
            print(f"  OK {fpath} ({fpath.stat().st_size} bytes)")
        else:
            print(f"  MISSING {fpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
