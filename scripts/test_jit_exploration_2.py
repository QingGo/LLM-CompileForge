"""Exploration: Test MLIR ExecutionEngine JIT viability — full model.

Continues from scripts/test_jit_exploration.py.
Tests whether the full model.lowered.mlir can pass through the
lowering pipeline to ExecutionEngine JIT compilation.

Usage:
  source .venv/bin/activate
  python scripts/test_jit_exploration_2.py
"""

from __future__ import annotations

import time

from scripts.test_jit_exploration import (
    COMPILED,
    add_emit_c_interface,
)

# ── Test 2: Load model.lowered.mlir and JIT ─────────────────────────


def test_model_lowered_jit():
    """Try to load model.lowered.mlir through pipeline to ExecutionEngine."""
    print("\n" + "=" * 70)
    print("TEST 2: Load model.lowered.mlir → lower → ExecutionEngine")
    print("=" * 70)

    import mlir.ir as ir
    import mlir.passmanager as pm
    from mlir._mlir_libs import _mlirRegisterEverything
    from mlir.execution_engine import ExecutionEngine

    lowered_path = COMPILED / "model.lowered.mlir"
    if not lowered_path.exists():
        print(f"  ❌ {lowered_path} not found")
        return False

    lowered_text = lowered_path.read_text()
    n_funcs = lowered_text.count("func.func")
    n_ops = len(lowered_text)
    print(f"  Loaded model.lowered.mlir: {n_funcs} functions, {n_ops} chars")

    # Create context
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx.append_dialect_registry(reg)

    # Parse the lowered module
    with ir.Location.unknown(ctx):
        module = ir.Module.parse(lowered_text, ctx)
        print(f"  Parsed {len(list(module.body))} blocks")

    # Run the full lower_linalg_to_llvm_ir pipeline
    print("  Running lower_linalg_to_llvm_ir pipeline...")
    try:
        from compiler.mlir_dialect.llvm_backend import lower_linalg_to_llvm_ir

        t0 = time.time()
        llvm_text = lower_linalg_to_llvm_ir(module)
        dt = time.time() - t0
        print(f"    Pipeline completed in {dt:.1f}s")
        print(f"    Output LLVM IR: {len(llvm_text)} chars")
    except Exception as e:
        print(f"    ❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Re-parse the LLVM dialect module
    llvm_ctx = ir.Context()
    llvm_ctx.allow_unregistered_dialects = True
    with ir.Location.unknown(llvm_ctx):
        llvm_mod = ir.Module.parse(llvm_text, llvm_ctx)
        add_emit_c_interface(llvm_mod, llvm_ctx)

        # Run func-to-llvm + reconcile-unrealized-casts
        try:
            pm.PassManager.parse(
                "builtin.module(convert-func-to-llvm,reconcile-unrealized-casts)",
                llvm_ctx,
            ).run(llvm_mod.operation)
            print("    ✅ func-to-llvm + casts reconciled")
        except Exception as e:
            print(f"    ❌ func-to-llvm failed: {e}")
            return False

        # Count functions
        mod_str = str(llvm_mod)
        n_funcs_llvm = mod_str.count("llvm.func @")
        n_funcs_total = mod_str.count("func @")
        print(f"    LLVM functions: {n_funcs_llvm}")
        print(f"    Total functions: {n_funcs_total}")

        # Try creating ExecutionEngine
        print("  Creating ExecutionEngine (this may take a while)...")
        try:
            t0 = time.time()
            ExecutionEngine(llvm_mod, opt_level=0)
            dt = time.time() - t0
            print(f"    ✅ ExecutionEngine created in {dt:.1f}s")
            print("    JIT compilation of full model WORKS")
            print()
            print("    NOTE: The model's main_0 function has ~215 memref args.")
            print("    Calling it via JIT requires constructing all those args")
            print("    as ctypes memref descriptors — extremely complex.")
            print()
            print("    But the JIT CAN compile and is available.")
            print("    Next step would be layer-by-layer JIT testing.")
            return True
        except Exception as e:
            print(f"    ❌ ExecutionEngine creation failed: {e}")
            return False


# ── Main ─────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("MLIR ExecutionEngine JIT Exploration — Full Model")
    print("=" * 70)
    print(f"Compiled artifacts: {COMPILED}")

    r = test_model_lowered_jit()
    status = "✅" if r is True else "❌"
    print(f"\n  {status} model_lowered_jit")


if __name__ == "__main__":
    main()
