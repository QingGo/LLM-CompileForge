"""Bisect accuracy: JIT (ExecutionEngine) vs full pipeline vs HF.

Levels:
  1. HF reference (baseline, cos=1.0)
  2. Python executor (sf dialect ops)  → cos 0.999999
  3. MLIR ExecutionEngine JIT of lowered+LLVMIR  → cos ?
  4. Compiled .dylib (llc + ld)         → cos 0.865

If L3 ≈ L2 → bug is in llc/linking (LLVM codegen)
If L3 ≈ L4 → bug is in MLIR→LLVM pipeline (bufferize/lowering)
If L3 < 0.999 → bug is in C++ lowering or vectorization
"""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_mlir_path():
    _pkg = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
    if _pkg.is_dir() and str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))


def cosine_similarity(a, b):
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))


def make_memref_f32(arr):
    """Build a double-pointer memref arg for ExecutionEngine.invoke()."""
    from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
    desc = get_ranked_memref_descriptor(arr)
    inner = ctypes.pointer(desc)
    outer = ctypes.pointer(inner)
    return inner, outer


def read_memref_output(inner_ptr, arr_dummy):
    """Read output from a memref descriptor written by JIT."""
    from mlir.runtime.np_to_memref import get_ranked_memref_descriptor
    clone = get_ranked_memref_descriptor(arr_dummy)
    ctypes.memmove(
        ctypes.addressof(clone),
        ctypes.cast(inner_ptr, ctypes.c_void_p).value,
        ctypes.sizeof(clone),
    )
    return np.ctypeslib.as_array(clone.aligned, shape=tuple(clone.shape)).copy()


def add_emit_c_interface(module, ctx):
    import mlir.ir as ir
    def _cb(op):
        if hasattr(op, "name") and op.name == "func.func":
            with ctx:
                op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        return ir.WalkResult.ADVANCE
    module.operation.walk(_cb)


def main():
    artifact_dir = "./compiled/opt_125m_v8"
    baseline_dir = os.path.join(artifact_dir, "baselines")

    # Load baselines
    print("=== Loading baselines ===")
    hf_logits = np.load(os.path.join(baseline_dir, "hf_logits.npy"))
    py_logits = np.load(os.path.join(baseline_dir, "python_executor_logits.npy"))
    print(f"  HF:      {hf_logits.shape}, first={hf_logits[0,0,0]:.6f}")
    print(f"  Python:  {py_logits.shape}, first={py_logits[0,0,0]:.6f}")
    sim_py = cosine_similarity(hf_logits, py_logits)
    print(f"  cos(Python vs HF): {sim_py:.10f}")

    # Build JIT input: same as what the Rust executor uses
    # [2, 32826, 85, 4129, 0, 0, 0, 0] reshaped to [2, 4]
    input_ids = np.array([[2, 32826, 85, 4129], [0, 0, 0, 0]], dtype=np.int64)

    # We need to load weights from pytorch_model.bin to match the Python executor
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]  # new snapshot has model.safetensors
    st_path = os.path.join(snapshots, snap, "model.safetensors")

    # Load safetensors into a name→tensor dict (matching hf_key_map)
    import safetensors.torch
    st = safetensors.torch.load_file(st_path)
    import json
    with open(os.path.join(artifact_dir, "metadata.json")) as f:
        meta = json.load(f)
    hfk = meta.get("hf_key_map", {})

    # Build weight dict: compiled_name → numpy f32 tensor
    weights = {}
    for compiled_name, hf_key in hfk.items():
        if hf_key in st:
            weights[compiled_name] = st[hf_key].to(torch.float32).numpy()
    print(f"  Loaded {len(weights)} weights from safetensors")

    # ── Test via ExecutionEngine JIT ──
    print("\n=== JIT execution path ===")
    _setup_mlir_path()
    import mlir.ir as ir
    import mlir.passmanager as pm
    from mlir.execution_engine import ExecutionEngine
    from mlir._mlir_libs import _mlirRegisterEverything

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx.append_dialect_registry(reg)

    # Load the lowered module
    lowered_path = os.path.join(artifact_dir, "model.lowered.mlir")
    with open(lowered_path) as f:
        lowered_text = f.read()

    with ir.Location.unknown(ctx):
        module = ir.Module.parse(lowered_text, ctx)

        # Run vectorization + bufferize + lowering pipeline
        from compiler.mlir_dialect.llvm_backend import lower_linalg_to_llvm_ir, _vectorize_via_transform
        try:
            llvm_text = lower_linalg_to_llvm_ir(module)
            print(f"  Pipeline OK, LLVM text: {len(llvm_text)} chars")
        except Exception as e:
            print(f"  Pipeline FAILED: {e}")
            return

    # Re-parse the LLVM dialect module
    llvm_ctx = ir.Context()
    llvm_ctx.allow_unregistered_dialects = True
    with ir.Location.unknown(llvm_ctx):
        llvm_mod = ir.Module.parse(llvm_text, llvm_ctx)
        add_emit_c_interface(llvm_mod, llvm_ctx)
        # Add func-to-llvm + reconcile-unrealized-casts
        try:
            pm.PassManager.parse(
                "builtin.module(convert-func-to-llvm,reconcile-unrealized-casts)",
                llvm_ctx
            ).run(llvm_mod.operation)
        except Exception as e:
            print(f"  func-to-llvm FAILED: {e}")
            return

        # Try JIT
        print("  Creating ExecutionEngine...")
        try:
            engine = ExecutionEngine(llvm_mod, opt_level=0)
        except Exception as e:
            print(f"  ExecutionEngine FAILED: {e}")
            return

        # For JIT, we need to call main_0 with ALL weight inputs.
        # This is extremely complex (215 memref args).
        # Let's try a simpler approach: call a single matmul to verify
        # the JIT produces correct results for basic ops.
        print("  ExecutionEngine created successfully")

        # Check how many functions are available
        mod_str = str(llvm_mod)
        n_funcs = mod_str.count("llvm.func @")
        print(f"  Functions in LLVM module: {n_funcs}")

    # ── Test a single matmul through the full lowered+compiled path ──
    print("\n=== Layer-by-layer bisect ===")

    # Instead of running the full model through JIT (which needs all 215 inputs),
    # let's compare the Rust executor output vs Python executor by looking at
    # per-layer hidden states if we had them.
    # For now, let's just check the Rust logits more carefully.
    rust_csv = np.loadtxt("/tmp/rust_logits.csv", delimiter=",")
    rust_logits = rust_csv.reshape(hf_logits.shape)
    sim_rust = cosine_similarity(hf_logits, rust_logits)
    print(f"  cos(Rust dylib vs HF): {sim_rust:.10f}")

    # ── Compare: does the lowered module (before LLVM) match? ──
    # We can't easily run the lowered module through JIT with all 215 inputs.
    # But we CAN check if the Python executor with OFF-THE-SHELF ops matches.
    # Let's instead focus on one specific operation.
    print("\n=== Checking specific ops ===")

    # Check if the output has the right overall statistics vs Python
    # If the means/std differ drastically, the issue is systematic.
    print(f"  HF:     mean={hf_logits.mean():.4f}, std={hf_logits.std():.4f}")
    print(f"  Python: mean={py_logits.mean():.4f}, std={py_logits.std():.4f}")
    print(f"  Rust:   mean={rust_logits.mean():.4f}, std={rust_logits.std():.4f}")

    # Check per-position cosine
    print("\n  Per-position cos (Rust vs HF):")
    for pos in range(4):
        for b in range(2):
            s = cosine_similarity(rust_logits[b, pos], hf_logits[b, pos])
            print(f"    batch={b}, pos={pos}: cos={s:.8f}")

    # ── Conclusion ──
    print("\n=== Summary ===")
    print(f"  Python executor (sf dialect): cos={sim_py:.10f} (≥0.999)")
    print(f"  Rust compiled (.dylib):       cos={sim_rust:.10f}")
    print(f"  Gap: {sim_py - sim_rust:.10f}")
    print()
    print("  Possible root causes:")
    print("  1. C++ sf→linalg lowering: linalg.generic reduction accumulation order")
    print("     (arith.addf is non-associative; differs from PyTorch's cublas)")
    print("  2. Constant folding: canonicalize may fold expressions differently")
    print("  3. ONESHOT bufferize: may introduce extra copy or fill ops")
    print("  4. convert-linalg-to-loops: scalar loops have different FP accumulation")
    print("  5. LLVM codegen: -ffast-math or vectorization changes FMA behavior")


if __name__ == "__main__":
    main()
