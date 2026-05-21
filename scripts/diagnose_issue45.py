"""Diagnose Issue #45: Rust forward cos=0.525 vs Python reference.

Provides a rich feedback loop with multiple signals:
  - Per-token cosine similarity
  - Argmax per token
  - Top-5 logit values comparison
  - Optional ctypes .dylib direct call (to isolate bug location)

Usage:
    python scripts/diagnose_issue45.py                      # full diagnose
    python scripts/diagnose_issue45.py --save-ref           # save Python reference
    python scripts/diagnose_issue45.py --skip-rust          # only show Python vs HF
    python scripts/diagnose_issue45.py --ctypes             # also try ctypes direct call
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compiler.serialize import load_artifact
from engine.mlir_executor import MlirExecutor
from hal.pytorch_backend import PyTorchBackend
from utils.logging import init_logging

_log = None

INPUT_IDS = [2, 32826, 85, 4129]  # must match m1_tests.rs
ARTIFACT_DIR = "./compiled/opt_125m_fresh"
REF_DIR = os.path.join(ARTIFACT_DIR, "baselines")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return float(np.dot(a_f, b_f) / (denom + 1e-12))


def _patch_transformers_torch():
    import transformers.utils.generic as _generic
    import transformers.utils.import_utils as _iu
    _iu._torch_available = True
    _iu._torch_version = torch.__version__
    _generic._torch_pytree = torch.utils._pytree
    def _flatten(output):
        return list(output.values()), list(output.keys())
    def _unflatten(values, context, output_type=None):
        return (output_type or type(context[0]))(**dict(zip(context, values, strict=False)))
    _generic._model_output_flatten = _flatten
    _generic._model_output_unflatten = _unflatten


def get_hf_reference() -> np.ndarray:
    _patch_transformers_torch()
    from transformers.models.opt.configuration_opt import OPTConfig
    from transformers.models.opt.modeling_opt import OPTForCausalLM
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    model_path = os.path.join(snapshots, snap, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    config_path = os.path.join(snapshots, snap, "config.json")
    config = OPTConfig.from_pretrained(config_path)
    model = OPTForCausalLM(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long)  # [1, 4]
    with torch.no_grad():
        output = model(input_ids)
    return output.logits.detach().numpy()  # [1, 4, 50272]


def get_python_executor_logits() -> np.ndarray:
    mod = load_artifact(ARTIFACT_DIR)
    backend = PyTorchBackend("cpu")
    executor = MlirExecutor(mod, backend)
    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long)  # [1, 4]
    logits = executor.forward(input_ids)
    return logits.detach().numpy()  # [1, 4, 50272]


def save_reference():
    os.makedirs(REF_DIR, exist_ok=True)
    print("\n=== Generating Python Executor reference ===")
    py_logits = get_python_executor_logits()
    py_path = os.path.join(REF_DIR, "python_ref_logits.npy")
    np.save(py_path, py_logits)
    print(f"  Shape: {py_logits.shape}, First: {py_logits[0,0,0]:.6f}, Saved: {py_path}")

    print("\n=== Generating HuggingFace reference ===")
    hf_logits = get_hf_reference()
    hf_path = os.path.join(REF_DIR, "hf_ref_logits.npy")
    np.save(hf_path, hf_logits)
    print(f"  Shape: {hf_logits.shape}, First: {hf_logits[0,0,0]:.6f}, Saved: {hf_path}")

    sim = cosine_similarity(hf_logits, py_logits)
    print(f"\n  Cosine(Python Executor vs HF): {sim:.10f}")
    if sim > 0.999:
        print("  ✅ PASS: Python executor > 0.999")
    else:
        print("  ❌ FAIL: Python executor below 0.999")
    return py_logits, hf_logits


def load_reference() -> tuple[np.ndarray, np.ndarray]:
    """Return (py_ref, hf_ref)."""
    py_path = os.path.join(REF_DIR, "python_ref_logits.npy")
    hf_path = os.path.join(REF_DIR, "hf_ref_logits.npy")
    if os.path.exists(py_path) and os.path.exists(hf_path):
        return np.load(py_path), np.load(hf_path)
    print("  Reference files not found. Run with --save-ref first.")
    return None, None


def run_rust_forward() -> np.ndarray | None:
    """Run Rust forward test and return logits from /tmp/rust_logits.csv."""
    manifest_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rust_dir = os.path.join(manifest_dir, "rust")

    print("\n=== Running Rust forward ===")
    result = subprocess.run(
        ["cargo", "test", "test_opt_125m_forward_runs", "--", "--nocapture"],
        cwd=rust_dir,
        capture_output=True, text=True,
        timeout=120,
    )
    for line in result.stdout.splitlines():
        print(f"  [Rust] {line}")
    for line in result.stderr.splitlines():
        print(f"  [Rust stderr] {line}")

    if result.returncode != 0 and "FAILED" in result.stdout:
        print(f"  Rust test FAILED (exit={result.returncode})")

    csv_path = "/tmp/rust_logits.csv"
    if not os.path.exists(csv_path):
        print(f"  ❌ No logits CSV at {csv_path}")
        return None

    rust_raw = np.loadtxt(csv_path, delimiter=",")
    print(f"  Rust logits loaded: shape={rust_raw.shape}, first={rust_raw.flat[0]:.6f}, last={rust_raw.flat[-1]:.6f}")
    # Rust output shape: depends on model. OPT-125m uses batch=1, fixme?
    return rust_raw


def reshape_rust_logits(rust_logits: np.ndarray, ref_shape: tuple) -> np.ndarray:
    """Try to reshape Rust logits to match reference shape [B, S, V]."""
    if rust_logits.shape == ref_shape:
        return rust_logits
    n_elem = rust_logits.size
    b, s, v = ref_shape
    if n_elem == b * s * v:
        return rust_logits.reshape(ref_shape)
    if n_elem == 2 * s * v:
        # Rust sometimes outputs batch=2 (due to static batch dim in lowering)
        return rust_logits.reshape(2, s, v)[0:1]
    print(f"  ⚠ Cannot reshape Rust logits {rust_logits.shape} to ref {ref_shape}")
    return rust_logits


def diagnose_per_token(rust: np.ndarray, ref: np.ndarray, label: str):
    """Print per-token detailed diagnosis."""
    print(f"\n=== Per-token diagnosis ({label}) ===")
    for tok in range(rust.shape[1]):
        r_tok = rust[0, tok]
        ref_tok = ref[0, tok]

        sim = cosine_similarity(r_tok, ref_tok)
        r_argmax = int(np.argmax(r_tok))
        ref_argmax = int(np.argmax(ref_tok))
        r_top5 = np.argsort(r_tok)[-5:][::-1]
        ref_top5 = np.argsort(ref_tok)[-5:][::-1]

        print(f"  token[{tok}]:")
        print(f"    cosine={sim:.8f}")
        print(f"    Rust argmax={r_argmax} (top5={r_top5.tolist()})")
        print(f"    Ref  argmax={ref_argmax} (top5={ref_top5.tolist()})")
        if r_argmax != ref_argmax:
            print(f"    ❌ ARGMAX MISMATCH: Rust={r_argmax} vs Ref={ref_argmax}")
            r_val = r_tok[r_argmax]
            ref_val_at_r = r_tok[ref_argmax]
            print(f"    Rust[{r_argmax}]={r_val:.4f}, Rust[{ref_argmax}]={ref_val_at_r:.4f}")


def diagnose_stats(rust: np.ndarray, ref: np.ndarray):
    """Print global stats comparison."""
    print("\n=== Global stats ===")
    for name, arr in [("Rust", rust), ("Ref", ref)]:
        f = arr.ravel().astype(np.float64)
        print(f"  {name}: mean={f.mean():.4f} std={f.std():.4f} min={f.min():.4f} max={f.max():.4f}")


def diagnose_layers(rust_dump_dir: str) -> None:
    """Compare per-function outputs between Rust dump and reference.

    Args:
        rust_dump_dir: Directory containing DUMP_LAYERS output (func_*.npy)
    """
    files = sorted(glob.glob(os.path.join(rust_dump_dir, "func_*.npy")))
    print("\n=== Per-Layer Diagnosis ===")
    print(f"Found {len(files)} function outputs in {rust_dump_dir}")

    if not files:
        print("No DUMP_LAYERS output found. Run with DUMP_LAYERS set.")
        return

    by_func = defaultdict(list)
    for f in files:
        basename = os.path.basename(f).replace('.npy', '')
        parts = basename.split('_')
        func_idx = int(parts[1])
        call_idx = int(parts[2])
        by_func[func_idx].append((call_idx, f))

    print(f"Unique functions: {len(by_func)}")
    for fid in sorted(by_func.keys()):
        calls = sorted(by_func[fid], key=lambda x: x[0])
        print(f"  func_{fid}: {len(calls)} calls")

        first_arr = np.load(calls[0][1])
        last_arr = np.load(calls[-1][1])
        print(f"    First call: {first_arr.shape}, last call: {last_arr.shape}")

    print("\nNote: Full Python vs Rust per-layer comparison requires")
    print("Python-side function dump (add --py-dump to MlirExecutor)")
    print("For now, this validates the DUMP_LAYERS mechanism itself.")


def call_dylib_ctypes() -> np.ndarray | None:
    """Try calling the compiled .dylib directly via ctypes."""
    import ctypes
    from ctypes import c_int64, c_void_p

    dylib_path = os.path.join(ARTIFACT_DIR, "libopt_125m.dylib")
    if not os.path.exists(dylib_path):
        print(f"  ❌ No .dylib at {dylib_path}")
        return None

    print("\n=== ctypes direct call to .dylib ===")
    ctypes.CDLL(dylib_path)

    # Check symbol availability for main_0
    try:
        pass
    except AttributeError:
        print("  ❌ _mlir_ciface_main_0 not found in dylib")
        return None

    # Build input: rank-2 i64 memref descriptor for [1, 4]
    input_np = np.array([INPUT_IDS], dtype=np.int64)  # [1, 4]
    p = input_np.ctypes.data_as(c_void_p)

    class MemRef2(ctypes.Structure):
        _fields_ = [
            ("allocated", c_void_p),
            ("aligned", c_void_p),
            ("offset", c_int64),
            ("sizes", c_int64 * 2),
            ("strides", c_int64 * 2),
        ]

    MemRef2(
        allocated=p, aligned=p, offset=0,
        sizes=(c_int64 * 2)(1, 4),
        strides=(c_int64 * 2)(4, 1),
    )

    print("\n  ┌─ Three-way comparison (ctypes mode) ──────────────────────────")
    print("  │")
    print("  │  This script would isolate which layer introduces the bug:")
    print("  │")
    print("  │  Python Executor ──── cos(Py, HF) ────► HuggingFace (reference)")
    print("  │       │                                      validates compilation")
    print("  │       │                                      pipeline lowering")
    print("  │       │")
    print("  │    cos(Rust, Py)")
    print("  │       │")
    print("  │       ▼")
    print("  │  Rust Executor ───── cos(Rust, ctypes) ──► ctypes (.dylib)")
    print("  │                              isolates Rust runtime vs dylib")
    print("  │")
    print("  │  cos(Py, HF)          — validates compilation pipeline")
    print("  │  cos(Rust, ctypes)    — isolates Rust executor vs dylib")
    print("  │  cos(ctypes, Py)      — isolates dylib vs Python executor")
    print("  │")
    print("  │  Currently ctypes is stub (needs all 215 weight args).")
    print("  │  Use `--skip-rust` to test Python pipeline separately.")
    print("  │")
    print("  └────────────────────────────────────────────────────────────")

    if os.path.exists("/tmp/rust_logits.csv"):
        rust_raw = np.loadtxt("/tmp/rust_logits.csv", delimiter=",")
        ref_path = os.path.join(REF_DIR, "python_ref_logits.npy")
        if os.path.exists(ref_path):
            py_ref = np.load(ref_path)
            r = rust_raw.ravel().astype(np.float64)
            p = py_ref.ravel().astype(np.float64)
            denom = float(np.linalg.norm(r) * np.linalg.norm(p))
            c = float(np.dot(r, p) / (denom + 1e-12))
            print(f"\n  cos(Rust vs Python) from /tmp/rust_logits.csv: {c:.6f}")
            if c < 0.99:
                print("  → Bug is between Python executor output and Rust executor output.")
                print("  → To further isolate: implement ctypes call to compare")
                print("    Rust runtime vs dylib directly.")
            else:
                print("  → Rust matches Python. Bug is in Python compilation pipeline vs HF.")
        else:
            print("  (python_ref_logits.npy not found — run with --save-ref first)")
    else:
        print("  (/tmp/rust_logits.csv not found — run Rust forward to get data)")

    return None


def main():
    global _log
    init_logging()
    _log = logging.getLogger("diagnose_issue45")

    parser = argparse.ArgumentParser(description="Diagnose Issue #45")
    parser.add_argument("--save-ref", action="store_true", help="Save Python + HF reference logits")
    parser.add_argument("--skip-rust", action="store_true", help="Skip Rust forward run")
    parser.add_argument("--fast", action="store_true",
        help="Skip Rust cargo test, read pre-computed logits from /tmp/rust_logits.csv")
    parser.add_argument("--ctypes", action="store_true", help="Also try ctypes direct call")
    parser.add_argument("--layers", type=str, nargs="?", const="/tmp/ld_rs", default=None,
        help="Run per-layer diagnosis using DUMP_LAYERS output dir")
    args = parser.parse_args()

    print("=" * 60)
    print("Issue #45 Diagnose: Rust forward cos=0.525 vs Python ref")
    print(f"Input: {INPUT_IDS}")
    print("=" * 60)

    # Step 1: Load or generate reference
    if args.save_ref:
        py_ref, hf_ref = save_reference()
    else:
        py_ref, hf_ref = load_reference()

    if py_ref is None:
        print("\n❌ No reference found. Run with --save-ref first.")
        return 1

    print(f"\nPython ref shape: {py_ref.shape}, First: {py_ref[0,0,0]:.6f}")
    print(f"HF ref shape:      {hf_ref.shape}, First: {hf_ref[0,0,0]:.6f}")

    # Python vs HF baseline check
    sim_py_hf = cosine_similarity(py_ref, hf_ref)
    print(f"\n  Cosine(Python Executor vs HF): {sim_py_hf:.10f}")
    if sim_py_hf > 0.999:
        print("  ✅ Python executor baseline good")
    else:
        print("  ⚠ Python executor degraded from HF — pipeline issue, not Rust-only")
        diagnose_per_token(py_ref, hf_ref, "Python vs HF")

    # Step 2: Run Rust forward
    if not args.skip_rust:
        if args.fast:
            csv_path = "/tmp/rust_logits.csv"
            if not os.path.exists(csv_path):
                print("  ❌ --fast mode requires /tmp/rust_logits.csv from forward_check binary")
                print("     Run: cargo run --bin forward_check")
                sys.exit(1)
            rust_raw = np.loadtxt(csv_path, delimiter=",")
            rust_logits = reshape_rust_logits(rust_raw, py_ref.shape)
            print(f"  Rust logits loaded (--fast): shape={rust_logits.shape}, first={rust_logits[0,0,0]:.6f}")
        else:
            rust_raw = run_rust_forward()
            if rust_raw is None:
                print("\n❌ No Rust logits available")
                return 1
            rust_logits = reshape_rust_logits(rust_raw, py_ref.shape)
        print(f"  Reshaped to: {rust_logits.shape}")

        # Rust vs Python ref
        sim_rust_py = cosine_similarity(rust_logits, py_ref)
        print(f"\n  Cosine(Rust vs Python Executor): {sim_rust_py:.10f}")

        if sim_rust_py > 0.999:
            print("  ✅ Rust matches Python executor!")
        elif sim_rust_py > 0.9:
            print("  ⚠ Rust close but below threshold — minor bug")
        elif sim_rust_py > 0.5:
            print(f"  ❌ Rust significantly differs (cos={sim_rust_py:.4f}) — Issue #45")
        else:
            print(f"  💥 Rust completely wrong (cos={sim_rust_py:.4f})")

        # Per-token diagnosis
        diagnose_per_token(rust_logits, py_ref, "Rust vs Python")
        diagnose_stats(rust_logits, py_ref)

        # Rust vs HF
        sim_rust_hf = cosine_similarity(rust_logits, hf_ref)
        print(f"\n  Cosine(Rust vs HF): {sim_rust_hf:.10f}")

        # Argmax summary
        print(f"\n  Rust last-token argmax: {int(np.argmax(rust_logits[0, -1]))}")
        print(f"  Python last-token argmax: {int(np.argmax(py_ref[0, -1]))}")
        print(f"  HF last-token argmax: {int(np.argmax(hf_ref[0, -1]))}")

        if args.ctypes:
            call_dylib_ctypes()

        if args.layers:
            manifest_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            rust_dir = os.path.join(manifest_dir, "rust")

            if args.layers == "auto":
                dump_dir = tempfile.mkdtemp(prefix="ld_")
                print(f"\n=== Auto-running DUMP_LAYERS -> {dump_dir} ===")
                env = os.environ.copy()
                env["DUMP_LAYERS"] = dump_dir
                result = subprocess.run(
                    ["cargo", "run", "--bin", "forward_check"],
                    cwd=rust_dir,
                    capture_output=True, text=True,
                    timeout=120,
                    env=env,
                )
                for line in result.stdout.splitlines():
                    print(f"  [forward_check] {line}")
                for line in result.stderr.splitlines():
                    print(f"  [forward_check stderr] {line}")
                if result.returncode != 0:
                    print(f"  ⚠ forward_check exited with code {result.returncode}")
                diagnose_layers(dump_dir)
            else:
                diagnose_layers(args.layers)

        per_token_cos = [
            cosine_similarity(rust_logits[0, t:t+1], py_ref[0, t:t+1])
            for t in range(rust_logits.shape[1])
        ]

        # Diagnosis summary
        print("\n" + "=" * 60)
        print("DIAGNOSIS SUMMARY")
        print("=" * 60)
        identify_bug_location(sim_rust_py, sim_py_hf, per_token_cos)

    return 0


def identify_bug_location(cos_rust_py: float, cos_py_hf: float = 1.0, per_token_cos: list[float] | None = None):
    """Based on cosine values, suggest where the bug might be.

    Args:
        cos_rust_py: Cosine similarity between Rust executor and Python executor.
        cos_py_hf: Cosine similarity between Python executor and HuggingFace reference.
        per_token_cos: Per-token cosine similarities (Rust vs Python) for positional diagnosis.
    """
    print(f"  cos(Rust vs Python) = {cos_rust_py:.6f}")
    print(f"  cos(Python vs HF)   = {cos_py_hf:.6f}")

    if cos_py_hf < 0.99:
        print("  → Bug 在 Python 编译流水线中（pipeline 或 lowering）")
        return
    if cos_rust_py > 0.99:
        print("  → ✅ Issue #45 已解决")
        return

    if per_token_cos:
        print(f"  Per-token cos: {[f'{c:.4f}' for c in per_token_cos]}")

        if per_token_cos[0] >= 0.99 and per_token_cos[-1] < 0.8:
            print("  → 模式：token[0] 正确，后续 token 退化")
            print("  → 怀疑：position embedding / KV cache / SSA routing")
            print("  → 排查：使用 DUMP_LAYERS 做 per-layer 对比")
            print("     DUMP_LAYERS=/tmp/ld cargo run --bin forward_check")
        elif all(c < 0.1 for c in per_token_cos):
            print("  → 所有 token 都错 → 权重加载或常量 folding 问题")
        else:
            print("  → 系统性计算错误（layer norm / matmul 精度）")
    else:
        print("  (no per-token data — run with full diagnose for per-token analysis)")
        print("\n  Likely bug locations based on cosine:")
        print("    If cos < 0.1:         Weight loading corrupted (f16→f32 bug, wrong weights)")
        print("    If cos ≈ 0.5:         Systematic error in computation (position embed, layer norm)")
        print("    If cos token[0] good, token[3] bad: Position embedding issue")
        print("    If all tokens equally bad: Weight loading or constant folding issue")


if __name__ == "__main__":
    sys.exit(main())
