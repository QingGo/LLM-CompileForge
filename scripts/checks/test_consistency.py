#!/usr/bin/env python3
"""Four-way cosine similarity test: HF ↔ Python Executor ↔ ctypes dylib ↔ Rust forward_check.

All four paths run forward pass on the same input (prompt "The capital of France is"),
and pairwise cosine similarities are computed on the flattened logits.
Every pair must have COS ≥ 0.999.

Usage:
    # Via Makefile (sets DYLD_LIBRARY_PATH and PYTHONPATH automatically):
    make test-consistency

    # Direct (requires env setup):
    DYLD_LIBRARY_PATH="..." PYTHONPATH="..." python scripts/test_consistency.py

Exit code: 0 if all COS ≥ 0.999, 1 otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

ARTIFACT_DIR = "compiled/opt_125m_fresh"
DYLIB_PATH = os.path.join(ARTIFACT_DIR, "libopt_125m.dylib")
FORWARD_CHECK_BIN = "rust/target/release/forward_check"
RUST_LOGITS_CSV = "/tmp/rust_logits.csv"
HF_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--facebook--opt-125m"
)
EXPECTED_INPUT_IDS = [2, 133, 812, 9, 1470, 16]


# =====================================================================
# Helpers
# =====================================================================


def ensure(path: str, make_cmd: str | None, desc: str) -> None:
    """Check that *path* exists, otherwise print error + exit."""
    if not os.path.exists(path):
        print(f"❌ {desc} not found: {path}")
        if make_cmd:
            print(f"   请运行: {make_cmd}")
        sys.exit(1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays (float64, flattened)."""
    a64 = a.astype(np.float64).flatten()
    b64 = b.astype(np.float64).flatten()
    # If shapes differ (e.g. batch=1 vs batch=2), truncate to smaller
    if len(a64) != len(b64):
        min_len = min(len(a64), len(b64))
        print(f"  ⚠️ shape mismatch: {a.shape} vs {b.shape}, truncating to {min_len} elements")
        a64 = a64[:min_len]
        b64 = b64[:min_len]
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    if denom == 0.0:
        return 1.0 if np.all(a64 == b64) else 0.0
    return float(a64 @ b64 / denom)


# =====================================================================
# Forward runners
# =====================================================================


def run_hf_forward(input_ids_np: np.ndarray) -> np.ndarray:
    """Run HuggingFace OPT-125M forward (local cache only)."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    with torch.no_grad():
        output = model(input_ids=torch.from_numpy(input_ids_np))
    logits = output.logits.numpy()
    print(f"  ✅ HF forward: shape={logits.shape}")
    return logits


def run_python_executor_forward(input_ids_np: np.ndarray) -> np.ndarray:
    """Run Python MlirExecutor forward."""
    from scripts.ctypes_forward import run_python_executor as _run_py

    result = _run_py(artifact_dir=ARTIFACT_DIR, input_ids=input_ids_np)
    logits = result.logits
    print(f"  ✅ Python executor forward: shape={logits.shape}")
    return logits


def run_ctypes_forward(input_ids_np: np.ndarray) -> np.ndarray:
    """Run ctypes dylib forward."""
    from scripts.ctypes_forward import run_ctypes as _run_ct

    result = _run_ct(artifact_dir=ARTIFACT_DIR, input_ids=input_ids_np)
    logits = result.logits
    print(f"  ✅ ctypes forward: shape={logits.shape}")
    return logits


def run_rust_forward(input_ids: list[int] | None = None) -> np.ndarray:
    """Run Rust forward_check binary, read /tmp/rust_logits.csv."""
    binary = os.path.join(_PROJECT_ROOT, FORWARD_CHECK_BIN)
    if not os.path.exists(binary):
        print(f"❌ forward_check binary not found: {binary}")
        print("   请运行: cd rust && cargo build --release --bin forward_check")
        sys.exit(1)

    # Remove stale CSV before run
    if os.path.exists(RUST_LOGITS_CSV):
        os.remove(RUST_LOGITS_CSV)

    # Pass common input tokens so Rust uses the same input as HF/PY/CTYPES
    rust_env = os.environ.copy()
    if input_ids is not None:
        rust_env["FORWARD_CHECK_TOKENS"] = ",".join(str(t) for t in input_ids)

    result = subprocess.run(
        [binary],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=rust_env,
        timeout=120,
    )
    # Print stdout (forward_check logs) for transparency
    for line in result.stdout.strip().split("\n"):
        print(f"    {line}")

    if result.returncode != 0:
        print(f"  ❌ forward_check failed (exit code {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                print(f"    {line}")
        sys.exit(1)

    if not os.path.exists(RUST_LOGITS_CSV):
        print(f"❌ forward_check did not produce {RUST_LOGITS_CSV}")
        sys.exit(1)

    raw = np.loadtxt(RUST_LOGITS_CSV, delimiter=",")
    print(f"  ✅ Rust forward: numel={raw.size}")
    return raw


# =====================================================================
# Main
# =====================================================================


def main() -> int:
    print("=" * 64)
    print("  四路一致性测试 (HF / Python Executor / ctypes / Rust)")
    print("=" * 64)

    # ── Step 1: Check compiled artifacts ──────────────────────────
    print("\n[1/5] 检查编译产物...")
    ensure(
        os.path.join(ARTIFACT_DIR, "model.mlir"),
        "make test-dylib-cos",
        "编译产物 model.mlir",
    )
    ensure(
        DYLIB_PATH,
        "make test-dylib-cos",
        "编译产物 .dylib",
    )
    ensure(
        os.path.join(ARTIFACT_DIR, "weights.safetensors"),
        None,
        "权重文件 weights.safetensors",
    )
    print("  ✅ 所有编译产物已就绪")

    # ── Step 2: Check HF cache ────────────────────────────────────
    print("\n[2/5] 检查 HuggingFace 模型缓存...")
    if not os.path.isdir(HF_CACHE):
        print(f"❌ HF 模型缓存未找到: {HF_CACHE}")
        print("   请首次下载:")
        print('   source .venv/bin/activate && python -c \\')
        print('     "from transformers import AutoModelForCausalLM; \\')
        print("      AutoModelForCausalLM.from_pretrained('facebook/opt-125m')\"")
        sys.exit(1)
    print("  ✅ HF 缓存已就绪")

    # ── Step 3: Tokenize input ────────────────────────────────────
    print("\n[3/5] Tokenize 输入...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/opt-125m",
        local_files_only=True,
    )
    input_text = "The capital of France is"
    encoded = tokenizer(input_text, return_tensors="np")
    input_ids_np = encoded["input_ids"]  # shape [1, seq_len]
    actual_ids = input_ids_np[0].tolist()
    print(f"  Input text: '{input_text}'")
    print(f"  Input IDs:  {actual_ids}")

    if actual_ids != EXPECTED_INPUT_IDS:
        print(f"  ⚠️  Tokenized IDs differ from expected {EXPECTED_INPUT_IDS}")
        print("     Using tokenized IDs for all four paths")
    else:
        print("  ✅ Tokenized IDs match expected")

    # ── Step 4: Run all four forwards ─────────────────────────────
    print("\n[4/5] 运行四个 forward 路径...")

    print("\n  --- HF (HuggingFace) ---")
    hf_logits = run_hf_forward(input_ids_np)

    print("\n  --- Python Executor (MlirExecutor) ---")
    py_logits = run_python_executor_forward(input_ids_np)

    print("\n  --- ctypes (dylib) ---")
    ct_logits = run_ctypes_forward(input_ids_np)

    print("\n  --- Rust (forward_check binary) ---")
    rust_logits = run_rust_forward(input_ids=actual_ids)

    # Reshape Rust logits to match reference shape (if needed)
    # forward_check writes flat CSV; reshape to [batch, seq, vocab]
    if rust_logits.ndim == 1:
        try:
            rust_logits = rust_logits.reshape(py_logits.shape)
            print(f"  Reshaped Rust logits to {py_logits.shape}")
        except ValueError:
            print(
                f"  ⚠️  Cannot reshape Rust logits ({rust_logits.size} elements)"
                f" to match Python executor shape {py_logits.shape}"
            )
            print("     Will compare with truncation")

    # ── Step 5: Compute all pairwise cosines ──────────────────────
    print("\n[5/5] 计算 pairwise cosine 相似度...")

    pairs = [
        ("HF", hf_logits, "PY", py_logits),
        ("HF", hf_logits, "CTYPES", ct_logits),
        ("HF", hf_logits, "RUST", rust_logits),
        ("PY", py_logits, "CTYPES", ct_logits),
        ("PY", py_logits, "RUST", rust_logits),
        ("CTYPES", ct_logits, "RUST", rust_logits),
    ]

    results: list[tuple[str, str, float, bool]] = []
    all_pass = True

    print()
    # Header
    print(f"  {'Pair':<28} {'COS':<14} {'Status':<8}")
    print(f"  {'-'*28} {'-'*14} {'-'*8}")
    for name_a, arr_a, name_b, arr_b in pairs:
        try:
            cos_val = cosine(arr_a, arr_b)
        except Exception as e:
            print(f"  {name_a:>8} vs {name_b:<8}  ERROR: {e}")
            results.append((name_a, name_b, 0.0, False))
            all_pass = False
            continue

        passed = cos_val >= 0.999
        status = "✅ PASS" if passed else "❌ FAIL"
        result_str = f"  {name_a:>8} vs {name_b:<8}  {cos_val:.10f}    {status}"
        print(result_str)
        results.append((name_a, name_b, cos_val, passed))
        if not passed:
            all_pass = False

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 64)
    if all_pass:
        print("  ✅ 所有 pairwise cosine ≥ 0.999 — 一致性验证通过")
        print("=" * 64)
        return 0
    else:
        print("  ❌ 以下 pair 的 cosine < 0.999:")
        for name_a, name_b, cos_val, passed in results:
            if not passed:
                print(f"     {name_a:>8} vs {name_b:<8}: cos={cos_val:.10f}")
        print()
        print("  可能的原因:")
        print("    - 编译产物已过期 (运行 make test-dylib-cos 重新生成)")
        print("    - 权重不一致 (运行 make verify-consistency 检查)")
        print("    - Rust forward_check 二进制过期 (运行 cd rust && cargo build --release --bin forward_check)")
        print("    - 词表/输入 ID 不一致 (检查 tokenizer)")
        print("=" * 64)
        return 1


if __name__ == "__main__":
    sys.exit(main())
