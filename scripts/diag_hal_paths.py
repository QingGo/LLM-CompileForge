#!/usr/bin/env python3
"""Diagnose HAL IR accuracy gap between Path A (MlirExecutor) and Path B (forward_check_hal).

Runs both paths with identical inputs, captures per-function intermediate outputs
and final logits, then computes per-function cosine similarity and per-token
statistics to identify where the accuracy gap originates.

Usage:
    source .venv/bin/activate
    unset CONDA_PREFIX
    export KMP_DUPLICATE_LIB_OK=TRUE
    export DYLD_LIBRARY_PATH="..."
    python scripts/diag_hal_paths.py [--artifact compiled/opt_125m_fresh] [--tokens 2,32826,85,4129]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ── Binary format constants ─────────────────────────────────────────
# Path B .bin format: rank(i32) + dims(i32)*rank + data(f32 bytes)


def read_path_b_bin(path: str) -> np.ndarray:
    """Read a Path B per-function dump binary file.

    Format: rank (i32 LE) + dims (i32 LE each) + data (f32 LE).
    """
    with open(path, "rb") as f:
        rank_bytes = f.read(4)
        if len(rank_bytes) < 4:
            raise ValueError(f"Truncated file: {path}")
        rank = struct.unpack("<i", rank_bytes)[0]
        dims: list[int] = []
        for _ in range(rank):
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                raise ValueError(f"Truncated dims in: {path}")
            dims.append(struct.unpack("<i", dim_bytes)[0])
        data = f.read()
    expected = int(np.prod(dims)) * 4
    if len(data) != expected:
        print(f"  WARNING: {path}: shape={dims}, expected={expected}B, got={len(data)}B")
    arr = np.frombuffer(data, dtype=np.float32)
    if dims:
        arr = arr.reshape(dims)
    return arr


def read_path_b_csv(csv_path: str) -> np.ndarray:
    """Read Path B logits CSV into a flat f32 array."""
    vals: list[float] = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(float(line))
    return np.array(vals, dtype=np.float32)


def read_path_a_npy(npy_path: str) -> np.ndarray:
    """Read a Path A per-function .npy dump."""
    return np.load(npy_path)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.shape != b.shape:
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    denom = na * nb
    if denom < 1e-30:
        return 0.0
    return dot / denom


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    """Fraction of top-k indices that match between a and b."""
    a_topk = set(np.argsort(-a.ravel())[:k])
    b_topk = set(np.argsort(-b.ravel())[:k])
    return len(a_topk & b_topk) / k


def run_path_a(
    artifact_dir: str, tokens: list[int], dump_dir: str
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Run Path A (MlirExecutor) and capture per-function outputs + final logits.

    Returns:
        (func_outputs: dict[func_idx -> np.ndarray], logits: np.ndarray)
    """
    from compiler.serialize import load_artifact
    from engine.mlir_executor import MlirExecutor
    from hal.pytorch_backend import PyTorchBackend

    artifact = load_artifact(artifact_dir)
    backend = PyTorchBackend("cpu")
    executor = MlirExecutor(artifact, backend, dump_dir=dump_dir)

    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits_tensor = executor.forward(input_ids)
    logits = logits_tensor.detach().cpu().numpy().astype(np.float32)

    # Collect per-function wire outputs from dump_dir.
    # Use heuristic: for each function, find the .npy with ndim >= 3
    # (the dynamic 3D wire tensor), skipping 2D weight files and 1D constants.
    func_outputs: dict[int, np.ndarray] = {}
    import glob as _glob
    for fi in range(len(artifact.functions)):
        pattern = os.path.join(dump_dir, f"py_func_{fi}_out*_*.npy")
        all_matches = sorted(_glob.glob(pattern))
        if not all_matches:
            fpath = os.path.join(dump_dir, f"py_func_{fi}_0.npy")
            if os.path.exists(fpath):
                all_matches = [fpath]
        wire_file = None
        for fpath in all_matches:
            try:
                arr = np.load(fpath)
                if arr.ndim >= 3:
                    wire_file = fpath
                    break
            except Exception:
                continue
        if wire_file is None and all_matches:
            wire_file = all_matches[0]
        if wire_file:
            func_outputs[fi] = np.load(wire_file)
    return func_outputs, logits


def run_path_b(
    artifact_dir: str, tokens: list[int], binary_path: str
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Run Path B (forward_check_hal) and collect per-function + logits.

    Returns:
        (func_outputs: dict[func_idx -> np.ndarray], logits: np.ndarray)
    """
    import glob as _glob

    # Clean up old dump files
    for f in _glob.glob("/tmp/hal_func_*.bin"):
        os.remove(f)
    for f in _glob.glob("/tmp/rust_hal_logits.csv"):
        os.remove(f)

    env = os.environ.copy()
    tokens_str = ",".join(str(t) for t in tokens)
    env["FORWARD_CHECK_TOKENS"] = tokens_str

    result = subprocess.run(
        [binary_path],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"[ERROR] Path B exited with code {result.returncode}")
        print("STDOUT:", result.stdout[-2000:])
        print("STDERR:", result.stderr[-2000:])
        return {}, np.array([], dtype=np.float32)

    # Collect per-function .bin files
    func_outputs: dict[int, np.ndarray] = {}
    for f in sorted(_glob.glob("/tmp/hal_func_*.bin")):
        # Parse func index from filename: /tmp/hal_func_{fi}.bin
        base = os.path.basename(f)
        fi_str = base.replace("hal_func_", "").replace(".bin", "")
        try:
            fi = int(fi_str)
            func_outputs[fi] = read_path_b_bin(f)
        except (ValueError, struct.error) as e:
            print(f"  WARNING: failed to read {f}: {e}")

    # Collect logits CSV
    csv_path = "/tmp/rust_hal_logits.csv"
    logits = np.array([], dtype=np.float32)
    if os.path.exists(csv_path):
        logits = read_path_b_csv(csv_path)
    return func_outputs, logits


def compute_per_token_stats(
    logits_a: np.ndarray, logits_b: np.ndarray, vocab_size: int = 50272
) -> list[dict[str, Any]]:
    """Compute per-token cos_sim, argmax diff, top-5 overlap for each position.

    Assumes logits shape [batch=1, seq, vocab].
    """
    if logits_a.ndim != logits_b.ndim:
        return []
    seq_len = logits_a.shape[1] if logits_a.ndim >= 2 else 0
    if seq_len == 0:
        return []

    results: list[dict[str, Any]] = []
    for pos in range(seq_len):
        a_vec = logits_a[0, pos, :].astype(np.float64)
        b_vec = logits_b[0, pos, :].astype(np.float64)
        cos = cos_sim(a_vec, b_vec)
        arg_a = int(np.argmax(a_vec))
        arg_b = int(np.argmax(b_vec))
        top5 = top_k_overlap(a_vec, b_vec, 5)
        results.append({
            "token_pos": pos,
            "cos_sim": round(float(cos), 6),
            "argmax_a": arg_a,
            "argmax_b": arg_b,
            "argmax_match": arg_a == arg_b,
            "top5_overlap": round(float(top5), 4),
            "mean_a": round(float(np.mean(a_vec)), 4),
            "mean_b": round(float(np.mean(b_vec)), 4),
            "std_a": round(float(np.std(a_vec)), 4),
            "std_b": round(float(np.std(b_vec)), 4),
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose HAL IR Path A vs Path B accuracy gap"
    )
    parser.add_argument(
        "--artifact", default="compiled/opt_125m_fresh",
        help="Path to compiled model artifact"
    )
    parser.add_argument(
        "--tokens", default="2,32826,85,4129",
        help="Comma-separated input token IDs"
    )
    parser.add_argument(
        "--binary", default="rust/target/debug/forward_check_hal",
        help="Path to forward_check_hal binary"
    )
    parser.add_argument(
        "--skip-path-a", action="store_true",
        help="Skip Path A execution (use existing dumps)"
    )
    parser.add_argument(
        "--skip-path-b", action="store_true",
        help="Skip Path B execution (use existing dumps)"
    )
    parser.add_argument(
        "--json-output", default=None,
        help="Save detailed JSON report to path"
    )
    args = parser.parse_args()

    tokens = [int(t.strip()) for t in args.tokens.split(",")]
    artifact_dir = args.artifact
    binary_path = args.binary

    print("=" * 72)
    print("DIAG: HAL IR Path A vs Path B Accuracy Diagnostic")
    print(f"  Artifact: {artifact_dir}")
    print(f"  Tokens: {tokens}")
    print("=" * 72)

    # ── Run Path A ───────────────────────────────────────────────────
    path_a_dump_dir: str | None = None
    pa_funcs: dict[int, np.ndarray] = {}
    pa_logits: np.ndarray = np.array([], dtype=np.float32)

    if not args.skip_path_a:
        td = tempfile.mkdtemp(prefix="diag_path_a_")
        path_a_dump_dir = td
        print(f"\n[Path A] Running MlirExecutor (dump_dir={td})...")
        try:
            pa_funcs, pa_logits = run_path_a(artifact_dir, tokens, td)
        except Exception as e:
            print(f"[Path A] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1
        print(f"[Path A] Captured {len(pa_funcs)} function outputs")
        print(f"[Path A] Final logits shape: {pa_logits.shape}")
        print(f"[Path A] Logits stats: mean={np.mean(pa_logits):.4f}, "
              f"std={np.std(pa_logits):.4f}, "
              f"range=[{np.min(pa_logits):.4f}, {np.max(pa_logits):.4f}]")
    else:
        print("\n[Path A] SKIPPED (--skip-path-a)")

    # ── Run Path B ───────────────────────────────────────────────────
    pb_funcs: dict[int, np.ndarray] = {}
    pb_logits: np.ndarray = np.array([], dtype=np.float32)

    if not args.skip_path_b:
        print(f"\n[Path B] Running forward_check_hal ({binary_path})...")
        try:
            pb_funcs, pb_logits = run_path_b(artifact_dir, tokens, binary_path)
        except Exception as e:
            print(f"[Path B] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return 1
        print(f"[Path B] Captured {len(pb_funcs)} function outputs")
        print(f"[Path B] Final logits len: {len(pb_logits)}")
        if len(pb_logits) > 0:
            print(f"[Path B] Logits stats: mean={np.mean(pb_logits):.4f}, "
                  f"std={np.std(pb_logits):.4f}, "
                  f"range=[{np.min(pb_logits):.4f}, {np.max(pb_logits):.4f}]")
    else:
        print("\n[Path B] SKIPPED (--skip-path-b)")

    # ── Per-Function Cosine Similarity ───────────────────────────────
    print("\n" + "=" * 72)
    print("PER-FUNCTION COSINE SIMILARITY (Path A vs Path B)")
    print("=" * 72)

    common_funcs = sorted(set(pa_funcs.keys()) & set(pb_funcs.keys()))
    if common_funcs:
        func_cos_results: list[dict[str, Any]] = []
        for fi in common_funcs:
            a_arr = pa_funcs[fi]
            b_arr = pb_funcs[fi]
            a_arr_flat = a_arr.ravel().astype(np.float32)
            b_arr_flat = b_arr.ravel().astype(np.float32)

            # Truncate to same length if shapes differ
            min_len = min(len(a_arr_flat), len(b_arr_flat))
            a_cmp = a_arr_flat[:min_len]
            b_cmp = b_arr_flat[:min_len]

            cos = cos_sim(a_cmp, b_cmp)
            max_diff = float(np.max(np.abs(a_cmp - b_cmp)))
            mean_diff = float(np.mean(np.abs(a_cmp - b_cmp)))

            status = "✅" if cos > 0.999 else ("⚠️" if cos > 0.95 else "❌")
            func_cos_results.append({
                "func_idx": fi,
                "cos_sim": round(float(cos), 6),
                "max_abs_diff": round(max_diff, 6),
                "mean_abs_diff": round(mean_diff, 6),
                "shape_a": list(a_arr.shape),
                "shape_b": list(b_arr.shape),
            })
            print(f"  func[{fi:2d}] {status} cos={cos:.6f}  "
                  f"shape_a={list(a_arr.shape)} shape_b={list(b_arr.shape)}  "
                  f"max_diff={max_diff:.4f} mean_diff={mean_diff:.4f}")
    else:
        print("  No common function outputs to compare!")
        func_cos_results = []

    # ── Per-Token Statistics ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PER-TOKEN LOGITS STATISTICS (Path A vs Path B)")
    print("=" * 72)

    token_stats: list[dict[str, Any]] = []
    overall_cos = 0.0
    if len(pa_logits) > 0 and len(pb_logits) > 0:
        # Reshape logits to [1, seq, vocab] if flat
        vocab_size = 50272
        seq_len = len(tokens)
        expected_len = seq_len * vocab_size

        pa_reshaped = pa_logits
        pb_reshaped = pb_logits

        if pa_logits.ndim == 1 and len(pa_logits) == expected_len:
            pa_reshaped = pa_logits.reshape(1, seq_len, vocab_size)
        elif pa_logits.ndim == 1:
            n = min(len(pa_logits), len(pb_logits))
            pa_reshaped = pa_logits[:n].reshape(1, -1, vocab_size) if n % vocab_size == 0 else pa_logits[:n]
            if pb_logits.ndim == 1:
                pb_reshaped = pb_logits[:n]

        if pb_logits.ndim == 1 and len(pb_logits) == expected_len:
            pb_reshaped = pb_logits.reshape(1, seq_len, vocab_size)

        # Overall cosine
        pa_flat = pa_reshaped.ravel().astype(np.float64)
        pb_flat: np.ndarray = pb_reshaped.ravel().astype(np.float64)  # type: ignore[no-untyped-call]
        min_overall = min(len(pa_flat), len(pb_flat))
        overall_cos = cos_sim(pa_flat[:min_overall], pb_flat[:min_overall])

        print(f"  Overall cosine similarity: {overall_cos:.6f}")
        print(f"  Path A shape: {pa_reshaped.shape}, Path B shape: {pb_reshaped.shape}")
        print()

        # Per-token analysis
        if pa_reshaped.ndim >= 2 and pb_reshaped.ndim >= 2:
            actual_seq = min(pa_reshaped.shape[1], pb_reshaped.shape[1])
            token_stats = compute_per_token_stats(pa_reshaped, pb_reshaped)
        elif pa_reshaped.ndim == 1:
            # Flat logits — compute single set of stats
            token_stats = [{
                "token_pos": -1,
                "cos_sim": round(float(overall_cos), 6),
                "argmax_a": int(np.argmax(pa_flat)),
                "argmax_b": int(np.argmax(pb_flat)),
                "argmax_match": int(np.argmax(pa_flat)) == int(np.argmax(pb_flat)),
                "top5_overlap": round(float(top_k_overlap(pa_flat, pb_flat, 5)), 4),
                "mean_a": round(float(np.mean(pa_flat)), 4),
                "mean_b": round(float(np.mean(pb_flat)), 4),
                "std_a": round(float(np.std(pa_flat)), 4),
                "std_b": round(float(np.std(pb_flat)), 4),
            }]

        for ts in token_stats:
            pos = ts["token_pos"]
            match_str = "MATCH" if ts["argmax_match"] else "DIFF"
            print(
                f"  Token [{pos}] cos={ts['cos_sim']:.6f}  "
                f"argmax: {ts['argmax_a']} vs {ts['argmax_b']} ({match_str})  "
                f"top5={ts['top5_overlap']:.2f}  "
                f"(mean: {ts['mean_a']:.2f} vs {ts['mean_b']:.2f}, "
                f"std: {ts['std_a']:.2f} vs {ts['std_b']:.2f})"
            )
    else:
        print("  No logits to compare (one or both paths failed)")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    num_bad_funcs = sum(1 for r in func_cos_results if r["cos_sim"] < 0.95)
    num_arg_diff = sum(1 for ts in token_stats if not ts["argmax_match"])

    print(f"  Functions compared: {len(func_cos_results)}")
    print(f"  Functions with cos < 0.95: {num_bad_funcs}")
    print(f"  Overall logits cos: {overall_cos:.6f}")
    print(f"  Token argmax diffs: {num_arg_diff}/{len(token_stats)}")

    worst_func = None
    if func_cos_results:
        sorted_funcs = sorted(func_cos_results, key=lambda r: r["cos_sim"])
        worst_func = sorted_funcs[0]
        print(f"  Worst function: func[{worst_func['func_idx']}] cos={worst_func['cos_sim']:.6f}")
        for r in sorted_funcs[:5]:
            print(f"    func[{r['func_idx']:2d}] cos={r['cos_sim']:.6f} max_diff={r['max_abs_diff']:.4f}")

    # ── Save JSON report ─────────────────────────────────────────────
    report: dict[str, Any] = {
        "artifact": artifact_dir,
        "tokens": tokens,
        "overall_cos": round(float(overall_cos), 6),
        "path_a_logits_shape": list(pa_logits.shape) if len(pa_logits) > 0 else None,
        "path_b_logits_len": len(pb_logits) if len(pb_logits) > 0 else 0,
        "per_function": func_cos_results,
        "per_token": token_stats,
    }

    json_out = args.json_output or "/tmp/diag_hal_paths.json"
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {json_out}")

    return 0


if __name__ == "__main__":
    sys.path.insert(0, PROJECT_ROOT)
    sys.exit(main())
