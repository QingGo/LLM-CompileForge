#!/usr/bin/env python3
"""Layer-by-layer cosine comparison: Rust dumps vs Python reference.

Usage:
    python scripts/compare_layers.py <rust_dump_dir> <python_layer_dump_dir>
    python scripts/compare_layers.py <rust_dump_dir> <python_layer_dump_dir> --threshold 0.9
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flattened arrays."""
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return float(np.dot(a_f, b_f) / (denom + 1e-12))


def load_rust_dump(dump_dir: str) -> dict[int, np.ndarray]:
    """Load Rust dump files (.bin + .json pairs) from directory.

    Expected naming: func_<idx>[_<out_idx>].json + func_<idx>[_<out_idx>].bin
    """
    dump_dir = Path(dump_dir)
    tensors: dict[int, np.ndarray] = {}

    if not dump_dir.exists():
        print(f"  [WARN] Directory not found: {dump_dir}", file=sys.stderr)
        return tensors

    for json_path in sorted(dump_dir.glob("func_*.json")):
        parts = json_path.stem.split("_")
        func_idx = int(parts[1])
        out_idx = int(parts[2]) if len(parts) > 2 else 0

        with open(json_path) as f:
            meta = json.load(f)

        shape = meta["shape"]
        bin_path = json_path.with_suffix(".bin")

        if not bin_path.exists():
            print(f"  [WARN] Missing bin file: {bin_path}", file=sys.stderr)
            continue

        tensor = np.fromfile(bin_path, dtype=np.float32).reshape(shape)
        tensors[func_idx] = tensor  # last output wins for multi-output funcs

    return tensors


def load_python_dumps(dump_dir: str) -> dict[str, np.ndarray]:
    """Load Python .npy dump files from directory."""
    dump_dir = Path(dump_dir)
    tensors: dict[str, np.ndarray] = {}

    if not dump_dir.exists():
        print(f"  [WARN] Directory not found: {dump_dir}", file=sys.stderr)
        return tensors

    for npy_path in sorted(dump_dir.glob("*.npy")):
        name = npy_path.stem
        tensors[name] = np.load(str(npy_path))

    return tensors


def unify_shape(tensor: np.ndarray, target_ndim: int = 3) -> np.ndarray:
    """Unify tensor to target number of dimensions by unsqueezing/squeezing leading dims."""
    while tensor.ndim < target_ndim:
        tensor = np.expand_dims(tensor, axis=0)
    while tensor.ndim > target_ndim:
        tensor = np.squeeze(tensor, axis=0)
    return tensor


def main() -> int:
    threshold = 0.9
    args = sys.argv[1:]

    if not args or "-h" in args or "--help" in args:
        print(__doc__)
        return 0

    filtered: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--threshold" and i + 1 < len(args):
            threshold = float(args[i + 1])
            i += 2
        else:
            filtered.append(args[i])
            i += 1
    args = filtered

    if len(args) < 2:
        print(
            "Usage: python scripts/compare_layers.py <rust_dump_dir> <python_layer_dump_dir> [--threshold N]",
            file=sys.stderr,
        )
        return 1

    rust_dir, py_dir = args[0], args[1]

    print(f"Loading Rust dumps from: {rust_dir}")
    rust_tensors = load_rust_dump(rust_dir)
    print(f"  Loaded {len(rust_tensors)} Rust function dumps")

    print(f"Loading Python dumps from: {py_dir}")
    py_tensors = load_python_dumps(py_dir)
    print(f"  Loaded {len(py_tensors)} Python layer dumps")

    num_layers = max(len(rust_tensors) - 2, 0)
    layer_map: list[tuple[str, str]] = [
        ("embed_prefix", "layer_0_input"),
    ]
    for i in range(num_layers):
        layer_map.append((f"layer_{i}", f"layer_{i}_output"))
    layer_map.append(("output", "hf_logits"))

    print()
    header = f"{'Function':<20} {'Layer':<20} {'Cosine':<15} {'Verdict':<10}"
    print(header)
    print("-" * 65)

    first_divergent: tuple[int, str, float] | None = None
    results: list[tuple[str, str, float, str]] = []

    for rust_name, py_name in layer_map:
        if rust_name == "embed_prefix":
            rust_idx = 0
        elif rust_name == "output":
            rust_idx = len(rust_tensors) - 1
        else:
            rust_idx = int(rust_name.split("_")[1]) + 1

        rust_t = rust_tensors.get(rust_idx)
        py_t = py_tensors.get(py_name)

        if rust_t is None:
            print(f"{rust_name:<20} {py_name:<20} {'NO RUST DUMP':<15} {'SKIP':<10}")
            continue
        if py_t is None:
            print(f"{rust_name:<20} {py_name:<20} {'NO PYTHON DUMP':<15} {'SKIP':<10}")
            continue

        r = unify_shape(rust_t, 3)
        p = unify_shape(py_t, 3)

        if r.shape != p.shape:
            if r.ndim == 3 and p.ndim == 3 and r.shape[1:] == p.shape[1:]:
                min_batch = min(r.shape[0], p.shape[0])
                r = r[:min_batch]
                p = p[:min_batch]
            else:
                print(f"{rust_name:<20} {py_name:<20} {'SHAPE MISMATCH':<15} {'SKIP':<10}")
                print(f"        rust={r.shape} py={p.shape}", file=sys.stderr)
                continue

        cos = cosine_similarity(r, p)
        verdict = "PASS" if cos >= threshold else "FAIL"

        if cos < threshold and first_divergent is None:
            first_divergent = (rust_idx, rust_name, cos)

        print(f"{rust_name:<20} {py_name:<20} {cos:<15.8f} {verdict:<10}")
        results.append((rust_name, py_name, cos, verdict))

    print("-" * 65)

    print()
    if first_divergent is not None:
        print(
            f"FIRST DIVERGING: func_{first_divergent[0]} ({first_divergent[1]}) "
            f"— cos={first_divergent[2]:.6f}"
        )
        print(f"THRESHOLD: cos < {threshold}")
    else:
        print("ALL LAYERS PASS — no divergence detected")

    print(f"\nRust functions loaded: {len(rust_tensors)}")
    for idx in sorted(rust_tensors):
        print(f"  func_{idx}: shape={rust_tensors[idx].shape}")
    print(f"Python layers loaded: {len(py_tensors)}")
    for name in sorted(py_tensors):
        print(f"  {name}: shape={py_tensors[name].shape}")

    return 0 if first_divergent is None else 1


if __name__ == "__main__":
    sys.exit(main())
