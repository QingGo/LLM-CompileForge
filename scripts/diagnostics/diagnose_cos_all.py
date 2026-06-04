#!/usr/bin/env python3
"""Per-function and per-position cosine similarity for ALL functions in a compiled model.

Usage:
    python scripts/diagnose_cos_all.py
    python scripts/diagnose_cos_all.py outputs/compiled/my_model
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts._cos import cosine_similarity  # noqa: E402
from scripts.ctypes_forward import run_ctypes, run_python_executor  # noqa: E402


def _safe_cos(dylib_arr: np.ndarray, py_arr: np.ndarray) -> float:
    if dylib_arr.shape != py_arr.shape:
        return -1.0
    return cosine_similarity(dylib_arr, py_arr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-function and per-position cosine similarity for ALL functions"
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="outputs/compiled/opt_125m_fresh",
        help="Compiled model directory (default: outputs/compiled/opt_125m_fresh)",
    )
    args = parser.parse_args()

    try:
        dylib = run_ctypes(args.model_dir)
        py_ref = run_python_executor(args.model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the model is compiled first (see compiler/compile.py)")
        sys.exit(1)
    except Exception as e:
        print(f"Error running forward pass: {e}")
        sys.exit(1)

    num_funcs = len(dylib)
    print(f"Total functions: {num_funcs}")
    print()

    # ── Per-function cos table ──────────────────────────────────
    func_cos: list[float] = []
    func_shapes: list[str] = []

    for fi in range(num_funcs):
        try:
            dylib_out = dylib[fi]
            py_out = py_ref[fi]
        except (IndexError, TypeError):
            print(f"  func[{fi:2d}]  SKIPPED  (no output)")
            func_cos.append(-1.0)
            func_shapes.append("N/A")
            continue

        cos = _safe_cos(dylib_out, py_out)
        shape_str = str(list(dylib_out.shape)) if hasattr(dylib_out, "shape") else "?"
        func_cos.append(cos)
        func_shapes.append(shape_str)

        if cos < 0:
            print(f"  func[{fi:2d}]  SHAPE MISMATCH  dylib={list(dylib_out.shape)} py={list(py_out.shape)}")
        else:
            print(f"  func[{fi:2d}]  cos={cos:.6f}  shape={shape_str}")

    # ── Top-3 worst functions (skip func 0 = embedding) ─────────
    print()
    scored = [
        (cos, fi, shape)
        for fi, (cos, shape) in enumerate(zip(func_cos, func_shapes, strict=True))
        if fi > 0 and cos >= 0
    ]
    scored.sort(key=lambda x: x[0])  # ascending = worst first
    worst = scored[:3]

    print("Top-3 worst functions (excluding func 0):")
    for cos, fi, shape in worst:
        print(f"  func[{fi:2d}]  cos={cos:.6f}  shape={shape}")
    print()

    # ── Per-position cos for the last function ──────────────────
    last_fi = num_funcs - 1
    try:
        last_dylib = dylib[last_fi]
        last_py = py_ref[last_fi]
    except (IndexError, TypeError):
        print(f"func[{last_fi}] has no output — skipping per-position analysis")
        sys.exit(0)

    if last_dylib.shape != last_py.shape:
        print(f"func[{last_fi}] shape mismatch: dylib={list(last_dylib.shape)} py={list(last_py.shape)}")
        sys.exit(0)

    # last function output shape: (batch, seq, hidden)
    print(f"Per-position cos for func[{last_fi}] (shape={list(last_dylib.shape)}):")
    for batch in range(last_dylib.shape[0]):
        for pos in range(last_dylib.shape[1]):
            d_vec = last_dylib[batch, pos]
            p_vec = last_py[batch, pos]
            cos = cosine_similarity(d_vec, p_vec)
            print(f"  batch={batch} pos={pos:2d}  cos={cos:.6f}")

    # ── Total cos for the last function (flattened) ─────────────
    total_cos = cosine_similarity(
        last_dylib.ravel(), last_py.ravel()
    )
    print(f"\nTotal cos for func[{last_fi}] (flattened): {total_cos:.6f}")


if __name__ == "__main__":
    main()
