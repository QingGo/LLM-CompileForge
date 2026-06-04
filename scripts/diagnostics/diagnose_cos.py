#!/usr/bin/env python3
"""Per-dimension cosine similarity diagnostic tool.

Compares compiled model output (via ctypes dylib) against Python executor
reference and produces per-dimension precision analysis.

Usage:
    python scripts/diagnose_cos.py --layer 12 --position 1
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


def _validate_outputs(dylib_arr: np.ndarray, py_arr: np.ndarray, layer: int, position: int) -> None:
    if dylib_arr.shape != py_arr.shape:
        print(f"Error: Shape mismatch at layer {layer}: "
              f"dylib={list(dylib_arr.shape)}, py={list(py_arr.shape)}")
        sys.exit(1)
    if position >= dylib_arr.shape[1]:
        print(f"Error: position {position} out of range "
              f"(seq_len={dylib_arr.shape[1]})")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-dimension cosine similarity diagnostic"
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/compiled/opt_125m_fresh",
        help="Compiled model directory (default: outputs/compiled/opt_125m_fresh)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Layer index to diagnose (default: 12)",
    )
    parser.add_argument(
        "--position",
        type=int,
        default=1,
        help="Position index within sequence (default: 1)",
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

    if args.layer >= len(dylib) or args.layer >= len(py_ref):
        print(f"Error: layer {args.layer} out of range "
              f"(dylib has {len(dylib)} functions, "
              f"py executor has {len(py_ref)} functions)")
        sys.exit(1)

    try:
        dylib_out = dylib[args.layer]
        py_out = py_ref[args.layer]
    except (IndexError, TypeError) as e:
        print(f"Error: layer {args.layer} has no output: {e}")
        print("The function may have been skipped (symbol not found in dylib)")
        sys.exit(1)

    _validate_outputs(dylib_out, py_out, args.layer, args.position)

    dylib_vec = dylib_out[0, args.position, :].astype(np.float64)
    py_vec = py_out[0, args.position, :].astype(np.float64)

    cos = cosine_similarity(py_vec, dylib_vec)
    diff = py_vec - dylib_vec
    abs_diff = np.abs(diff)
    max_dim = int(np.argmax(abs_diff))
    keep = np.ones(len(py_vec), dtype=bool)
    keep[max_dim] = False
    cos_no_outlier = cosine_similarity(py_vec[keep], dylib_vec[keep])

    print(f"layer {args.layer}, position {args.position}: cos={cos:.3f}")
    print(f"Max diff dimension: dim[{max_dim}] diff={diff[max_dim]:+.4f} "
          f"(py={py_vec[max_dim]:.1f} dylib={dylib_vec[max_dim]:.1f})")
    print(f"Cos excluding dim {max_dim}: {cos_no_outlier:.3f}")

    if args.layer > 0:
        try:
            prev_dylib_out = dylib[args.layer - 1]
            prev_py_out = py_ref[args.layer - 1]
            _validate_outputs(prev_dylib_out, prev_py_out, args.layer - 1, args.position)
            prev_diff = (prev_py_out[0, args.position, :].astype(np.float64)
                         - prev_dylib_out[0, args.position, :].astype(np.float64))
            dir_cos = cosine_similarity(diff, prev_diff)
            print(f"Difference direction consistency vs layer {args.layer - 1}: "
                  f"cos={dir_cos:.3f}")
        except Exception:
            pass

    top5 = np.argsort(abs_diff)[-5:][::-1]
    print("\nTop-5 worst dimensions:")
    for d in top5:
        print(f"  dim[{d}]: diff={diff[d]:+.4f} "
              f"(py={py_vec[d]:.1f} dylib={dylib_vec[d]:.1f})")

    print("\nStats:")
    print(f"  Mean abs diff: {abs_diff.mean():.6f}")
    print(f"  Max abs diff: {abs_diff.max():.6f}")
    print(f"  Diff std: {diff.std():.6f}")


if __name__ == "__main__":
    main()
