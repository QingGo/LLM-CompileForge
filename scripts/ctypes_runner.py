#!/usr/bin/env python3
"""Run ctypes forward pass on a compiled .dylib and compare against Python executor.

Usage:
    python scripts/ctypes_runner.py compiled/opt_125m_fresh/libopt_125m.dylib
    python scripts/ctypes_runner.py compiled/opt_125m_fresh/libopt_125m.dylib --threshold 0.99
    python scripts/ctypes_runner.py compiled/opt_125m_fresh/libopt_125m.dylib --dump /tmp/layer_dump
"""

import argparse
import sys

sys.path.insert(0, ".")

from scripts.ctypes_oracle import CtypesOracle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare compiled dylib against Python executor"
    )
    parser.add_argument("dylib_path", help="Path to compiled .dylib")
    parser.add_argument(
        "--artifact-dir",
        default="./compiled/opt_125m_fresh",
        help="Artifact directory with model weights",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Minimum cosine threshold (exit 1 if below)",
    )
    parser.add_argument(
        "--dump",
        help="DUMP_LAYERS dir for intermediate tensors",
    )
    args = parser.parse_args()

    oracle = CtypesOracle(artifact_dir=args.artifact_dir)
    cos = oracle.compare(args.dylib_path)

    print(f"cos(ctypes, Python executor): {cos:.10f}")

    if args.threshold and cos < args.threshold:
        print(f"\u274c FAIL: cos={cos:.6f} < threshold={args.threshold}")
        sys.exit(1)
    else:
        print(f"\u2705 PASS: cos={cos:.6f} >= threshold={args.threshold}")
        sys.exit(0)


if __name__ == "__main__":
    main()
