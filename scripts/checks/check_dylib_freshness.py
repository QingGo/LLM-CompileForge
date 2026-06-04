#!/usr/bin/env python3
"""Check that dylib is newer than model.mlir (freshness check).

Usage:
    python scripts/check_dylib_freshness.py outputs/compiled/opt_125m_fresh

Exits 0 if dylib is fresh (newer than model.mlir), 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path


def check_freshness(model_dir: str) -> bool:
    model_path = Path(model_dir)
    mlir_path = model_path / "model.mlir"
    dylib_path = model_path / f"lib{model_path.name}.dylib"

    if not mlir_path.exists():
        print(f"WARN: {mlir_path} not found, skipping freshness check")
        return True

    if not dylib_path.exists():
        print(f"FAIL: {dylib_path} not found")
        return False

    mlir_mtime = mlir_path.stat().st_mtime
    dylib_mtime = dylib_path.stat().st_mtime

    if dylib_mtime < mlir_mtime:
        print("FAIL: dylib is older than model.mlir")
        print(f"  model.mlir: {mlir_mtime}")
        print(f"  dylib:      {dylib_mtime}")
        return False

    print("OK: dylib is fresh")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_dylib_freshness.py <model_dir>")
        return 1
    return 0 if check_freshness(sys.argv[1]) else 1


if __name__ == "__main__":
    sys.exit(main())
