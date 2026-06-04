#!/usr/bin/env python3
"""CI consistency check: verify _OP_DEFS (Python) ↔ SfOps.td (C++) alignment.

Scans both sources:
  1. ``compiler/mlir_dialect/_op_defs.py`` — the canonical ``hal_name`` values
  2. ``sf-dialect/include/Sf/SfOps.td`` — the C++ dialect op mnemonics

Reports:
  - Ops present in _OP_DEFS but missing from SfOps.td  (C++ gap)
  - Ops present in SfOps.td but missing from _OP_DEFS  (Python gap, excluding
    internal ops like ``weight``, ``constant``)

Exits with code 1 if any mismatch is found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ops that are intentionally not dialect ops (they expand to other ops
# during lowering or are purely internal).
SPECIAL_OPS: set[str] = {
    "getitem",
    "split",
    "chunk",
    "_skip_wrap",
}

# Ops defined in SfOps.td that are internal (not in _OP_DEFS).
INTERNAL_CXX_OPS: set[str] = {
    "weight",
    "constant",
}


def parse_op_defs_hal_names(path: Path) -> set[str]:
    """Extract all ``hal_name`` values from ``_OP_DEFS``."""
    text = path.read_text()
    names: set[str] = set()
    for m in re.finditer(r'_OpDef\(\s*["\'](\w+)["\']', text):
        names.add(m.group(1))
    return names


def parse_sfops_mnemonics(path: Path) -> set[str]:
    """Extract all mnemonic strings from ``Sf_Op<"mnemonic">`` in SfOps.td."""
    text = path.read_text()
    names: set[str] = set()
    for m in re.finditer(r'Sf_\w+Op\s*:\s*Sf_\w+\s*<\s*"(\w+)"', text):
        names.add(m.group(1))
    return names


def main() -> int:
    op_defs_path = PROJECT_ROOT / "compiler" / "mlir_dialect" / "_op_defs.py"
    sfops_path = PROJECT_ROOT / "sf-dialect" / "include" / "Sf" / "SfOps.td"

    for p, label in [(op_defs_path, "_op_defs.py"), (sfops_path, "SfOps.td")]:
        if not p.exists():
            print(f"❌ {label} not found at {p}")
            return 1

    py_ops = parse_op_defs_hal_names(op_defs_path)
    cxx_ops = parse_sfops_mnemonics(sfops_path)

    py_ops_filtered = py_ops - SPECIAL_OPS
    cxx_ops_filtered = cxx_ops - INTERNAL_CXX_OPS

    missing_from_cxx = py_ops_filtered - cxx_ops_filtered
    missing_from_py = cxx_ops_filtered - py_ops_filtered

    exit_code = 0

    if missing_from_cxx:
        print(f"❌ {len(missing_from_cxx)} ops in _OP_DEFS but missing from SfOps.td:")
        for name in sorted(missing_from_cxx):
            print(f"   - {name}")
        exit_code = 1
    else:
        print("✅ All _OP_DEFS ops have corresponding SfOps.td definitions.")

    if missing_from_py:
        print(f"\n⚠️  {len(missing_from_py)} ops in SfOps.td but missing from _OP_DEFS:")
        for name in sorted(missing_from_py):
            print(f"   - {name}")
        exit_code = 1
    else:
        print("✅ All SfOps.td ops (excluding internal) have _OP_DEFS entries.")

    if exit_code == 0:
        print("\n✅ Op definitions are fully consistent!")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
