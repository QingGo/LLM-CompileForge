#!/usr/bin/env python3
"""Diagnose compute graph SSA wiring — log all fallback to (0,0).

Loads the artifact and checks every function input SSA name against
the producer map. Reports any that fall through to the default (0,0).

Usage:
    python scripts/diagnose_compute_graph_wiring.py compiled/opt_125m_fresh
"""

from __future__ import annotations

import os
import sys


def diagnose(compiled_dir: str) -> int:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from compiler.serialize import load_artifact

    mod = load_artifact(compiled_dir)
    print(f"Module: {len(mod.functions)} functions\n")

    # Build producer map same as _emit_compute_graph_section
    producer: dict[str, tuple[int, int]] = {}
    for fi, func in enumerate(mod.functions):
        for oi, (out_name, _out_type, _consumed_internally) in enumerate(func.outputs):
            clean = out_name.lstrip("%")
            producer[clean] = (fi, oi)

    print("=== Producer Map (output SSA names -> (func, out)) ===")
    for name, (fi, oi) in sorted(producer.items(), key=lambda x: (x[1][0], x[1][1])):
        print(f"  %{name} -> func[{fi}].out[{oi}]")

    print("\n=== Fallback Analysis ===")
    fallbacks = []
    for fi, func in enumerate(mod.functions):
        for in_idx, (in_name, in_type_str) in enumerate(func.inputs):
            clean = in_name.lstrip("%")
            if clean not in producer:
                fallbacks.append((fi, func.name, in_idx, in_name, in_type_str))
                print(f"  func[{fi}] {func.name} input[{in_idx}] {in_name} type={in_type_str}")
                print("    -> FALLBACK to func[0].out[0]")

    print(f"\n{'='*60}")
    if fallbacks:
        print(f"Found {len(fallbacks)} fallback(s) to (0,0)")
        for fi, name, ii, in_name, in_type in fallbacks:
            print(f"  [{fi}] {name} input[{ii}]: {in_name} ({in_type})")
        print("\nThese are incorrectly wired to main_0 output 0!")
    else:
        print("No fallbacks - all SSA bindings found in producer map")

    return len(fallbacks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_compute_graph_wiring.py <compiled_dir>")
        sys.exit(1)
    sys.exit(diagnose(sys.argv[1]))
