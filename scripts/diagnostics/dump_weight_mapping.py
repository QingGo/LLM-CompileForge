#!/usr/bin/env python3
"""Dump SFCF name_mapping entries for a compiled model.

Shows per-function weight argument bindings with weight names, HF key aliases,
and tensor shapes (when artifact loading succeeds).

Usage:
    python scripts/dump_weight_mapping.py outputs/compiled/opt_125m_fresh
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler.sfcf_parser import parse_compute_graph, parse_sfcf_blob  # noqa: I001

# =====================================================================
# Helpers
# =====================================================================

def _pretty_symbol(sym: str) -> str:
    """Strip _mlir_ciface_ prefix for readability."""
    return re.sub(r'^_*mlir_ciface_', '', sym)


def _shape_str(shape: list[int]) -> str:
    """Format shape for display."""
    if not shape:
        return "()"
    return "(" + ", ".join(str(s) if s > 0 else "?" for s in shape) + ")"


def _get_weight_shapes(artifact_dir: str) -> dict[str, tuple[int, ...]]:
    """Try to load artifact weights and return {name: shape} mapping.

    Falls back to empty dict on any failure.
    """
    try:
        from compiler.serialize import load_artifact
        artifact = load_artifact(artifact_dir)
        shapes: dict[str, tuple[int, ...]] = {}
        for func in artifact.functions:
            for wname, wtensor in func.weights.items():
                if wname not in shapes:
                    shapes[wname] = tuple(wtensor.shape)
        return shapes
    except Exception:
        return {}


def _get_hf_key_map(artifact_dir: str) -> dict[str, str]:
    """Read hf_key_map from metadata.json."""
    meta_path = os.path.join(artifact_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("hf_key_map", {})
        except Exception:
            pass
    return {}


def _get_weight_classification(artifact_dir: str) -> dict[str, Any]:
    """Read weight_classification from metadata.json."""
    meta_path = os.path.join(artifact_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("weight_classification", {})
        except Exception:
            pass
    return {}


def _get_num_functions(artifact_dir: str) -> int:
    """Read num_functions from metadata.json."""
    meta_path = os.path.join(artifact_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("num_functions", 0)
        except Exception:
            pass
    return 0


# =====================================================================
# Main
# =====================================================================

def dump_weight_mapping(model_dir: str) -> int:
    """Dump per-function weight mapping for a compiled model.

    Returns 0 on success, 1 on error.
    """
    bin_path = os.path.join(model_dir, "constants.bin")
    if not os.path.exists(bin_path):
        print(f"❌ constants.bin not found in {model_dir}")
        return 1

    # Read blob
    with open(bin_path, "rb") as f:
        blob = f.read()

    # Parse
    try:
        name_mapping, constants, graph_pos, sfcf_version = parse_sfcf_blob(blob)
        graph, _ = parse_compute_graph(blob, graph_pos, version=sfcf_version)
    except Exception as e:
        print(f"❌ Failed to parse SFCF blob: {e}")
        return 1

    # Load auxiliary data (best-effort)
    weight_shapes = _get_weight_shapes(model_dir)
    hf_key_map = _get_hf_key_map(model_dir)
    weight_classification = _get_weight_classification(model_dir)
    num_functions_meta = _get_num_functions(model_dir)

    functions = graph["functions"]
    global_input = graph["global_input"]
    global_output = graph["global_output"]

    print(f"SFCF version: {sfcf_version}")
    print(f"Functions:    {len(functions)}")
    print(f"Name mapping: {len(name_mapping)} entries")
    print(f"Constants:    {len(constants)} tensors")
    print(f"Global input: func[{global_input[0]}].arg[{global_input[1]}]")
    print(f"Global output: func[{global_output[0]}].output[{global_output[1]}]")
    if num_functions_meta:
        print(f"(metadata.json says {num_functions_meta} functions)")

    # Collect SSA producers for annotation
    func_output_counts: dict[int, int] = {}
    for fi, func in enumerate(functions):
        func_output_counts[fi] = func["num_outputs"]

    # Print per-function header
    for fi, func in enumerate(functions):
        sym = _pretty_symbol(func["symbol"])
        inputs = func["inputs"]
        outputs = func["outputs"]

        # Determine "assert_names" from weight_classification
        wc_key = f"main_{fi}"
        wc = weight_classification.get(wc_key, {})
        const_names: list[str] = wc.get("constants", [])

        # Also collect weight names from the actual bindings
        actual_weight_names: list[str] = []
        for inp in inputs:
            if inp["binding"][0] == "weight":
                actual_weight_names.append(inp["binding"][1])

        # Build the header
        header_parts = [f"=== {sym}"]
        if actual_weight_names:
            # Show first few weight names as a hint
            short_list = actual_weight_names[:5]
            suffix = "..." if len(actual_weight_names) > 5 else ""
            header_parts.append(f'({len(actual_weight_names)} weights, first: {short_list}{suffix})')
        print()
        print(" ".join(header_parts))
        print(f"  Outputs: {len(outputs)} tensors, shapes: {[_shape_str(o['shape']) for o in outputs]}")

        # Print each input arg
        weight_idx_in_func = 0  # counter for weight args within this function
        for ai, inp in enumerate(inputs):
            binding = inp["binding"]
            btype = binding[0]
            shape = inp["shape"]
            shape_s = _shape_str(shape)

            if btype == "global_input":
                print(f"  arg[{ai}]: GlobalInput (shape={shape_s})")

            elif btype == "weight":
                wname = binding[1]
                # Get shape
                wshape = weight_shapes.get(wname)
                wshape_s = f" (shape: {wshape})" if wshape else ""

                # Get HF key alias
                hf_key = hf_key_map.get(wname, "")
                hf_s = f"  (aka {hf_key})" if hf_key else ""

                print(f"  arg[{ai}]: Weight → {wname}{wshape_s}{hf_s}")
                weight_idx_in_func += 1

            elif btype == "ssa":
                pf = binding[1]
                oi = binding[2]
                src_sym = _pretty_symbol(functions[pf]["symbol"]) if pf < len(functions) else f"func_{pf}"
                print(f"  arg[{ai}]: SSA ← {src_sym}.output[{oi}] (shape={shape_s})")

            else:
                print(f"  arg[{ai}]: Unknown binding {binding}")

        # Print constants belonging to this function (compact)
        if const_names:
            print(f"  Constants ({len(const_names)}): {const_names[:6]}{'...' if len(const_names) > 6 else ''}")

    return 0


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/compiled/opt_125m_fresh"
    if model_dir == "--help" or model_dir == "-h":
        print("Usage: python scripts/dump_weight_mapping.py [model_dir]")
        print(f"  Default model_dir: {model_dir}")
        return 0
    return dump_weight_mapping(model_dir)


if __name__ == "__main__":
    sys.exit(main())
