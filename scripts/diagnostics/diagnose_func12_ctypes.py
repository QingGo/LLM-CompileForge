#!/usr/bin/env python3
"""Direct ctypes call to _mlir_ciface_main_12 — isolate dylib vs Rust runtime.

Compares the dylib's func[12] output against the Python executor reference
using inputs from the executor's intermediate results.

This tells us whether func[12]'s low cos (0.538 reported) is caused by:
  - The compiled dylib itself (dylib cos ≈ Rust cos = 0.538) → pipeline bug
  - The Rust runtime's calling convention (ctypes cos ≈ 0.999, Rust cos = 0.538) → Rust bug

Usage:
    source .venv/bin/activate
    export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/torch/lib:\
        $PWD/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"
    export KMP_DUPLICATE_LIB_OK=TRUE
    unset CONDA_PREFIX
    python scripts/diagnose_func12_ctypes.py [--model-dir outputs/compiled/opt_125m_fresh]
"""

from __future__ import annotations

import argparse
import ctypes
import faulthandler
import os
import sys
from typing import Any

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

faulthandler.enable()

from compiler.sfcf_parser import (  # noqa: E402
    make_memref_descriptor,
    parse_compute_graph,
    parse_sfcf_blob,
    parse_sret_outputs,
)
from scripts._cos import cosine_similarity  # noqa: E402
from scripts.ctypes_forward import (  # noqa: E402
    run_ctypes,
    run_python_executor,
)

# =====================================================================
# Helpers
# =====================================================================


def _desc_size(rank: int) -> int:
    return 24 + 16 * rank


def _sret_size_for_func(func_def: dict[str, Any]) -> int:
    total = 0
    for out in func_def["outputs"]:
        total += _desc_size(out["rank"])
    return total


def _binding_str(binding: tuple[object, ...]) -> str:
    if binding[0] == "global_input":
        return "GlobalInput"
    if binding[0] == "weight":
        return f"Weight({binding[1]})"
    if binding[0] == "ssa":
        return f"SSA(func[{binding[1]}].output[{binding[2]}])"
    return f"Unknown({binding})"


def _get_ssa_output(
    func_outputs: list[list[np.ndarray[Any, Any]]],
    pf: int,
    oi: int,
    io_shape: list[int],
) -> np.ndarray[Any, Any]:
    """Retrieve an SSA output or synthesize zeros as fallback."""
    if pf < len(func_outputs) and oi < len(func_outputs[pf]):
        arr = func_outputs[pf][oi]
    else:
        shape = [int(s) if s > 0 else 1 for s in io_shape] or [1]
        arr = np.zeros(shape, dtype=np.float32)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


# =====================================================================
# Main diagnostic
# =====================================================================


def diagnose_func12(
    model_dir: str,
    dylib_path: str | None = None,
    func_index: int = 12,
) -> int:
    """Run ctypes-forward for a single function and compare with Python executor.

    Strategy:
      1. Run the full ctypes pipeline to get ALL dylib func_outputs.
         (func[0] produces ~200+ SSA weight outputs that later funcs need.)
      2. Run the Python executor for the reference output.
      3. Call func[12] directly via ctypes using the dylib's func[0] outputs
         and the Python executor's func[11] output as inputs.
      4. Compare the result with the Python executor's func[12] output.

    Returns 0 on pass (cos > 0.99), 1 on failure.
    """
    if dylib_path is None:
        dylib_path = os.path.join(model_dir, "libopt_125m.dylib")

    # Load dylib and parse SFCF blob
    lib = ctypes.CDLL(dylib_path)

    constants_bin = os.path.join(model_dir, "constants.bin")
    with open(constants_bin, "rb") as f:
        blob = f.read()

    _, _, graph_pos, sfcf_version = parse_sfcf_blob(blob)
    graph, _ = parse_compute_graph(blob, graph_pos, version=sfcf_version)
    functions = graph["functions"]

    if func_index >= len(functions):
        print(f"ERROR: func[{func_index}] out of range (total {len(functions)})")
        return 1

    func_def = functions[func_index]
    symbol = func_def["symbol"]

    print(f"{'='*72}")
    print(f"Diagnostic: {symbol}")
    print(f"  Function index: {func_index}")
    print(f"  Num inputs: {func_def['num_inputs']}")
    print(f"  Num outputs: {func_def['num_outputs']}")
    print()

    print("  Input bindings:")
    for ai, inp in enumerate(func_def["inputs"]):
        shape_str = "x".join(str(s) if s > 0 else "?" for s in inp["shape"])
        print(f"    arg[{ai:2d}]: {_binding_str(inp['binding']):42s} "
              f"shape=[{shape_str}]")
    print()

    print("  Output shapes:")
    for oi, out in enumerate(func_def["outputs"]):
        shape_str = "x".join(str(s) if s > 0 else "?" for s in out["shape"])
        print(f"    output[{oi}]: rank={out['rank']} shape=[{shape_str}]")
    print()

    # Run the full ctypes pipeline (populates ALL func_outputs,
    # including func[0]'s ~200+ SSA weight outputs)
    print("  Running full ctypes pipeline (to get all func outputs)...")
    try:
        dylib_full = run_ctypes(model_dir, dylib_path=dylib_path)
    except Exception as e:
        print(f"  ERROR running ctypes pipeline: {e}")
        return 1
    dylib_func_outputs = dylib_full._func_outputs
    num_dylib_funcs = len(dylib_func_outputs)
    print(f"  ctypes pipeline returned {num_dylib_funcs} functions")

    # Check if func[0] has enough outputs
    if len(dylib_func_outputs) > 0:
        print(f"  func[0] has {len(dylib_func_outputs[0])} outputs")
        # Show some representative shapes for debugging
        for oi in [0, 49, 50, 209, 210]:
            if oi < len(dylib_func_outputs[0]):
                arr = dylib_func_outputs[0][oi]
                print(f"    output[{oi}]: shape={list(arr.shape)} dtype={arr.dtype}")
    print()

    # Run Python executor for reference
    print("  Running Python executor for reference...")
    try:
        py_ref = run_python_executor(model_dir)
    except Exception as e:
        print(f"  ERROR running Python executor: {e}")
        return 1
    num_py_funcs = len(py_ref)
    print(f"  Python executor returned {num_py_funcs} functions")
    ref_output = py_ref[func_index]
    print(f"  Reference func[{func_index}] shape: {list(ref_output.shape)}")

    # Also get func[11] reference output (needed as input)
    if func_index - 1 >= 0:
        try:
            prev_ref = py_ref[func_index - 1]
            print(f"  Reference func[{func_index - 1}] shape: {list(prev_ref.shape)}")
        except Exception:
            print(f"  WARNING: could not get func[{func_index - 1}] reference")
    print()

    # Build input MemRef descriptors
    print("  Building input MemRef descriptors...")
    input_descs: list[ctypes.Structure] = []
    input_args: list[object] = []
    _keep_arrs: list[np.ndarray[Any, Any]] = []

    for ai, inp in enumerate(func_def["inputs"]):
        binding = inp["binding"]
        io_shape = inp["shape"]
        btype = binding[0]

        if btype == "ssa":
            pf, oi = binding[1], binding[2]
            # For SSA from func[0]: use dylib func_outputs (weight outputs)
            # For SSA from func[11]: use py_ref func_outputs (hidden state)
            if pf < num_dylib_funcs and oi < len(dylib_func_outputs[pf]):
                arr = _get_ssa_output(dylib_func_outputs, pf, oi, io_shape)
            elif pf < num_py_funcs:
                arr = _get_ssa_output(py_ref._func_outputs, pf, oi, io_shape)
            else:
                shape = [int(s) if s > 0 else 1 for s in io_shape] or [1]
                arr = np.zeros(shape, dtype=np.float32)
                print(f"    WARNING: arg[{ai}] fallback to zeros "
                      f"shape={shape}")
        elif btype == "weight":
            # Load from artifact weights
            shape = [int(s) if s > 0 else 1 for s in io_shape] or [1]
            arr = np.zeros(shape, dtype=np.float32)
            print(f"    WARNING: arg[{ai}] weight binding resolved as zeros")
        elif btype == "global_input":
            # Standard input_ids: (2, 4)
            shape = [int(s) if s > 0 else 1 for s in io_shape] or [1]
            arr = np.zeros(shape, dtype=np.int64)
            print(f"    WARNING: arg[{ai}] global_input binding, using zeros")
        else:
            shape = [int(s) if s > 0 else 1 for s in io_shape] or [1]
            arr = np.zeros(shape, dtype=np.float32)
            print(f"    WARNING: arg[{ai}] unknown binding {binding}")

        # Ensure float32 contiguous (convert int64 to float32 for memref)
        if arr.dtype == np.int64 and btype == "global_input":
            pass  # keep int64 for index inputs
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        _keep_arrs.append(arr)
        desc = make_memref_descriptor(arr)
        input_descs.append(desc)
        input_args.append(ctypes.byref(desc))

        flat = arr.ravel()
        nprint = min(3, len(flat))
        disp = f"data[:{nprint}]={[f'{v:.4f}' for v in flat[:nprint]]}"
        print(f"    arg[{ai:2d}]: {_binding_str(binding):42s} "
              f"shape={list(arr.shape)} dtype={arr.dtype} "
              f"{disp}")

    print(f"  Total inputs: {len(input_args)}")
    print()

    # Build output sret buffer
    sret_size = max(_sret_size_for_func(func_def), 4096)
    sret = (ctypes.c_uint8 * sret_size)()
    print(f"  Output sret buffer: {sret_size} bytes")
    print()

    # Look up ciface symbol
    try:
        kernel = getattr(lib, symbol)
    except AttributeError:
        print(f"  ERROR: symbol '{symbol}' not found in dylib")
        return 1

    # Call ciface function
    all_args = [ctypes.byref(sret)] + input_args
    kernel.argtypes = [ctypes.c_void_p] * len(all_args)
    kernel.restype = None

    print(f"  Calling {symbol} with {len(all_args)} args "
          f"(1 sret + {len(input_args)} inputs)...")
    kernel(*all_args)
    print("  Function returned successfully.")
    print()

    # Parse output from sret buffer
    outputs = parse_sret_outputs(bytes(sret), func_def["outputs"])
    if not outputs:
        print("  ERROR: no outputs parsed from sret buffer")
        return 1

    ctypes_output = outputs[0]
    print(f"  ctypes output shape: {list(ctypes_output.shape)}")
    print()

    # Compare with reference
    if ctypes_output.shape != ref_output.shape:
        print(f"  SHAPE MISMATCH: ctypes={list(ctypes_output.shape)} "
              f"ref={list(ref_output.shape)}")
        c_flat = ctypes_output.ravel()
        r_flat = ref_output.ravel()
        min_len = min(len(c_flat), len(r_flat))
        cos = cosine_similarity(c_flat[:min_len], r_flat[:min_len])
        print(f"  Truncated cos (first {min_len} elements): {cos:.6f}")
    else:
        cos = cosine_similarity(ctypes_output, ref_output)
        print(f"  Cosine similarity: {cos:.6f}")

    # Diagnostic verdict
    print()
    if cos > 0.99:
        print(f"  PASS: cos={cos:.6f} > 0.99")
        print()
        print("  DIAGNOSIS: The compiled dylib's func[12] output matches")
        print("  the Python executor reference. This means the BUG is in")
        print("  the Rust runtime's calling convention (weight ordering,")
        print("  memref layout, input shapes, or ciface wrapper).")
        return 0
    else:
        print(f"  FAIL: cos={cos:.6f}")
        print()
        print("  DIAGNOSIS: The compiled dylib's func[12] output DOES NOT")
        print("  match the Python executor reference. The bug is likely in")
        print("  the compiled dylib / lowering pipeline (bufferization,")
        print("  LLVM codegen, FP accumulation, or pass ordering).")
        print()
        print("  Next steps:")
        print("  1. Check the lowering pipeline for func[12]'s MLIR:")
        print(f"     grep -n 'main_12' {model_dir}/model.lowered.mlir | head -20")
        print("  2. Enable IR dump for func[12]'s stage and inspect")
        print("  3. Check if vectorization or bufferization changed shapes")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose func[12] cos by direct ctypes call"
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/compiled/opt_125m_fresh",
        help="Compiled model directory (default: outputs/compiled/opt_125m_fresh)",
    )
    parser.add_argument(
        "--func",
        type=int,
        default=12,
        help="Function index to diagnose (default: 12)",
    )
    parser.add_argument(
        "--dylib",
        default=None,
        help="Path to .dylib (default: <model_dir>/libopt_125m.dylib)",
    )
    args = parser.parse_args()

    return diagnose_func12(
        model_dir=args.model_dir,
        dylib_path=args.dylib,
        func_index=args.func,
    )


if __name__ == "__main__":
    sys.exit(main())
