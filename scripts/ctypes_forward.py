#!/usr/bin/env python3
"""Call the compiled .dylib directly via ctypes, using Python executor weights.

Compares two paths side-by-side:
  1. Python MlirExecutor (sf dialect)  — cos ≈ 0.999994
  2. Python ctypes calling .dylib       — cos = ?
  3. Rust executor calling .dylib       — cos ≈ 0.869

Expected signal patterns:
  - If ctypes cos ≈ Python executor cos (0.999) -> bug is in Rust runtime's
    ciface calling convention (weight ordering, memref layout, input shape)
  - If ctypes cos ≈ Rust runtime cos (0.869) -> bug is in compiled dylib /
    lowering pipeline (bufferization, LLVM codegen, FP accumulation)

Usage:
    from scripts.ctypes_forward import run_ctypes, run_python_executor, DylibResult
"""

import ctypes
import faulthandler
import os
import sys

import numpy as np

from compiler.sfcf_parser import (
    make_memref_descriptor,
    parse_compute_graph,
    parse_sfcf_blob,
    parse_sret_outputs,
)

faulthandler.enable()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =====================================================================
# Helpers
# =====================================================================



# =====================================================================
# Convenience API: DylibResult + run_ctypes / run_python_executor
# =====================================================================


class DylibResult:
    """Holds per-function outputs and final logits from a forward pass.

    Supports both indexing patterns used by tests:
        ``result[batch, pos]`` — per-position logit vector (via final logits)
        ``result[fi]``         — first output array of function *fi* (via func_outputs)
    """

    def __init__(
        self, func_outputs: list[list[np.ndarray]], logits: np.ndarray
    ) -> None:
        self._func_outputs = func_outputs
        self._logits = logits

    @property
    def logits(self) -> np.ndarray:
        return self._logits

    @property
    def func_outputs(self) -> list[list[np.ndarray]]:
        return self._func_outputs

    def __getitem__(self, key: int | tuple[int, ...]) -> np.ndarray:
        if isinstance(key, tuple):
            return self._logits[key]
        # Single int -> first output of function *key*
        return self._func_outputs[key][0]

    def __len__(self) -> int:
        return len(self._func_outputs)


def run_ctypes(
    artifact_dir: str = "./compiled/opt_125m_fresh",
    dylib_path: str | None = None,
    input_ids: np.ndarray | None = None,
) -> DylibResult:
    """Run compiled .dylib via ctypes and return per-function + logit results.

    Args:
        artifact_dir: Path to compiled artifact directory (default:
            ``./compiled/opt_125m_fresh``).
        dylib_path: Path to the .dylib (default: ``<artifact_dir>/libopt_125m.dylib``).
        input_ids: Input token IDs (default: batch=2, seq=4 sample).

    Returns:
        ``DylibResult`` with ``.logits`` (final logits) and ``.func_outputs``
        (per-function output list).
    """
    if input_ids is None:
        input_ids = np.array(
            [[2, 32826, 85, 4129], [0, 0, 0, 0]], dtype=np.int64
        )
    if dylib_path is None:
        dylib_path = os.path.join(artifact_dir, "libopt_125m.dylib")

    # Load artifact weights
    from compiler.serialize import load_artifact

    artifact = load_artifact(artifact_dir)
    all_weights: dict[str, np.ndarray] = {}
    for func in artifact.functions:
        for wname, wtensor in func.weights.items():
            if wname not in all_weights:
                all_weights[wname] = np.ascontiguousarray(wtensor.numpy())
    hf_key_map = artifact.metadata.get("hf_key_map", {})
    ws = artifact.metadata.get("weight_source", {})
    tied_weights = (
        ws.get("tied_weights", {}) or artifact.metadata.get("tied_weights", {})
    )

    # Load dylib + parse SFCF blob
    lib = ctypes.CDLL(dylib_path)
    data_ptr = ctypes.cast(
        ctypes.addressof(ctypes.c_int64.in_dll(lib, "serveforge_constants_data")),
        ctypes.c_void_p,
    )
    size_ptr = ctypes.cast(
        ctypes.addressof(ctypes.c_int64.in_dll(lib, "serveforge_constants_size")),
        ctypes.POINTER(ctypes.c_uint64),
    )
    blob_bytes = bytes(
        (ctypes.c_uint8 * size_ptr[0]).from_address(data_ptr.value)
    )
    _, sfcf_constants, graph_pos, sfcf_version = parse_sfcf_blob(blob_bytes)
    graph, _ = parse_compute_graph(blob_bytes, graph_pos, version=sfcf_version)

    # Weight lookup with multi-strategy resolution
    def _get_weight(name: str) -> np.ndarray:
        if name in all_weights:
            return all_weights[name]
        hf_key = hf_key_map.get(name)
        if hf_key and hf_key in all_weights:
            return all_weights[hf_key]
        for alias_name, primary_name in tied_weights.items():
            if primary_name == name:
                alias_hf = hf_key_map.get(alias_name)
                if alias_hf and alias_hf in all_weights:
                    return all_weights[alias_hf]
                if alias_name in all_weights:
                    return all_weights[alias_name]
        bare_name = name.split(".", 1)[-1] if "." in name else name
        if bare_name != name:
            if bare_name in all_weights:
                return all_weights[bare_name]
            if bare_name in sfcf_constants:
                return np.ascontiguousarray(sfcf_constants[bare_name])
        prefixed = f"main_0.{name}"
        if prefixed in all_weights:
            return all_weights[prefixed]
        if name in sfcf_constants:
            return np.ascontiguousarray(sfcf_constants[name])
        raise KeyError(f"Weight '{name}' not found")

    # Run forward pass
    sret_size = 131072
    func_outputs: list[list[np.ndarray]] = [
        [] for _ in range(len(graph["functions"]))
    ]

    for fi, func_def in enumerate(graph["functions"]):
        symbol = func_def["symbol"]
        try:
            kernel = getattr(lib, symbol)
        except AttributeError:
            import logging
            logging.warning("ctypes_forward: symbol '%s' not found in dylib, skipping function %d", symbol, fi)
            continue

        input_descs: list[ctypes.Structure] = []
        input_args: list[ctypes.pointer] = []
        _keep_arrs: list[np.ndarray] = []

        for inp in func_def["inputs"]:
            binding = inp["binding"]
            io_shape = inp["shape"]

            if binding[0] == "global_input":
                arr = input_ids
            elif binding[0] == "weight":
                key = binding[1]
                try:
                    arr = _get_weight(key)
                except KeyError:
                    shape = [int(s) if s > 0 else 1 for s in io_shape] or [1]
                    arr = np.zeros(shape, dtype=np.float32)
            elif binding[0] == "ssa":
                pf, oi = binding[1], binding[2]
                if pf < len(func_outputs) and oi < len(func_outputs[pf]):
                    arr = func_outputs[pf][oi]
                else:
                    shape = [int(s) if s > 0 else 1 for s in io_shape] or [1]
                    arr = np.zeros(shape, dtype=np.float32)
            else:
                raise ValueError(f"Unknown binding: {binding}")

            if arr.dtype != np.float32 and arr.dtype != np.int64:
                arr = arr.astype(np.float32)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            _keep_arrs.append(arr)
            desc = make_memref_descriptor(arr)
            input_descs.append(desc)
            input_args.append(ctypes.byref(desc))

        sret = (ctypes.c_uint8 * sret_size)()
        all_args = [ctypes.byref(sret)] + input_args
        kernel.argtypes = [ctypes.c_void_p] * len(all_args)
        kernel.restype = None
        kernel(*all_args)
        outputs = parse_sret_outputs(bytes(sret), func_def["outputs"])
        func_outputs[fi] = outputs

    go_func, go_idx = graph["global_output"]
    logits = func_outputs[go_func][go_idx]
    return DylibResult(func_outputs, logits)


def run_python_executor(
    artifact_dir: str = "./compiled/opt_125m_fresh",
    input_ids: np.ndarray | None = None,
) -> DylibResult:
    """Run Python MlirExecutor and return per-function + logit results.

    Per-function outputs are captured via the executor's internal dump
    mechanism (``dump_dir`` parameter).  The final logits are the output
    of the last function.

    Args:
        artifact_dir: Path to compiled artifact directory.
        input_ids: Input token IDs (default: batch=2, seq=4 sample).

    Returns:
        ``DylibResult`` with ``.logits`` (final logits) and ``.func_outputs``
        (per-function output list reconstructed from the dump files).
    """
    import tempfile

    import torch

    from compiler.serialize import load_artifact
    from engine.mlir_executor import MlirExecutor
    from hal.pytorch_backend import PyTorchBackend

    if input_ids is None:
        input_ids_np = np.array(
            [[2, 32826, 85, 4129], [0, 0, 0, 0]], dtype=np.int64
        )
    else:
        input_ids_np = input_ids

    artifact = load_artifact(artifact_dir)
    num_funcs = len(artifact.functions)

    with tempfile.TemporaryDirectory(prefix="py_dump_") as dump_dir:
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(artifact, backend, dump_dir=dump_dir)

        with torch.no_grad():
            logits_tensor = executor.forward(
                torch.tensor(input_ids_np)
            )
        logits = np.ascontiguousarray(logits_tensor.numpy().astype(np.float32))

        func_outputs: list[list[np.ndarray]] = [[] for _ in range(num_funcs)]
        for fi in range(num_funcs):
            # Match new dump naming: py_func_{fi}_out0_*.npy (first output)
            import glob as _glob
            pattern = os.path.join(dump_dir, f"py_func_{fi}_out0_*.npy")
            matches = _glob.glob(pattern)
            if not matches:
                # Fall back to old naming
                fpath = os.path.join(dump_dir, f"py_func_{fi}_0.npy")
                if os.path.exists(fpath):
                    matches = [fpath]
            if matches:
                func_outputs[fi] = [np.load(matches[0])]
        return DylibResult(func_outputs, logits)
