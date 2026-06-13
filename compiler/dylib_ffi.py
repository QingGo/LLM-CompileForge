"""ctypes/FFI helpers for calling compiled dylib functions.

Provides utilities for constructing MemRef descriptors, allocating/parsing
sret (struct-return) output buffers, and verifying output tensor shapes
against the compute graph.

These functions are the Python-side counterpart to the Rust runtime's
``MemRefDesc`` and sret parsing in ``runtime/src/hal/cpu/``.
"""

import ctypes
import struct
from typing import Any

import numpy as np

# =====================================================================
# MemRef descriptor construction
# =====================================================================


def make_memref_descriptor(arr: np.ndarray[Any, Any]) -> Any:
    """Build a ctypes MemRef descriptor for a numpy array.

    The descriptor is a packed struct:
        allocated: void* (8)
        aligned:   void* (8)
        offset:    int64 (8)
        sizes:     int64[rank] (8*rank)
        strides:   int64[rank] (8*rank)
    Total: 24 + 16*rank bytes.

    Matches the binary layout of ``MemRefDesc<RANK>`` in
    ``runtime/src/hal/cpu/memref.rs``.
    """
    rank = arr.ndim
    fields = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * rank),
        ("strides", ctypes.c_int64 * rank),
    ]
    desc_type = type(f"MemRefDesc{rank}", (ctypes.Structure,), {"_fields_": fields})

    # Element-wise strides (in number of elements, not bytes)
    elem_size = arr.itemsize
    elem_strides = tuple(s // elem_size for s in arr.strides)

    return desc_type(
        allocated=arr.ctypes.data_as(ctypes.c_void_p),
        aligned=arr.ctypes.data_as(ctypes.c_void_p),
        offset=0,
        sizes=(ctypes.c_int64 * rank)(*arr.shape),
        strides=(ctypes.c_int64 * rank)(*elem_strides),
    )


# =====================================================================
# SRET output parsing
# =====================================================================

# Default sret buffer size (64 KB) — generous for most compiled functions.
# Callers should prefer compute_sret_size() when output descriptors are known.
DEFAULT_SRET_SIZE: int = 65536


def compute_sret_size(
    output_defs: list[dict[str, Any]],
    floor: int = 4096,
) -> int:
    """Compute required sret buffer size from output descriptors.

    Each ciface function writes one packed output memref descriptor per
    output.  The descriptor size is ``24 + 16 * rank`` bytes.  The total
    sret size is the sum of all output descriptor sizes, floored at
    *floor* bytes.
    """
    if output_defs:
        total = sum(24 + 16 * od.get("rank", 0) for od in output_defs)
        return max(total, floor)
    return floor


def desc_size(rank: int) -> int:
    """Size in bytes of a single sret memref descriptor for the given rank."""
    return 24 + 16 * rank


def parse_sret_outputs(sret_bytes: bytes, output_defs: list[dict[str, Any]]) -> list[np.ndarray[Any, Any]]:
    """Parse output tensors from the sret buffer written by the ciface kernel.

    Each output descriptor in the sret buffer has layout:
        offset 0:  allocated (i64)
        offset 8:  aligned (i64)  ← pointer to actual output data
        offset 16: offset (i64)
        offset 24: sizes[i64] * rank
        after sizes: strides[i64] * rank
    Total per-descriptor: 24 + 16*rank bytes.
    """
    tensors = []
    offset = 0
    for od in output_defs:
        rank = od["rank"]
        dsize = desc_size(rank)
        desc = sret_bytes[offset : offset + dsize]

        aligned = struct.unpack_from("<Q", desc, 8)[0]

        # Read runtime sizes from descriptor; fall back to static shape for 0 dims
        runtime_sizes = []
        for i in range(rank):
            s = struct.unpack_from("<q", desc, 24 + 8 * i)[0]
            # 0 means dynamic; use static shape as hint
            if s <= 0 or s > 1_000_000_000:
                s = od["shape"][i] if i < len(od["shape"]) and od["shape"][i] > 0 else 1
            runtime_sizes.append(int(s))

        n = int(np.prod(runtime_sizes))
        if n > 0 and aligned != 0:
            buf = (ctypes.c_float * n).from_address(aligned)
            arr = np.array(buf, dtype=np.float32).reshape(runtime_sizes)
        else:
            arr = np.array([], dtype=np.float32)

        tensors.append(arr)
        offset += dsize

    return tensors


# =====================================================================
# Output shape verification
# =====================================================================


def verify_output_shapes(
    func_outputs: list[list[np.ndarray[Any, Any]]],
    compute_graph_functions: list[dict[str, Any]],
) -> list[str]:
    """Verify parsed sret output tensors match compute graph declarations.

    For each function, checks:
      1. Output tensor count matches graph declaration
      2. Tensor rank matches declared rank
      3. Static shape dimensions match (dynamic dims marked as 0 are skipped)
      4. Null data pointer detection (aligned==0 in descriptor yields empty array)

    Args:
        func_outputs: Per-function list of parsed output tensors.
        compute_graph_functions: List of function dicts with ``outputs`` key,
            each output having ``rank`` and ``shape`` fields.

    Returns:
        List of error messages. Empty list = all checks pass.
    """
    errors: list[str] = []

    for fi, outputs in enumerate(func_outputs):
        if fi >= len(compute_graph_functions):
            errors.append(f"func[{fi}]: no compute graph entry for this function index")
            continue

        func_def = compute_graph_functions[fi]
        expected_outputs = func_def["outputs"]
        symbol = func_def.get("symbol", f"func_{fi}")

        # Check 1: output count
        if len(outputs) != len(expected_outputs):
            errors.append(f"func[{fi}] ({symbol}): expected {len(expected_outputs)} output(s), got {len(outputs)}")
            continue

        for oi, (arr, expected) in enumerate(zip(outputs, expected_outputs, strict=True)):
            expected_rank = expected["rank"]
            expected_shape = expected["shape"]

            # Check 2: null data pointer
            if arr.size == 0 and expected_rank > 0:
                static_dims = [d for d in expected_shape if d > 0] if expected_shape else []
                if static_dims:
                    errors.append(
                        f"func[{fi}] ({symbol}) output[{oi}]: "
                        f"null data pointer (aligned==0 in descriptor), "
                        f"expected shape {expected_shape}"
                    )
                    continue

            # Check 3: rank
            if arr.ndim != expected_rank:
                errors.append(
                    f"func[{fi}] ({symbol}) output[{oi}]: "
                    f"expected rank {expected_rank}, "
                    f"got rank {arr.ndim} (actual shape={list(arr.shape)}, "
                    f"expected shape={expected_shape})"
                )
                continue

            # Check 4: static shape dimensions
            for dim_i in range(expected_rank):
                if dim_i < len(expected_shape):
                    static_val = expected_shape[dim_i]
                    if static_val > 0:
                        if dim_i >= len(arr.shape):
                            errors.append(
                                f"func[{fi}] ({symbol}) output[{oi}]: "
                                f"dim[{dim_i}] expected {static_val}, "
                                f"but array has only {len(arr.shape)} dims"
                            )
                        elif arr.shape[dim_i] != static_val:
                            errors.append(
                                f"func[{fi}] ({symbol}) output[{oi}]: "
                                f"dim[{dim_i}] expected {static_val}, "
                                f"got {arr.shape[dim_i]} "
                                f"(actual shape={list(arr.shape)}, "
                                f"expected shape={expected_shape})"
                            )

    return errors


# =====================================================================
# Proto-based compute graph loading (replaces legacy SFCF parsing)
# =====================================================================


def _read_proto_symbol(lib, name: str, size_name: str) -> bytes:
    """Read a protobuf-encoded data symbol from a loaded dylib.

    The dylib exports ``name`` as a const array address and ``size_name``
    as a u64 companion size value.
    """
    data_ptr = ctypes.cast(
        ctypes.addressof(ctypes.c_int64.in_dll(lib, name)),
        ctypes.c_void_p,
    )
    size_ptr = ctypes.cast(
        ctypes.addressof(ctypes.c_int64.in_dll(lib, size_name)),
        ctypes.POINTER(ctypes.c_uint64),
    )
    return bytes((ctypes.c_uint8 * size_ptr[0]).from_address(data_ptr.value))


def load_graph_from_proto(
    lib,
    dylib_constants: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Load compute graph from proto symbols embedded in the dylib.

    Replaces the legacy ``parse_sfcf_blob`` / ``parse_compute_graph``
    path.  Reads ``sfa_abi`` and ``sfa_weights`` proto symbols and
    builds a compute graph dict compatible with existing ctypes code.

    Returns a dict with:
        functions: list of per-function dicts (symbol, inputs, outputs)
        global_input: (func_idx, arg_idx) tuple
        global_output: (func_idx, output_idx) tuple
    """
    from gen.proto.python.sfa_abi_pb2 import (  # type: ignore[attr-defined]
        SfaAbiHeader,
        SfaWeightData,
        SFA_INPUT_GLOBAL,
        SFA_INPUT_WEIGHT,
        SFA_INPUT_SSA,
    )

    constants: dict[str, np.ndarray] = dylib_constants if dylib_constants is not None else {}

    abi_bytes = _read_proto_symbol(lib, "sfa_abi", "sfa_abi_size")
    abi = SfaAbiHeader()
    abi.ParseFromString(abi_bytes)

    weights_bytes = _read_proto_symbol(lib, "sfa_weights", "sfa_weights_size")
    weights = SfaWeightData()
    weights.ParseFromString(weights_bytes)

    for ce in weights.constant_entries:
        dtype_map = {0: np.float32, 1: np.float16, 2: np.float16,
                     3: np.int64, 4: np.int32, 5: np.int8, 6: np.uint8}
        np_dtype = dtype_map.get(ce.dtype_code, np.float32)
        shape = list(ce.shape) if ce.shape else [1]
        raw = ce.data
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        constants[ce.name] = arr

    functions: list[dict[str, Any]] = []
    for fm in abi.funcs:
        inputs: list[dict[str, Any]] = []
        for fld in fm.input_fields:
            binding: tuple
            if fld.kind == SFA_INPUT_GLOBAL:
                binding = ("global_input",)
            elif fld.kind == SFA_INPUT_WEIGHT:
                binding = ("weight", fld.weight_name)
            elif fld.kind == SFA_INPUT_SSA:
                binding = ("ssa", fld.ssa.producer_func, fld.ssa.producer_out)
            else:
                binding = ("global_input",)
            inputs.append({
                "binding": binding,
                "rank": fld.rank,
                "shape": list(fld.dims) if fld.dims else [0, 0],
            })
        outputs: list[dict[str, Any]] = []
        for od in fm.outputs:
            outputs.append({
                "rank": od.rank,
                "shape": list(od.dims) if od.dims else [0, 0],
                "consumed_internally": False,
            })
        functions.append({
            "symbol": fm.symbol,
            "num_inputs": fm.num_inputs,
            "num_outputs": len(fm.outputs),
            "inputs": inputs,
            "outputs": outputs,
        })

    global_input = (0, 0)
    for fi, fm in enumerate(abi.funcs):
        for ii, fld in enumerate(fm.input_fields):
            if fld.kind == SFA_INPUT_GLOBAL:
                global_input = (fi, ii)
                break
        if global_input != (0, 0) or fm.input_fields:
            pass
    if global_input == (0, 0) and functions:
        for ii, fld in enumerate(abi.funcs[0].input_fields):
            if fld.kind == SFA_INPUT_GLOBAL:
                global_input = (0, ii)
                break
    global_output = (len(functions) - 1, 0) if functions else (0, 0)
    return {
        "functions": functions,
        "global_input": global_input,
        "global_output": global_output,
    }
