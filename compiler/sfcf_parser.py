"""SFCF binary format parsing for compiled dylib artifacts.

Provides functions to parse the ServeForge Constants Format (SFCF) blob
embedded in compiled .dylib files, including:

- ``parse_sfcf_blob`` — name mappings, constants, compute graph position
- ``parse_compute_graph`` — function definitions with I/O bindings
- ``parse_sret_outputs`` — output tensors from sret buffer
- ``make_memref_descriptor`` — ctypes MemRef struct for numpy arrays
- ``verify_output_shapes`` — cross-check parsed tensors against graph declarations
"""

import ctypes
import struct
from typing import Any

import numpy as np

# =====================================================================
# Low-level binary readers (little-endian)
# =====================================================================

def _read_u8(data: bytes, pos: int) -> tuple[int, int]:
    return data[pos], pos + 1


def _read_u32(data: bytes, pos: int) -> tuple[int, int]:
    return struct.unpack_from('<I', data, pos)[0], pos + 4


def _read_u64(data: bytes, pos: int) -> tuple[int, int]:
    return struct.unpack_from('<Q', data, pos)[0], pos + 8


def _read_str(data: bytes, pos: int) -> tuple[str, int]:
    n, pos = _read_u32(data, pos)
    s = data[pos:pos + n].decode('utf-8')
    return s, pos + n


# =====================================================================
# Top-level SFCF blob parser
# =====================================================================

def parse_sfcf_blob(blob: bytes) -> tuple[dict[str, str], dict[str, np.ndarray[Any, Any]], int, int]:
    """Parse SFCF v2/v3 blob → (name_mapping, constants_dict, graph_start_pos, version)."""
    assert blob[:4] == b'SFCF', f"Bad magic: {blob[:4]}"  # type: ignore[str-bytes-safe]
    v, pos = _read_u32(blob, 4)
    assert 2 <= v <= 4, f"Unsupported SFCF version: {v}"

    # Name mappings  (compiled → hf_key)
    nm_count, pos = _read_u32(blob, pos)
    name_mapping: dict[str, str] = {}
    for _ in range(nm_count):
        compiled, pos = _read_str(blob, pos)
        hf_key, pos = _read_str(blob, pos)
        name_mapping[compiled] = hf_key

    # Constants (compiler-synthesized tensors)
    const_count, pos = _read_u32(blob, pos)
    constants: dict[str, np.ndarray[Any, Any]] = {}
    for _ in range(const_count):
        name, pos = _read_str(blob, pos)
        dtype_code = blob[pos]
        pos += 1
        ndim = blob[pos]
        pos += 1
        shape: list[int] = []
        for _ in range(ndim):
            d, pos = _read_u64(blob, pos)
            shape.append(int(d))
        data_len, pos = _read_u64(blob, pos)
        raw = blob[pos:pos + data_len]
        pos += data_len
        # dtype codes: 0=F32, 1=F16, 2=BF16, 3=I64, 4=I32, 5=I8, 6=U8
        dtype_map = {0: np.float32, 1: np.float16, 2: np.float16,
                     3: np.int64, 4: np.int32, 5: np.int8, 6: np.uint8}
        np_dtype = dtype_map.get(dtype_code, np.float32)
        # Handle rank-0 (scalar) constants: shape=[] → reshape to size 1
        np_shape = shape if len(shape) > 0 else [1]
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(np_shape)
        # Convert to f32 (ciface functions expect f32 weights)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if len(shape) == 0:
            arr = arr.squeeze()
        constants[name] = arr

    return name_mapping, constants, pos, v


def parse_contract_section(data: bytes, pos: int) -> dict[str, str]:
    """Parse the trailing contract section appended after the compute graph trailer.

    Contract format:
        contract_count: u32
        for each entry:
            key_len: u32
            key: UTF-8
            val_len: u32
            val: UTF-8

    Returns a dict of key-value pairs, or empty dict if pos >= len(data)
    (backward compatible: v2/v3 binaries have no contract section).
    """
    if pos >= len(data):
        return {}
    contract_count, pos = _read_u32(data, pos)
    contract: dict[str, str] = {}
    for _ in range(contract_count):
        key, pos = _read_str(data, pos)
        val, pos = _read_str(data, pos)
        contract[key] = val
    return contract


def parse_compute_graph(data: bytes, pos: int, version: int = 3) -> tuple[dict[str, Any], int]:
    """Parse compute graph → list of func dicts + global I/O indices.

    Args:
        data: Raw SFCF binary data.
        pos: Start position of compute graph section.
        version: SFCF format version. v3+ includes a consumed_internally
            flag byte before each output's rank byte. v2 defaults to False.
    """
    num_funcs, pos = _read_u32(data, pos)
    functions = []

    for _ in range(num_funcs):
        symbol, pos = _read_str(data, pos)
        num_inputs, pos = _read_u32(data, pos)
        num_outputs, pos = _read_u32(data, pos)

        inputs = []
        for _ in range(num_inputs):
            bt = data[pos]
            pos += 1
            if bt == 0:        # Weight
                key, pos = _read_str(data, pos)
                binding = ('weight', key)
            elif bt == 1:      # Ssa
                pf, pos = _read_u32(data, pos)
                oi, pos = _read_u32(data, pos)
                binding = ('ssa', pf, oi)  # type: ignore[assignment]
            elif bt == 2:      # GlobalInput
                binding = ('global_input',)  # type: ignore[assignment]
            else:
                raise ValueError(f"Unknown binding type {bt}")
            rank = data[pos]
            pos += 1
            ndims, pos = _read_u32(data, pos)
            shape = []
            for _ in range(ndims):
                d, pos = _read_u64(data, pos)
                shape.append(int(d))
            inputs.append({'binding': binding, 'rank': rank, 'shape': shape})

        outputs = []
        for _ in range(num_outputs):
            if version >= 3:
                consumed_internally = bool(data[pos])
                pos += 1
            else:
                consumed_internally = False
            rank = data[pos]
            pos += 1
            ndims, pos = _read_u32(data, pos)
            shape = []
            for _ in range(ndims):
                d, pos = _read_u64(data, pos)
                shape.append(int(d))
            outputs.append({
                'rank': rank,
                'shape': shape,
                'consumed_internally': consumed_internally,
            })

        functions.append({
            'symbol': symbol,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'inputs': inputs,
            'outputs': outputs,
        })

    gi_func, pos = _read_u32(data, pos)
    gi_arg, pos = _read_u32(data, pos)
    go_func, pos = _read_u32(data, pos)
    go_idx, pos = _read_u32(data, pos)

    return {
        'functions': functions,
        'global_input': (gi_func, gi_arg),
        'global_output': (go_func, go_idx),
    }, pos


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
        rank = od['rank']
        desc_size = 24 + 16 * rank
        desc = sret_bytes[offset:offset + desc_size]

        aligned = struct.unpack_from('<Q', desc, 8)[0]

        # Read runtime sizes from descriptor; fall back to static shape for 0 dims
        runtime_sizes = []
        for i in range(rank):
            s = struct.unpack_from('<q', desc, 24 + 8 * i)[0]
            # 0 means dynamic; use static shape as hint
            if s <= 0 or s > 1_000_000_000:
                s = od['shape'][i] if i < len(od['shape']) and od['shape'][i] > 0 else 1
            runtime_sizes.append(int(s))

        n = int(np.prod(runtime_sizes))
        if n > 0 and aligned != 0:
            buf = (ctypes.c_float * n).from_address(aligned)
            arr = np.array(buf, dtype=np.float32).reshape(runtime_sizes)
        else:
            arr = np.array([], dtype=np.float32)

        tensors.append(arr)
        offset += desc_size

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
            errors.append(
                f"func[{fi}]: no compute graph entry for this function index"
            )
            continue

        func_def = compute_graph_functions[fi]
        expected_outputs = func_def["outputs"]
        symbol = func_def.get("symbol", f"func_{fi}")

        # Check 1: output count
        if len(outputs) != len(expected_outputs):
            errors.append(
                f"func[{fi}] ({symbol}): "
                f"expected {len(expected_outputs)} output(s), "
                f"got {len(outputs)}"
            )
            # Cannot do per-output checks if counts differ
            continue

        for oi, (arr, expected) in enumerate(zip(outputs, expected_outputs, strict=True)):
            expected_rank = expected["rank"]
            expected_shape = expected["shape"]

            # Check 2: null data pointer
            # parse_sret_outputs returns empty array when aligned==0
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
            # Dynamic dims (0 in compute graph) are skipped
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
