#!/usr/bin/env python3
"""Call the compiled .dylib directly via ctypes, using Python executor weights.

Compares two paths side-by-side:
  1. Python MlirExecutor (sf dialect)  — cos ≈ 0.999994
  2. Python ctypes calling .dylib       — cos = ?
  3. Rust executor calling .dylib       — cos ≈ 0.869

Expected signal patterns:
  - If ctypes cos ≈ Python executor cos (0.999) → bug is in Rust runtime's
    ciface calling convention (weight ordering, memref layout, input shape)
  - If ctypes cos ≈ Rust runtime cos (0.869) → bug is in compiled dylib /
    lowering pipeline (bufferization, LLVM codegen, FP accumulation)
"""

import ctypes
import faulthandler
import os
import struct
import sys
import time

import numpy as np

faulthandler.enable()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================================
# Helpers
# =====================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return float(np.dot(a_f, b_f) / (denom + 1e-12))

# =====================================================================
# SFCF v2 binary format parsing
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

def parse_sfcf_blob(blob: bytes):
    """Parse SFCF v2 blob → (name_mapping, constants_dict, graph_start_pos)."""
    assert blob[:4] == b'SFCF', f"Bad magic: {blob[:4]}"
    v, pos = _read_u32(blob, 4)
    assert v == 2, f"Unsupported SFCF version: {v}"

    # Name mappings  (compiled → hf_key)
    nm_count, pos = _read_u32(blob, pos)
    name_mapping: dict[str, str] = {}
    for _ in range(nm_count):
        compiled, pos = _read_str(blob, pos)
        hf_key, pos = _read_str(blob, pos)
        name_mapping[compiled] = hf_key

    # Constants (compiler-synthesized tensors)
    const_count, pos = _read_u32(blob, pos)
    constants: dict[str, np.ndarray] = {}
    for _ in range(const_count):
        name, pos = _read_str(blob, pos)
        dtype_code = blob[pos]; pos += 1
        ndim = blob[pos]; pos += 1
        shape: list[int] = []
        for _ in range(ndim):
            d, pos = _read_u64(blob, pos)
            shape.append(int(d))
        data_len, pos = _read_u64(blob, pos)
        raw = blob[pos:pos + data_len]; pos += data_len
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

    return name_mapping, constants, pos


def parse_compute_graph(data: bytes, pos: int):
    """Parse compute graph → list of func dicts + global I/O indices."""
    num_funcs, pos = _read_u32(data, pos)
    functions = []

    for _ in range(num_funcs):
        symbol, pos = _read_str(data, pos)
        num_inputs, pos = _read_u32(data, pos)
        num_outputs, pos = _read_u32(data, pos)

        inputs = []
        for _ in range(num_inputs):
            bt = data[pos]; pos += 1
            if bt == 0:        # Weight
                key, pos = _read_str(data, pos)
                binding = ('weight', key)
            elif bt == 1:      # Ssa
                pf, pos = _read_u32(data, pos)
                oi, pos = _read_u32(data, pos)
                binding = ('ssa', pf, oi)
            elif bt == 2:      # GlobalInput
                binding = ('global_input',)
            else:
                raise ValueError(f"Unknown binding type {bt}")
            rank = data[pos]; pos += 1
            ndims, pos = _read_u32(data, pos)
            shape = []
            for _ in range(ndims):
                d, pos = _read_u64(data, pos)
                shape.append(int(d))
            inputs.append({'binding': binding, 'rank': rank, 'shape': shape})

        outputs = []
        for _ in range(num_outputs):
            rank = data[pos]; pos += 1
            ndims, pos = _read_u32(data, pos)
            shape = []
            for _ in range(ndims):
                d, pos = _read_u64(data, pos)
                shape.append(int(d))
            outputs.append({'rank': rank, 'shape': shape})

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
    }


# =====================================================================
# MemRef descriptor construction
# =====================================================================

def make_memref_descriptor(arr: np.ndarray):
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


def parse_sret_outputs(sret_bytes: bytes, output_defs: list[dict]) -> list[np.ndarray]:
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
# Main
# =====================================================================

def main():
    ARTIFACT_DIR = "./compiled/opt_125m_fresh"
    DYLIB_PATH = os.path.join(ARTIFACT_DIR, "libopt_125m.dylib")
    INPUT_IDS = np.array([[2, 32826, 85, 4129], [0, 0, 0, 0]], dtype=np.int64)

    # ── Step 1: Load Python executor reference weights ──────────────
    print("=" * 60)
    print("Step 1/5: Load artifact weights (Python executor path)")
    print("=" * 60)
    from compiler.serialize import load_artifact
    artifact = load_artifact(ARTIFACT_DIR)

    all_weights: dict[str, np.ndarray] = {}
    for func in artifact.functions:
        for wname, wtensor in func.weights.items():
            if wname not in all_weights:
                all_weights[wname] = np.ascontiguousarray(wtensor.numpy())
    print(f"  Loaded {len(all_weights)} unique weight tensors from artifact")

    # Check a few weights
    for key in ['model_decoder_embed_tokens_weight',
                'model_decoder_embed_positions_weight']:
        if key in all_weights:
            print(f"  {key}: shape={all_weights[key].shape}, "
                  f"dtype={all_weights[key].dtype}, "
                  f"first={all_weights[key].flat[0]:.6f}")

    # ── Step 2: Load reference Python logits ────────────────────────
    print()
    print("=" * 60)
    print("Step 2/5: Load reference Python executor logits")
    print("=" * 60)
    py_logits_path = '/tmp/py_logits_batch2.npy'
    if os.path.exists(py_logits_path):
        py_logits = np.load(py_logits_path)
    else:
        # Generate them on the fly
        from hal.pytorch_backend import PyTorchBackend
        from engine.mlir_executor import MlirExecutor
        import torch
        backend = PyTorchBackend('cpu')
        executor = MlirExecutor(artifact, backend)
        with torch.no_grad():
            logits = executor.forward(torch.tensor(INPUT_IDS))
        py_logits = logits.numpy()
        np.save(py_logits_path, py_logits)

    print(f"  Python executor logits: shape={py_logits.shape}, "
          f"first={py_logits[0, 0, 0]:.6f}, "
          f"last={py_logits[0, 0, -1]:.6f}")
    print(f"  Python logits mean={py_logits.mean():.6f}, std={py_logits.std():.6f}")

    # ── Step 3: Load dylib and parse SFCF blob ──────────────────────
    print()
    print("=" * 60)
    print("Step 3/5: Load dylib and parse embedded SFCF blob")
    print("=" * 60)
    lib = ctypes.CDLL(DYLIB_PATH)

    # Read embedded SFCF blob from dylib symbols
    data_ptr = ctypes.cast(
        ctypes.addressof(ctypes.c_int64.in_dll(lib, 'serveforge_constants_data')),
        ctypes.c_void_p,
    )
    size_ptr = ctypes.cast(
        ctypes.addressof(ctypes.c_int64.in_dll(lib, 'serveforge_constants_size')),
        ctypes.POINTER(ctypes.c_uint64),
    )
    blob_size = size_ptr[0]
    blob_bytes = bytes((ctypes.c_uint8 * blob_size).from_address(data_ptr.value))
    print(f"  SFCF blob: {blob_size} bytes")
    print(f"  Magic: {blob_bytes[:4]}")

    name_mapping, sfcf_constants, graph_pos = parse_sfcf_blob(blob_bytes)
    print(f"  Name mappings: {len(name_mapping)}")
    print(f"  SFCF constants: {len(sfcf_constants)}")
    for k in list(sfcf_constants.keys())[:5]:
        print(f"    constant {k}: shape={sfcf_constants[k].shape}")

    graph = parse_compute_graph(blob_bytes, graph_pos)
    print(f"  Compute graph: {len(graph['functions'])} functions")
    for fi, f in enumerate(graph['functions']):
        print(f"    [{fi:2d}] {f['symbol']}: "
              f"{f['num_inputs']} inputs → {f['num_outputs']} outputs")
    print(f"  Global output: func={graph['global_output'][0]}, "
          f"idx={graph['global_output'][1]}")

    # ── Step 4: Run forward via ctypes ──────────────────────────────
    print()
    print("=" * 60)
    print("Step 4/5: Run forward pass via ctypes")
    print("=" * 60)

    SRET_SIZE = 131072  # 128KB should be ample for 211 output descriptors
    func_outputs: list[list[np.ndarray]] = [[] for _ in range(len(graph['functions']))]

    # Build a combined weight lookup with multi-strategy key resolution.
    hf_key_map = artifact.metadata.get('hf_key_map', {})
    # Tied weights: some compiled names share weight data (e.g. lm_head_weight
    # shares with model_decoder_embed_tokens_weight in OPT).
    ws = artifact.metadata.get('weight_source', {})
    tied_weights = ws.get('tied_weights', {}) or artifact.metadata.get('tied_weights', {})

    def get_weight(name: str) -> np.ndarray:
        # Strategy 1: direct match
        if name in all_weights:
            return all_weights[name]

        # Strategy 2: translate via hf_key_map (compiled → hf key)
        hf_key = hf_key_map.get(name)
        if hf_key and hf_key in all_weights:
            return all_weights[hf_key]

        # Strategy 3: tied weights — if name is the primary, try the alias
        for alias_name, primary_name in tied_weights.items():
            if primary_name == name:
                # Alias maps to this name — try the alias's HF key
                alias_hf = hf_key_map.get(alias_name)
                if alias_hf and alias_hf in all_weights:
                    return all_weights[alias_hf]
                # Also try alias directly
                if alias_name in all_weights:
                    return all_weights[alias_name]

        # Strategy 4: try without function prefix (e.g. "main_0._const_7" → "_const_7")
        bare_name = name.split('.', 1)[-1] if '.' in name else name
        if bare_name != name:
            if bare_name in all_weights:
                return all_weights[bare_name]
            if bare_name in sfcf_constants:
                return np.ascontiguousarray(sfcf_constants[bare_name])

        # Strategy 5: try with main_0. prefix (for constants in artifact)
        prefixed = f"main_0.{name}"
        if prefixed in all_weights:
            return all_weights[prefixed]

        # Strategy 6: constant from SFCF blob
        if name in sfcf_constants:
            return np.ascontiguousarray(sfcf_constants[name])

        raise KeyError(f"Weight '{name}' not found")

    t0 = time.time()

    for fi, func_def in enumerate(graph['functions']):
        symbol = func_def['symbol']
        try:
            kernel = getattr(lib, symbol)
        except AttributeError:
            print(f"  [{fi:2d}] WARNING: {symbol} not found, skipping")
            continue

        # Build input descriptors
        input_descs = []
        input_args = []
        # CRITICAL: keep references to numpy arrays to prevent GC
        # from freeing the underlying buffer that descriptors point to.
        _keep_arrs: list[np.ndarray] = []

        for inp in func_def['inputs']:
            binding = inp['binding']
            io_shape = inp['shape']

            if binding[0] == 'global_input':
                arr = INPUT_IDS

            elif binding[0] == 'weight':
                key = binding[1]
                try:
                    arr = get_weight(key)
                except KeyError:
                    shape = [int(s) if s > 0 else 1 for s in io_shape] or [1]
                    arr = np.zeros(shape, dtype=np.float32)
                    print(f"    [{fi:2d}] WARNING: weight '{key}' not found, "
                          f"using zeros shape={shape}")

            elif binding[0] == 'ssa':
                pf, oi = binding[1], binding[2]
                if pf < len(func_outputs) and oi < len(func_outputs[pf]):
                    arr = func_outputs[pf][oi]
                else:
                    shape = [int(s) if s > 0 else 1 for s in io_shape] or [1]
                    arr = np.zeros(shape, dtype=np.float32)

            else:
                raise ValueError(f"Unknown binding: {binding}")

            # Ensure contiguous f32 for descriptors
            if arr.dtype != np.float32:
                if arr.dtype == np.int64:
                    pass  # input_ids stays as int64
                else:
                    arr = arr.astype(np.float32)
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)

            # KEEP ALIVE: prevent GC from freeing arr's buffer
            _keep_arrs.append(arr)

            desc = make_memref_descriptor(arr)
            input_descs.append(desc)
            input_args.append(ctypes.byref(desc))

        # Allocate sret buffer
        sret = (ctypes.c_uint8 * SRET_SIZE)()

        # Build arg list: sret + all input descriptors
        all_args = [ctypes.byref(sret)] + input_args

        # Set up calling convention
        n_args = len(all_args)
        kernel.argtypes = [ctypes.c_void_p] * n_args
        kernel.restype = None

        # Call the ciface function
        t1 = time.time()
        kernel(*all_args)
        call_time = time.time() - t1

        # Parse outputs
        outputs = parse_sret_outputs(bytes(sret), func_def['outputs'])
        func_outputs[fi] = outputs

        # Log function summary
        n_out = len(outputs)
        if fi == 0:
            print(f"  [{fi:2d}] {symbol}: {call_time:.3f}s, {n_out} outputs")
        elif n_out > 0:
            print(f"  [{fi:2d}] {symbol}: {call_time:.3f}s, out[0] shape={outputs[0].shape}")

    total_time = time.time() - t0

    # ── Step 5: Compare results ─────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 5/5: Results — ctypes vs Python executor")
    print("=" * 60)

    go_func, go_idx = graph['global_output']
    ctypes_logits = func_outputs[go_func][go_idx]
    print(f"  ctypes logits: shape={ctypes_logits.shape}")
    print(f"    first 8: {ctypes_logits.ravel()[:8].tolist()}")
    print(f"    last 8:  {ctypes_logits.ravel()[-8:].tolist()}")
    print(f"    mean={ctypes_logits.mean():.6f}, std={ctypes_logits.std():.6f}")

    # Compare with Python executor per-token
    print(f"\n  Per-token cosine (ctypes vs Python executor):")
    for b in range(min(ctypes_logits.shape[0], py_logits.shape[0])):
        for p in range(min(ctypes_logits.shape[1], py_logits.shape[1])):
            sim = cosine_similarity(ctypes_logits[b, p], py_logits[b, p])
            print(f"    batch={b}, pos={p}: cos={sim:.10f}")

    # Also compare with Rust runtime logits if available
    cos_ctypes_rust = -1.0
    rust_csv_path = '/tmp/rust_logits.csv'
    if os.path.exists(rust_csv_path):
        rust_csv = np.loadtxt(rust_csv_path, delimiter=',')
        rust_logits = rust_csv.reshape(ctypes_logits.shape)
        cos_ctypes_rust = cosine_similarity(ctypes_logits, rust_logits)
        cos_py_rust = cosine_similarity(py_logits, rust_logits)
        print(f"\n  ctypes vs Rust dylib: cos={cos_ctypes_rust:.10f}")
        print(f"  Python vs Rust dylib: cos={cos_py_rust:.10f}")

    # Overall cosine with Python executor
    cos = cosine_similarity(py_logits, ctypes_logits)
    print(f"\n  Overall cos(ctypes, Python executor): {cos:.10f}")
    print(f"  Total forward time: {total_time:.3f}s")

    # Interpretation
    print()
    print("-" * 60)
    print("DIAGNOSIS")
    print("-" * 60)
    if cos > 0.99:
        print("  ✓ ctypes ≈ Python executor (cos > 0.99)")
        print("  → Bug is in RUST RUNTIME's ciface calling convention")
    elif cos_ctypes_rust > 0.99:
        print(f"  ✓ ctypes ≈ Rust Runtime (cos={cos_ctypes_rust:.6f})")
        print("  → Bug is in COMPILED DYLIB / LOWERING PIPELINE")
    elif cos > 0.8:
        print("  ✗ ctypes differs from Python executor (check dylib pipeline)")
    else:
        print("  ✗ ctypes ≠ Python executor AND ctypes ≠ Rust runtime")
        print("  → Possible bug in ctypes calling convention")

    return cos


if __name__ == '__main__':
    main()
