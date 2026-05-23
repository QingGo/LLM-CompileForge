#!/usr/bin/env python3
"""CLI entry point for running compiled .dylib via ctypes and comparing with Python executor.

Usage:
    python scripts/ctypes_main.py

Compares ctypes output against the Python MlirExecutor reference and optionally
against the Rust runtime logits (if ``/tmp/rust_logits.csv`` exists).
"""

import ctypes
import os
import sys
import time

import numpy as np

from compiler.sfcf_parser import (
    make_memref_descriptor,
    parse_compute_graph,
    parse_sfcf_blob,
    parse_sret_outputs,
)
from scripts._cos import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    artifact_dir = "./compiled/opt_125m_fresh"
    dylib_path = os.path.join(artifact_dir, "libopt_125m.dylib")
    input_ids = np.array([[2, 32826, 85, 4129], [0, 0, 0, 0]], dtype=np.int64)

    # ── Step 1: Load Python executor reference weights ──────────────
    print("=" * 60)
    print("Step 1/5: Load artifact weights (Python executor path)")
    print("=" * 60)
    from compiler.serialize import load_artifact
    artifact = load_artifact(artifact_dir)

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
        import torch

        from engine.mlir_executor import MlirExecutor
        from hal.pytorch_backend import PyTorchBackend
        backend = PyTorchBackend('cpu')
        executor = MlirExecutor(artifact, backend)
        with torch.no_grad():
            logits = executor.forward(torch.tensor(input_ids))
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
    lib = ctypes.CDLL(dylib_path)

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

    name_mapping, sfcf_constants, graph_pos, sfcf_version = parse_sfcf_blob(blob_bytes)
    print(f"  Name mappings: {len(name_mapping)}")
    print(f"  SFCF constants: {len(sfcf_constants)}")
    print(f"  SFCF version: {sfcf_version}")
    for k in list(sfcf_constants.keys())[:5]:
        print(f"    constant {k}: shape={sfcf_constants[k].shape}")

    graph = parse_compute_graph(blob_bytes, graph_pos, version=sfcf_version)
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

    sret_size = 131072  # 128KB should be ample for 211 output descriptors
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
                arr = input_ids

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
        sret = (ctypes.c_uint8 * sret_size)()

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

        # Dump per-function output for layer-by-layer comparison
        if outputs:
            if fi > 0:
                np.save(f'/tmp/dylib_func_{fi}.npy', np.array(outputs[0]))
            # Also dump key func[0] outputs for debugging
            if fi == 0:
                for key_oi in [10, 12, 13]:
                    if key_oi < len(outputs):
                        np.save(f'/tmp/dylib_f0_out{key_oi}.npy', np.array(outputs[key_oi]))

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
    print("\n  Per-token cosine (ctypes vs Python executor):")
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
