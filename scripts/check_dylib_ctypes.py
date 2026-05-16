"""Bisect accuracy: call .dylib directly from Python ctypes.

Compares three paths side-by-side:
  1. Python MlirExecutor (sf dialect)  — cos ≈ 0.999999
  2. Python ctypes calling .dylib       — cos = ?
  3. Rust executor calling .dylib       — cos = 0.865

If (2) ≈ (3), the bug is in the compiled .dylib (compiler pipeline).
If (2) ≠ (3) and (2) ≈ (1), the bug is in the Rust executor's input construction.
"""

import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))


class CifaceBuffer(ctypes.Structure):
    """MLIR memref descriptor for rank-2 (2D)."""
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]


class CifaceBuffer1(ctypes.Structure):
    """MLIR memref descriptor for rank-1 (1D)."""
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 1),
        ("strides", ctypes.c_int64 * 1),
    ]


class CifaceBuffer4(ctypes.Structure):
    """MLIR memref descriptor for rank-4 (4D)."""
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 4),
        ("strides", ctypes.c_int64 * 4),
    ]


def make_memref_f32(data: np.ndarray) -> CifaceBuffer:
    """Build a rank-2 memref for an f32 numpy array."""
    assert data.dtype == np.float32
    assert data.ndim == 2
    rows, cols = data.shape
    p = data.ctypes.data_as(ctypes.c_void_p)
    return CifaceBuffer(
        allocated=p, aligned=p, offset=0,
        sizes=(ctypes.c_int64 * 2)(rows, cols),
        strides=(ctypes.c_int64 * 2)(cols, 1),
    )


def make_memref_i64(data: np.ndarray) -> CifaceBuffer:
    """Build a rank-2 memref for an i64 numpy array."""
    assert data.dtype == np.int64
    assert data.ndim == 2
    rows, cols = data.shape
    p = data.ctypes.data_as(ctypes.c_void_p)
    return CifaceBuffer(
        allocated=p, aligned=p, offset=0,
        sizes=(ctypes.c_int64 * 2)(rows, cols),
        strides=(ctypes.c_int64 * 2)(cols, 1),
    )


def make_sret_buffer(n: int = 131072) -> ctypes.Array[ctypes.c_uint8]:
    """Create a zero-initialized buffer for sret output."""
    return (ctypes.c_uint8 * n)()


def parse_sret_descriptor(sret: ctypes.Array, rank: int, offset: int) -> tuple[int, list[int]]:
    """Parse one output descriptor from sret buffer. Returns (aligned_ptr, sizes)."""
    desc_size = 24 + 16 * rank
    slice_bytes = bytes(sret[offset:offset + desc_size])
    aligned = int.from_bytes(slice_bytes[8:16], 'little')
    sizes = []
    for i in range(rank):
        s = int.from_bytes(slice_bytes[24 + 8*i:24 + 8*(i+1)], 'little')
        sizes.append(s)
    return aligned, sizes


def load_weight_from_safetensors(sf_path: str, hf_key: str) -> np.ndarray:
    """Load a weight from safetensors file as f32 numpy array."""
    import safetensors.torch
    sd = safetensors.torch.load_file(sf_path)
    t = sd[hf_key]
    return t.to(torch.float32).numpy()


def run_main_0(so: ctypes.CDLL, input_ids: np.ndarray,
               weights: dict[str, np.ndarray],
               name_map: dict[str, str]):
    """Call _mlir_ciface_main_0 with weight inputs constructed from loaded weights."""
    try:
        main_0 = so._mlir_ciface_main_0
    except AttributeError:
        raise RuntimeError("_mlir_ciface_main_0 not found in .dylib")

    sret = make_sret_buffer()

    # Build input args: sret + input memrefs
    args = [ctypes.byref(sret)]

    # Input 0: GlobalInput (i64)
    input_memref = make_memref_i64(input_ids)
    args.append(ctypes.byref(input_memref))

    # Now we need all 214 weight arguments. We need to know the exact order
    # and mapping of compiled names → HF keys.
    # From compute_graph, we know main_0 has inputs[1..214] as weights.
    # We need to order them correctly.
    # For now, just call the function to see if it crashes.

    print(f"Sret addr: {ctypes.addressof(sret)}")
    print(f"Input ids: {input_ids}")
    print(f"Input memref: allocated={input_memref.allocated}, sizes={list(input_memref.sizes)}")

    try:
        main_0(*args)
    except Exception as e:
        print(f"main_0 failed: {e}")
        return None

    # Parse sret for main_0 outputs
    sret_data = bytes(sret)
    print(f"Sret first 32 bytes: {sret_data[:32].hex()}")
    return sret


def main():
    import torch  # delayed import for monkey-patching

    artifact_dir = os.path.abspath("./compiled/opt_125m_v8")
    dylib_path = os.path.join(artifact_dir, "libopt_125m.dylib")
    baseline_dir = os.path.join(artifact_dir, "baselines")

    # Load baselines
    hf_logits = np.load(os.path.join(baseline_dir, "hf_logits.npy"))
    py_logits = np.load(os.path.join(baseline_dir, "python_executor_logits.npy"))

    # Load the dylib
    so = ctypes.CDLL(dylib_path)

    # Find safetensors
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--opt-125m")
    snapshots = os.path.join(hub_dir, "snapshots")
    snap = os.listdir(snapshots)[0]
    st_path = os.path.join(snapshots, snap, "pytorch_model.bin")
    print(f"Safetensors: {st_path}")

    # Load metadata for name mapping
    import json
    meta_path = os.path.join(artifact_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    hf_key_map = meta.get("hf_key_map", {})
    print(f"Name mappings: {len(hf_key_map)}")

    # Try to call main_0 with just the sret and global input to check ciface
    input_ids = np.array([[2, 32826, 85, 4129], [0, 0, 0, 0]], dtype=np.int64)

    # Actually, we need the weight inputs too. Let's read the compute graph
    # from constants.bin to know the input ordering.
    # For now, just check that the dylib loads and symbols are found.
    print(f"\nDylib loaded: {dylib_path}")
    print(f"Symbols in dylib:")
    symbols = []
    try:
        # macOS: use nm to list symbols
        import subprocess
        result = subprocess.run(["nm", "-g", dylib_path],
                               capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "_mlir_ciface" in line:
                symbols.append(line.strip())
                print(f"  {line.strip()}")
    except Exception:
        pass

    # Check if we can find and read constants data
    nm_result = subprocess.run(["nm", "-g", dylib_path],
                               capture_output=True, text=True)
    const_data_line = [l for l in nm_result.stdout.splitlines()
                       if "serveforge_constants_data" in l]
    const_size_line = [l for l in nm_result.stdout.splitlines()
                       if "serveforge_constants_size" in l]
    if const_data_line:
        print(f"\nConstants data symbol: {const_data_line[0]}")
    if const_size_line:
        print(f"\nConstants size symbol: {const_size_line[0]}")

    # Let's also check what the Rust executor's output looks like by reading
    # the saved Rust logits CSV
    rust_csv = np.loadtxt("/tmp/rust_logits.csv", delimiter=",")
    rust_logits = rust_csv.reshape(hf_logits.shape)
    print(f"\nHF logits first 8: {hf_logits[0, 0, :8]}")
    print(f"Python logits first 8: {py_logits[0, 0, :8]}")
    print(f"Rust logits first 8: {rust_logits[0, 0, :8]}")

    # Check if the Rust logits have the right scale by looking at the last few
    print(f"\nHF logits last 8: {hf_logits[0, -1, -8:]}")
    print(f"Python logits last 8: {py_logits[0, -1, -8:]}")
    print(f"Rust logits last 8: {rust_logits[0, -1, -8:]}")

    # Compute per-position cosine to see where the divergence happens
    print("\n=== Per-token cosine (Rust vs HF) ===")
    for tok in range(4):
        sim = cosine_similarity(rust_logits[0, tok], hf_logits[0, tok])
        print(f"  token[{tok}]: cos={sim:.8f}")

    # Also check if the Rust first row vs second row differ
    sim_rr = cosine_similarity(rust_logits[0], rust_logits[1])
    print(f"\n  Rust row0 vs row1: cos={sim_rr:.8f}")

    # Compare first-layer embedding weights
    print("\n=== Weight value check ===")
    state_dict = torch.load(st_path, map_location="cpu", weights_only=False)
    for key in ["model.decoder.embed_tokens.weight",
                "model.decoder.embed_positions.weight",
                "model.decoder.layers.0.self_attn.k_proj.weight"]:
        compiled_key = key.replace(".", "_")
        hf_safe = state_dict[key].to(torch.float32).numpy()
        print(f"\n  HF weight '{key}': shape={hf_safe.shape}, first={hf_safe.flat[0]:.6f}, last={hf_safe.flat[-1]:.6f}")

    print(f"\nExpected Rust logits: batch={hf_logits.shape[0]}, "
          f"seq={hf_logits.shape[1]}, vocab={hf_logits.shape[2]}")


if __name__ == "__main__":
    import torch
    main()
