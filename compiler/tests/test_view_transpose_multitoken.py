"""TDD isolation tests for sf.transpose and sf.view with dynamic shapes.

Goal: reproduce the dynamic shape bug where sf.transpose and sf.view produce
wrong runtime descriptor sizes for seq>1 in 4D tensors.

Known bug: for seq=2 input, the output MemRef descriptor reports seq=1
instead of seq=2, and strides collapse (dim1 stride = dim2 stride = 64).

Baseline: seq=1 tests pass with cos=1.0.  seq=2 tests should FAIL
(demonstrating the bug).

All tests use dynamic shapes (?x?x?x?xf32) with runtime sizes passed via
MemRef descriptors.  No HF model dependency — uses numpy arrays as input
and compares against numpy reference computations.
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Constants ────────────────────────────────────────────────────────

N_HEADS: int = 12
HEAD_DIM: int = 64
RANDOM_SEED: int = 42

# ── Tool finding ──────────────────────────────────────────────────────


def _find_tool(name: str) -> str:
    """Locate a binary needed for compilation (mlir-translate, cc)."""
    candidates = [name]
    if name in ("cc", "clang"):
        candidates.insert(0, "/usr/local/opt/llvm/bin/clang")
    candidates.append(str(ROOT / "llvm-project" / "build" / "bin" / name))
    for p in candidates:
        if Path(p).is_file():
            return str(p)
        try:
            if subprocess.run([p, "--version"], capture_output=True, timeout=5).returncode == 0:
                return p
        except FileNotFoundError:
            continue
    raise RuntimeError(f"{name} not found")


# ── Compilation pipeline ──────────────────────────────────────────────


def _compile(sf_mlir: str, tmp_dir: str, name: str) -> str:
    """Compile sf MLIR -> lowered -> LLVM -> cc -> dylib.

    Pipeline: sf->linalg (C++ pass) -> linalg->LLVM (Python) ->
              fixup unrealized casts -> mlir-translate -> cc -> link.
    """
    import mlir.ir as ir
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf  # noqa: F401

    from compiler.backend.compile_utils import _compile_serveforge_free, link_dylib
    from compiler.backend.fixups import _fixup_unrealized_casts_pass
    from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir
    from compiler.pipeline import _apply_sf_to_linalg

    lowered = _apply_sf_to_linalg(sf_mlir)
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)
    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(lowered, ctx)
        lower_linalg_to_llvm_ir(mod)
        _fixup_unrealized_casts_pass(mod)
        m = os.path.join(tmp_dir, "m.mlir")
        ll = os.path.join(tmp_dir, "m.ll")
        o = os.path.join(tmp_dir, "m.o")
        d = os.path.join(tmp_dir, f"{name}.dylib")
        with open(m, "w") as f:
            f.write(str(mod))
        subprocess.run(
            [_find_tool("mlir-translate"), "--mlir-to-llvmir", m, "-o", ll],
            capture_output=True, text=True, check=True, timeout=60,
        )
        subprocess.run(
            [_find_tool("cc"), "-c", ll, "-o", o, "-O0"],
            capture_output=True, text=True, check=True, timeout=60,
        )
        free_o = _compile_serveforge_free(tmp_dir)
        link_dylib([o, free_o], d)
        return d


# ── MemRef / SRet helpers ────────────────────────────────────────────


def _sret(size: int) -> ctypes.Array:
    """Allocate a ctypes byte buffer for sret output."""
    return (ctypes.c_uint8 * size)()


def _memref(arr: np.ndarray):
    """Build ctypes MemRef descriptor from numpy array.

    Strides are in element units (bytes / itemsize).
    """
    ndim = arr.ndim
    elem_strides = tuple(s // arr.itemsize for s in arr.strides)

    class M(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * ndim),
            ("strides", ctypes.c_int64 * ndim),
        ]
    return M(
        ctypes.c_void_p(arr.ctypes.data),
        ctypes.c_void_p(arr.ctypes.data),
        0,
        (ctypes.c_int64 * ndim)(*arr.shape),
        (ctypes.c_int64 * ndim)(*elem_strides),
    )


def _make_scalar_f32_array(value: float) -> np.ndarray:
    """Create a tensor<1xf32> array (rank-1, size-1) for dyn_shape operands."""
    return np.array([value], dtype=np.float32)


def _parse_sret_descriptor(
    sret_bytes: bytes, rank: int, offset: int = 0
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    """Parse a single MemRef descriptor from sret buffer.

    Returns (aligned_ptr, data_offset, sizes_tuple, strides_tuple).

    Descriptor layout:
        offset+0:  allocated (i64)
        offset+8:  aligned (i64)  → pointer to output data
        offset+16: sret_offset (i64)
        offset+24: sizes (i64 * rank)
        offset+24+8*rank: strides (i64 * rank)
    """
    aligned = struct.unpack_from("<Q", sret_bytes, offset + 8)[0]
    data_offset = struct.unpack_from("<q", sret_bytes, offset + 16)[0]
    sizes = tuple(
        struct.unpack_from("<q", sret_bytes, offset + 24 + 8 * i)[0]
        for i in range(rank)
    )
    strides = tuple(
        struct.unpack_from("<q", sret_bytes, offset + 24 + 8 * rank + 8 * i)[0]
        for i in range(rank)
    )
    return aligned, data_offset, sizes, strides


def _read_f32_data(sret_bytes: bytes, rank: int, offset: int = 0) -> np.ndarray:
    """Read f32 tensor data from a single sret descriptor.

    Uses the descriptor's reported sizes to interpret the data buffer.
    Returns (empty array) if aligned pointer is null.
    """
    aligned, _, sizes, _ = _parse_sret_descriptor(sret_bytes, rank, offset)
    # Treat negative/zero sizes as 1 (workaround for buggy descriptors).
    corrected = tuple(s if s > 0 else 1 for s in sizes)
    n = int(np.prod(corrected))
    if n > 0 and aligned != 0:
        buf = (ctypes.c_float * n).from_address(aligned)
        return np.array(buf, dtype=np.float32).reshape(corrected)
    return np.array([], dtype=np.float32)


def _read_f32_data_with_shape(
    sret_bytes: bytes, rank: int, shape: tuple[int, ...], offset: int = 0
) -> np.ndarray:
    """Read f32 tensor data using a forced shape (ignoring descriptor sizes).

    Use when descriptor reports wrong sizes for dynamic dims.
    """
    aligned, _, _, _ = _parse_sret_descriptor(sret_bytes, rank, offset)
    n = int(np.prod(shape))
    if n > 0 and aligned != 0:
        buf = (ctypes.c_float * n).from_address(aligned)
        return np.array(buf, dtype=np.float32).reshape(shape)
    return np.array([], dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two arrays."""
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    denom = np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12
    return float(np.dot(af, bf) / denom)


# ── Test 1: sf.transpose with 2 tokens (the bug case) ────────────────


TRANSPOSE_MLIR_DYNAMIC = """module {{
  func.func @main_0(%input: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {{
    %0 = "sf.transpose"(%input) {{dim0 = 1 : i64, dim1 = 2 : i64}}
         : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    func.return %0 : tensor<?x?x?x?xf32>
  }}
}}"""


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestTranspose4D:
    """Isolation tests for sf.transpose on dynamic 4D tensors."""

    def test_transpose_4d_2token(self):
        """sf.transpose on [1,12,2,64] -> [1,2,12,64] with dynamic shapes.

        The bug: for seq=2, output descriptor reports seq=1 instead of 2.
        Strides also collapse (dim1 stride = dim2 stride = 64).

        This test should FAIL, proving the dynamic shape bug.
        Expected output: cos=1.0 (same data, just reordered).
        """
        batch, heads, seq, dim = 1, N_HEADS, 2, HEAD_DIM
        rng = np.random.RandomState(RANDOM_SEED)
        arr = rng.randn(batch, heads, seq, dim).astype(np.float32)
        # Expected: numpy transpose swapping axes 1 and 2
        expected = np.transpose(arr, (0, 2, 1, 3))
        expected_shape = (batch, seq, heads, dim)

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(TRANSPOSE_MLIR_DYNAMIC.format(), td, "transpose_4d_2tok")
            lib = ctypes.CDLL(dylib)

            sret_buf = _sret(4096)
            input_mem = _memref(arr)

            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 2
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(
                ctypes.byref(sret_buf),
                ctypes.byref(input_mem),
            )

            # ── Diagnose: check descriptor sizes ──
            _, _, sizes, strides = _parse_sret_descriptor(bytes(sret_buf), 4)
            print(f"\ntranspose 2tok — sret sizes: {sizes}")
            print(f"transpose 2tok — sret strides: {strides}")
            print(f"transpose 2tok — expected shape: {expected_shape}")

            # Assert shapes match (THIS IS THE BUG CHECK)
            assert sizes == expected_shape, (
                f"BUG CONFIRMED: sf.transpose output descriptor reports sizes {sizes}, "
                f"expected {expected_shape}. "
                f"The dynamic shape handling is WRONG for seq>1. "
                f"Strides: {strides}"
            )

            # Also verify that the underlying data is correct (using forced shape)
            actual = _read_f32_data_with_shape(bytes(sret_buf), 4, expected_shape)

            cos_val = _cos(actual, expected)
            print(f"transpose 2tok — cos (forced shape): {cos_val:.10f}")

            # If descriptor sizes were wrong but data is correct, that's still a bug
            # because downstream ops read the descriptor sizes.
            assert cos_val > 0.999, (
                f"transpose seq=2 cos={cos_val:.10f} < 0.999 — data corruption "
                f"beyond descriptor. actual[0,0,0,:4]={actual[0,0,0,:4].tolist()}, "
                f"expected[0,0,0,:4]={expected[0,0,0,:4].tolist()}"
            )

    def test_transpose_4d_1token(self):
        """sf.transpose on [1,12,1,64] -> [1,1,12,64] (baseline).

        Same op, same dynamic shape MLIR, but seq=1.
        This should PASS — the bug only manifests at seq>1.
        """
        batch, heads, seq, dim = 1, N_HEADS, 1, HEAD_DIM
        rng = np.random.RandomState(RANDOM_SEED)
        arr = rng.randn(batch, heads, seq, dim).astype(np.float32)
        expected = np.transpose(arr, (0, 2, 1, 3))
        expected_shape = (batch, seq, heads, dim)

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(TRANSPOSE_MLIR_DYNAMIC.format(), td, "transpose_4d_1tok")
            lib = ctypes.CDLL(dylib)

            sret_buf = _sret(4096)
            input_mem = _memref(arr)

            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 2
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(
                ctypes.byref(sret_buf),
                ctypes.byref(input_mem),
            )

            _, _, sizes, strides = _parse_sret_descriptor(bytes(sret_buf), 4)
            print(f"\ntranspose 1tok — sret sizes: {sizes}")
            print(f"transpose 1tok — sret strides: {strides}")

            # Baseline: should pass
            assert sizes == expected_shape, (
                f"UNEXPECTED FAILURE: seq=1 baseline also broken. "
                f"sizes={sizes}, expected={expected_shape}, strides={strides}"
            )

            actual = _read_f32_data_with_shape(bytes(sret_buf), 4, expected_shape)
            cos_val = _cos(actual, expected)
            print(f"transpose 1tok — cos: {cos_val:.10f}")
            assert cos_val > 0.999, (
                f"transpose seq=1 cos={cos_val:.10f} < 0.999"
            )


# ── Test 2: sf.view reshape (rank-4 to rank-3) with 2 tokens ───────


# Dynamic shape rank-changing view from 4D to 3D.
# Uses explicit sentinels (-2, -3, -4) for all dynamic dims to avoid the
# -1 inference bug in the C++ lowering where -1 with unconsumed operands
# reuses a dyn_shape operand instead of falling through to inference.
VIEW_RESHAPE_MLIR_DYNAMIC = """module {{
  func.func @main_0(%input: tensor<?x?x?x?xf32>,
                     %batch: tensor<1xf32>, %seq: tensor<1xf32>,
                     %hidden: tensor<1xf32>)
      -> tensor<?x?x?xf32> {{
    %0 = "sf.view"(%input, %batch, %seq, %hidden) {{shape = [-2, -3, -4]}}
         : (tensor<?x?x?x?xf32>, tensor<1xf32>, tensor<1xf32>,
            tensor<1xf32>)
         -> tensor<?x?x?xf32>
    func.return %0 : tensor<?x?x?xf32>
  }}
}}"""


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestViewReshape:
    """Isolation tests for sf.view rank-changing reshape with dynamic shapes."""

    def test_view_reshape_4d_2token(self):
        """sf.view from [1,2,12,64] to [1,2,768] with dynamic shapes.

        Uses shape=[-2, -3, -4] with explicit sentinels for all dims
        (avoids -1 inference bug in C++ lowering).

        The bug: for seq=2, output descriptor reports seq=1 instead of 2.
        """
        batch, seq, heads, dim = 1, 2, N_HEADS, HEAD_DIM
        rng = np.random.RandomState(RANDOM_SEED + 1)
        arr = rng.randn(batch, seq, heads, dim).astype(np.float32)
        # Expected: reshape from [1, 2, 12, 64] to [1, 2, 768]
        expected = arr.reshape(batch, seq, heads * dim)
        expected_shape = (batch, seq, heads * dim)

        batch_val = _make_scalar_f32_array(float(batch))
        seq_val = _make_scalar_f32_array(float(seq))
        hidden_val = _make_scalar_f32_array(float(heads * dim))

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(VIEW_RESHAPE_MLIR_DYNAMIC.format(), td, "view_4d_2tok")
            lib = ctypes.CDLL(dylib)

            sret_buf = _sret(4096)
            input_mem = _memref(arr)
            batch_mem = _memref(batch_val)
            seq_mem = _memref(seq_val)
            hidden_mem = _memref(hidden_val)

            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 5
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(
                ctypes.byref(sret_buf),
                ctypes.byref(input_mem),
                ctypes.byref(batch_mem),
                ctypes.byref(seq_mem),
                ctypes.byref(hidden_mem),
            )

            # ── Diagnose: check descriptor sizes ──
            _, _, sizes, strides = _parse_sret_descriptor(bytes(sret_buf), 3)
            print(f"\nview 2tok — sret sizes: {sizes}")
            print(f"view 2tok — sret strides: {strides}")
            print(f"view 2tok — expected shape: {expected_shape}")

            # Assert shapes match (THIS IS THE BUG CHECK)
            assert sizes == expected_shape, (
                f"BUG CONFIRMED: sf.view output descriptor reports sizes {sizes}, "
                f"expected {expected_shape}. "
                f"Dynamic shape handling is WRONG for seq>1. "
                f"Strides: {strides}"
            )

            actual = _read_f32_data_with_shape(bytes(sret_buf), 3, expected_shape)
            cos_val = _cos(actual, expected)
            print(f"view 2tok — cos (forced shape): {cos_val:.10f}")
            assert cos_val > 0.999, (
                f"view seq=2 cos={cos_val:.10f} < 0.999"
            )

    def test_view_reshape_4d_1token(self):
        """sf.view from [1,1,12,64] to [1,1,768] with dynamic shapes (baseline).

        Same op, seq=1 — should PASS.
        """
        batch, seq, heads, dim = 1, 1, N_HEADS, HEAD_DIM
        rng = np.random.RandomState(RANDOM_SEED + 1)
        arr = rng.randn(batch, seq, heads, dim).astype(np.float32)
        expected = arr.reshape(batch, seq, heads * dim)
        expected_shape = (batch, seq, heads * dim)

        batch_val = _make_scalar_f32_array(float(batch))
        seq_val = _make_scalar_f32_array(float(seq))
        hidden_val = _make_scalar_f32_array(float(heads * dim))

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(VIEW_RESHAPE_MLIR_DYNAMIC.format(), td, "view_4d_1tok")
            lib = ctypes.CDLL(dylib)

            sret_buf = _sret(4096)
            input_mem = _memref(arr)
            batch_mem = _memref(batch_val)
            seq_mem = _memref(seq_val)
            hidden_mem = _memref(hidden_val)

            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 5
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(
                ctypes.byref(sret_buf),
                ctypes.byref(input_mem),
                ctypes.byref(batch_mem),
                ctypes.byref(seq_mem),
                ctypes.byref(hidden_mem),
            )

            _, _, sizes, strides = _parse_sret_descriptor(bytes(sret_buf), 3)
            print(f"\nview 1tok — sret sizes: {sizes}")
            print(f"view 1tok — sret strides: {strides}")

            assert sizes == expected_shape, (
                f"UNEXPECTED: seq=1 baseline broken. "
                f"sizes={sizes}, expected={expected_shape}, strides={strides}"
            )

            actual = _read_f32_data_with_shape(bytes(sret_buf), 3, expected_shape)
            cos_val = _cos(actual, expected)
            print(f"view 1tok — cos: {cos_val:.10f}")
            assert cos_val > 0.999


# ── Test 3: Composed transpose + view chain ──────────────────────────


TRANSPOSE_VIEW_MLIR_DYNAMIC = """module {{
  func.func @main_0(%input: tensor<?x?x?x?xf32>,
                     %batch: tensor<1xf32>, %seq: tensor<1xf32>,
                     %hidden: tensor<1xf32>)
      -> tensor<?x?x?xf32> {{
    %t = "sf.transpose"(%input) {{dim0 = 1 : i64, dim1 = 2 : i64}}
         : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    %v = "sf.view"(%t, %batch, %seq, %hidden) {{shape = [-2, -3, -4]}}
         : (tensor<?x?x?x?xf32>, tensor<1xf32>, tensor<1xf32>,
            tensor<1xf32>)
         -> tensor<?x?x?xf32>
    func.return %v : tensor<?x?x?xf32>
  }}
}}"""


# Exact model pattern: transpose(tensor<?x?x12x64> -> tensor<?x12x?x64>)
# followed by view(shape=[-2, -3, -1]) using 2 dyn_shape operands.
# This is the EXACT pattern from model.mlir line 283 for layer_0 SDPA output reshape.
# The -1 sentinel is supposed to be inferred from total element count.
TRANSPOSE_VIEW_MODEL_MLIR = """module {{
  func.func @main_0(%input: tensor<?x?x12x64xf32>,
                     %batch: tensor<1xf32>, %seq: tensor<1xf32>)
      -> tensor<?x?x?xf32> {{
    %t = "sf.transpose"(%input) {{dim0 = 1 : i64, dim1 = 2 : i64}}
         : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>
    %v = "sf.view"(%t, %batch, %seq) {{shape = [-2, -3, -1]}}
         : (tensor<?x12x?x64xf32>, tensor<1xf32>, tensor<1xf32>)
         -> tensor<?x?x?xf32>
    func.return %v : tensor<?x?x?xf32>
  }}
}}"""


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestTransposeThenView:
    """Verify the full transpose+view chain (typical attention reshape pattern)."""

    def test_transpose_then_view_2token(self):
        """transpose([1,12,2,64] -> [1,2,12,64]) then view -> [1,2,768].

        This is the exact pattern used in attention Q/K/V reshaping.
        The bug: descriptor sizes are wrong after these ops.
        """
        batch, heads, seq, dim = 1, N_HEADS, 2, HEAD_DIM
        rng = np.random.RandomState(RANDOM_SEED + 2)
        arr = rng.randn(batch, heads, seq, dim).astype(np.float32)
        # Expected: transpose -> view
        expected = np.transpose(arr, (0, 2, 1, 3)).reshape(batch, seq, heads * dim)
        expected_shape = (batch, seq, heads * dim)

        batch_val = _make_scalar_f32_array(float(batch))
        seq_val = _make_scalar_f32_array(float(seq))
        hidden_val = _make_scalar_f32_array(float(heads * dim))

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(TRANSPOSE_VIEW_MLIR_DYNAMIC.format(), td, "tpose_view_2tok")
            lib = ctypes.CDLL(dylib)

            sret_buf = _sret(4096)
            input_mem = _memref(arr)
            batch_mem = _memref(batch_val)
            seq_mem = _memref(seq_val)
            hidden_mem = _memref(hidden_val)

            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 5
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(
                ctypes.byref(sret_buf),
                ctypes.byref(input_mem),
                ctypes.byref(batch_mem),
                ctypes.byref(seq_mem),
                ctypes.byref(hidden_mem),
            )

            _, _, sizes, strides = _parse_sret_descriptor(bytes(sret_buf), 3)
            print(f"\ntranspose+view 2tok — sret sizes: {sizes}")
            print(f"transpose+view 2tok — sret strides: {strides}")
            print(f"transpose+view 2tok — expected shape: {expected_shape}")

            assert sizes == expected_shape, (
                f"BUG CONFIRMED: composed transpose+view output descriptor "
                f"reports sizes {sizes}, expected {expected_shape}. "
                f"Strides: {strides}"
            )

            actual = _read_f32_data_with_shape(bytes(sret_buf), 3, expected_shape)
            cos_val = _cos(actual, expected)
            print(f"transpose+view 2tok — cos (forced shape): {cos_val:.10f}")
            assert cos_val > 0.999, (
                f"transpose+view seq=2 cos={cos_val:.10f} < 0.999"
            )

    def test_transpose_then_view_1token(self):
        """transpose([1,12,1,64]) then view -> [1,1,768] (baseline, seq=1).

        Should PASS — bug only manifests at seq>1.
        """
        batch, heads, seq, dim = 1, N_HEADS, 1, HEAD_DIM
        rng = np.random.RandomState(RANDOM_SEED + 2)
        arr = rng.randn(batch, heads, seq, dim).astype(np.float32)
        expected = np.transpose(arr, (0, 2, 1, 3)).reshape(batch, seq, heads * dim)
        expected_shape = (batch, seq, heads * dim)

        batch_val = _make_scalar_f32_array(float(batch))
        seq_val = _make_scalar_f32_array(float(seq))
        hidden_val = _make_scalar_f32_array(float(heads * dim))

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(TRANSPOSE_VIEW_MLIR_DYNAMIC.format(), td, "tpose_view_1tok")
            lib = ctypes.CDLL(dylib)

            sret_buf = _sret(4096)
            input_mem = _memref(arr)
            batch_mem = _memref(batch_val)
            seq_mem = _memref(seq_val)
            hidden_mem = _memref(hidden_val)

            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 5
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(
                ctypes.byref(sret_buf),
                ctypes.byref(input_mem),
                ctypes.byref(batch_mem),
                ctypes.byref(seq_mem),
                ctypes.byref(hidden_mem),
            )

            _, _, sizes, strides = _parse_sret_descriptor(bytes(sret_buf), 3)
            print(f"\ntranspose+view 1tok — sret sizes: {sizes}")

            assert sizes == expected_shape, (
                f"UNEXPECTED: seq=1 baseline broken. "
                f"sizes={sizes}, expected={expected_shape}, strides={strides}"
            )

            actual = _read_f32_data_with_shape(bytes(sret_buf), 3, expected_shape)
            cos_val = _cos(actual, expected)
            print(f"transpose+view 1tok — cos: {cos_val:.10f}")
            assert cos_val > 0.999

    def test_model_pattern_transpose_view_2token(self):
        """Exact model pattern: sf.transpose + sf.view with shape=[-2,-3,-1].

        Uses the same tensor types and shape attr as model.mlir line 283.
        Input <?x?x12x64> -> transpose -> <?x12x?x64> -> view [-2,-3,-1] -> <?x?x?>

        This test exposes whether the -1 sentinel correctly resolves to 768
        when mixed with explicit -2/-3 sentinels using 2 dyn_shape operands.
        """
        batch, heads, seq, dim = 1, N_HEADS, 2, HEAD_DIM
        rng = np.random.RandomState(RANDOM_SEED + 3)
        # Input is <?x?x12x64>: [batch=1, seq=2, heads=12, dim=64]
        # Type <?x?x12x64xf32>: static 12 at dim 2 → heads, static 64 at dim 3 → dim,
        # dynamic dim 1 → seq. Shape must match: [batch, seq, heads, dim].
        arr = rng.randn(batch, seq, heads, dim).astype(np.float32)
        # transpose(dim0=1, dim1=2) → [1, 12, 2, 64]
        # view(shape=[-2, -3, -1]) → [1, 2, 768]
        expected = np.transpose(arr, (0, 2, 1, 3)).reshape(batch, seq, heads * dim)
        expected_shape = (batch, seq, heads * dim)

        batch_val = _make_scalar_f32_array(float(batch))
        seq_val = _make_scalar_f32_array(float(seq))

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(
                TRANSPOSE_VIEW_MODEL_MLIR.format(), td, "model_tv_2tok"
            )
            lib = ctypes.CDLL(dylib)

            sret_buf = _sret(4096)
            input_mem = _memref(arr)
            batch_mem = _memref(batch_val)
            seq_mem = _memref(seq_val)

            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 4
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(
                ctypes.byref(sret_buf),
                ctypes.byref(input_mem),
                ctypes.byref(batch_mem),
                ctypes.byref(seq_mem),
            )

            _, _, sizes, strides = _parse_sret_descriptor(bytes(sret_buf), 3)
            print(f"\nmodel-pattern 2tok — sret sizes: {sizes}")
            print(f"model-pattern 2tok — sret strides: {strides}")
            print(f"model-pattern 2tok — expected shape: {expected_shape}")

            # If this fails, the -1 sentinel is reusing a dyn_shape operand
            # instead of being inferred. dim 2 would be batch=1 instead of 768.
            assert sizes == expected_shape, (
                f"BUG: model-pattern view [-2,-3,-1] with 2 operands "
                f"produced sizes {sizes}, expected {expected_shape}. "
                f"The -1 sentinel likely consumed operand[0] (batch={batch}) "
                f"instead of inferring 768. Strides: {strides}"
            )

            actual = _read_f32_data_with_shape(bytes(sret_buf), 3, expected_shape)
            cos_val = _cos(actual, expected)
            print(f"model-pattern 2tok — cos (forced shape): {cos_val:.10f}")
            assert cos_val > 0.999


# ── Test 4: Static-shape transpose (should always pass) ──────────────


TRANSPOSE_MLIR_STATIC = """module {{
  func.func @main_0(%input: tensor<1x12x2x64xf32>) -> tensor<1x2x12x64xf32> {{
    %0 = "sf.transpose"(%input) {{dim0 = 1 : i64, dim1 = 2 : i64}}
         : (tensor<1x12x2x64xf32>) -> tensor<1x2x12x64xf32>
    func.return %0 : tensor<1x2x12x64xf32>
  }}
}}"""


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestStaticShapeBaseline:
    """Static-shape baseline — should always pass, confirms op logic is correct."""

    def test_transpose_static_4d_2token(self):
        """sf.transpose with static shapes [1,12,2,64] -> [1,2,12,64].

        Static shapes bypass the dynamic descriptor bug.
        This test should ALWAYS PASS if the transpose lowering itself is correct.
        """
        batch, heads, seq, dim = 1, N_HEADS, 2, HEAD_DIM
        rng = np.random.RandomState(RANDOM_SEED)
        arr = rng.randn(batch, heads, seq, dim).astype(np.float32)
        expected = np.transpose(arr, (0, 2, 1, 3))
        expected_shape = (batch, seq, heads, dim)

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(TRANSPOSE_MLIR_STATIC.format(), td, "tpose_static_2tok")
            lib = ctypes.CDLL(dylib)

            sret_buf = _sret(4096)
            input_mem = _memref(arr)

            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 2
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(
                ctypes.byref(sret_buf),
                ctypes.byref(input_mem),
            )

            actual = _read_f32_data(bytes(sret_buf), 4)
            print(f"\nstatic transpose 2tok — shape: {actual.shape}")
            print(f"static transpose 2tok — expected: {expected_shape}")

            assert actual.shape == expected_shape, (
                f"Static transpose shape mismatch: {actual.shape} != {expected_shape}"
            )

            cos_val = _cos(actual, expected)
            assert cos_val > 0.999, (
                f"Static transpose cos={cos_val:.10f} < 0.999 — "
                f"underlying transpose logic is broken"
            )
