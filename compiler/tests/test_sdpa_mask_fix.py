"""TDD: SDPA causal mask broadcast fix.

Bug: when mask [1,1,S,S] is broadcast to scores [1,H,S,S] via makeBinaryOp,
only the first element of the mask is read, producing all-zero additive
regardless of mask values. This makes causal masking a no-op.

Test: compile sf.scaled_dot_product_attention with a [[1,0],[1,1]] causal mask.
Verify masked position (0,1) gets near-zero attention weight.
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


def _find_tool(name: str) -> str:
    c = [name]
    if name in ("cc", "clang"):
        c.insert(0, "/usr/local/opt/llvm/bin/clang")
    c.append(str(ROOT / "llvm-project" / "build" / "bin" / name))
    for p in c:
        if Path(p).is_file():
            return str(p)
        try:
            if subprocess.run([p, "--version"], capture_output=True, timeout=5).returncode == 0:
                return p
        except FileNotFoundError:
            continue
    raise RuntimeError(f"{name} not found")


def _compile(sf_mlir: str, tmp_dir: str, name: str) -> str:
    """Compile sf MLIR → lowered → LLVM → cc → dylib."""
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
        ll_file = os.path.join(tmp_dir, "m.ll")
        o = os.path.join(tmp_dir, "m.o")
        d = os.path.join(tmp_dir, f"{name}.dylib")
        with open(m, "w") as f:
            f.write(str(mod))
        subprocess.run(
            [_find_tool("mlir-translate"), "--mlir-to-llvmir", m, "-o", ll_file],
            capture_output=True, text=True, check=True, timeout=60,
        )
        subprocess.run(
            [_find_tool("cc"), "-c", ll_file, "-o", o, "-O0"],
            capture_output=True, text=True, check=True, timeout=60,
        )
        free_o = _compile_serveforge_free(tmp_dir)
        link_dylib([o, free_o], d)
        return d


def _sret(size: int) -> ctypes.Array:
    return (ctypes.c_uint8 * size)()


def _memref(arr: np.ndarray):
    """Build ctypes MemRef descriptor from numpy array."""
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


def _parse_sret_f32(sret_bytes: bytes, rank: int) -> np.ndarray:
    """Parse a single rank-N f32 output from sret buffer."""
    # Descriptor layout: allocated(8) aligned(8) offset(8) sizes(8*rank) strides(8*rank)
    aligned = struct.unpack_from("<Q", sret_bytes, 8)[0]
    sizes = []
    for i in range(rank):
        s = struct.unpack_from("<q", sret_bytes, 24 + 8 * i)[0]
        sizes.append(s if s > 0 else 1)
    n = int(np.prod(sizes))
    if n > 0 and aligned != 0:
        buf = (ctypes.c_float * n).from_address(aligned)
        return np.array(buf, dtype=np.float32).reshape(sizes)
    return np.array([], dtype=np.float32)


# ── Test data ──────────────────────────────────────────────────────
# Q = K = [[[1,0], [0,0]]]  (batch=1, heads=1, seq=2, dim=2)
# V = [[[1,0], [0,0]]]
# mask = [[1,0], [1,1]]  (causal: position 0 can't see position 1)

Q_DATA = np.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=np.float32)
K_DATA = np.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=np.float32)
V_DATA = np.array([[[[1.0, 0.0], [0.0, 0.0]]]], dtype=np.float32)
MASK_DATA = np.array([[[[1.0, 0.0], [1.0, 1.0]]]], dtype=np.float32)


SDPA_MLIR = """module {{
  func.func @main_0(%q: tensor<1x1x2x2xf32>, %k: tensor<1x1x2x2xf32>,
                     %v: tensor<1x1x2x2xf32>, %mask: tensor<1x1x2x2xf32>)
      -> tensor<1x1x2x2xf32> {{
    %0 = "sf.scaled_dot_product_attention"(%q, %k, %v, %mask)
         {{scale = 1.0 : f64}}
         : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>,
            tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    func.return %0 : tensor<1x1x2x2xf32>
  }}
}}"""

# Dynamic-shape variant — matches the actual model scenario where
# batch, heads, and seq dims are all dynamic.
SDPA_MLIR_DYNAMIC = """module {{
  func.func @main_0(%q: tensor<?x?x?x2xf32>, %k: tensor<?x?x?x2xf32>,
                     %v: tensor<?x?x?x2xf32>, %mask: tensor<?x1x?x?xf32>)
      -> tensor<?x?x?x2xf32> {{
    %0 = "sf.scaled_dot_product_attention"(%q, %k, %v, %mask)
         {{scale = 1.0 : f64}}
         : (tensor<?x?x?x2xf32>, tensor<?x?x?x2xf32>,
            tensor<?x?x?x2xf32>, tensor<?x1x?x?xf32>) -> tensor<?x?x?x2xf32>
    func.return %0 : tensor<?x?x?x2xf32>
  }}
}}"""


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestSdpaMaskFix:
    """Verify causal mask is actually applied in SDPA computation."""

    def test_causal_mask_blocks_future_token(self):
        """With mask=[[1,0],[1,1]], token 0 should NOT attend to token 1.

        Expected output[0,0,0,:] = [1.0, 0.0] (only self: 1.0*[1,0] + 0.0*[0,0])
        Expected output[0,0,1,:] = [0.5, 0.0] (equal attn: 0.5*[1,0] + 0.5*[0,0])

        If mask is broken → token 0 gets ~0.731 instead of 1.0.
        """
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(SDPA_MLIR.format(), td, "sdpa_mask_test")
            lib = ctypes.CDLL(dylib)

            sret = _sret(4096)
            q_mem = _memref(Q_DATA)
            k_mem = _memref(K_DATA)
            v_mem = _memref(V_DATA)
            mask_mem = _memref(MASK_DATA)

            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 5
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(
                ctypes.byref(sret),
                ctypes.byref(q_mem), ctypes.byref(k_mem),
                ctypes.byref(v_mem), ctypes.byref(mask_mem),
            )

            out = _parse_sret_f32(bytes(sret), 4)
            assert out.shape == (1, 1, 2, 2), f"Unexpected shape: {out.shape}"

            val_00 = out[0, 0, 0, 0]  # token 0, dim 0 — should be 1.0
            val_10 = out[0, 0, 1, 0]  # token 1, dim 0 — should be 0.5

            assert abs(val_00 - 1.0) < 0.01, (
                f"Token 0 should get ~1.0 (only self-attention), got {val_00:.6f}. "
                f"Mask appears BROKEN. output[0,0]={out[0,0].tolist()}"
            )
            assert abs(val_10 - 0.5) < 0.01, (
                f"Token 1 should get ~0.5, got {val_10:.6f}"
            )

    def test_causal_mask_dynamic_shape(self):
        """Same test with dynamic shapes — matches actual model scenario.

        This is where the real bug manifests: linalg.generic broadcast
        fails when all dims are dynamic (<?x?x?x?xf32>).
        """
        q_dyn = Q_DATA.copy()
        k_dyn = K_DATA.copy()
        v_dyn = V_DATA.copy()
        mask_dyn = MASK_DATA.copy()

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(SDPA_MLIR_DYNAMIC.format(), td, "sdpa_mask_dyn")
            lib = ctypes.CDLL(dylib)

            sret = _sret(4096)
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 5
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(
                ctypes.byref(sret),
                ctypes.byref(_memref(q_dyn)), ctypes.byref(_memref(k_dyn)),
                ctypes.byref(_memref(v_dyn)), ctypes.byref(_memref(mask_dyn)),
            )

            out = _parse_sret_f32(bytes(sret), 4)
            assert out.shape == (1, 1, 2, 2), f"Unexpected shape: {out.shape}"

            val_00 = out[0, 0, 0, 0]
            val_10 = out[0, 0, 1, 0]

            assert abs(val_00 - 1.0) < 0.01, (
                f"DYNAMIC: Token 0 should get ~1.0, got {val_00:.6f}. "
                f"Mask broadcast fails with dynamic dims. output[0,0]={out[0,0].tolist()}"
            )
            assert abs(val_10 - 0.5) < 0.01, (
                f"DYNAMIC: Token 1 should get ~0.5, got {val_10:.6f}"
            )
