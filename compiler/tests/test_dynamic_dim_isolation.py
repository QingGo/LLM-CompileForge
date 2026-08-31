"""Isolation test: dynamic-dimension single FFN block vs HF reference.

Validates that the linalg lowering pipeline correctly handles
dynamic tensor dimensions (tensor<?x?x768>) — not just static.
If this fails, the bug is in dynamic-dimension handling, not
multi-function interaction.
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
import safetensors.torch
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from compiler.backend.compile_utils import (  # noqa: E402
    _compile_serveforge_free,
    _find_llc,
    _setup_mlir_path,
    link_dylib,
)
from compiler.backend.fixups import _fixup_unrealized_casts_pass  # noqa: E402
from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir  # noqa: E402
from compiler.dylib_ffi import DEFAULT_SRET_SIZE  # noqa: E402
from compiler.pipeline.lowering import SF_LOWERING_PIPELINE  # noqa: E402

_setup_mlir_path()
import mlir.ir as ir  # noqa: E402
import mlir.passmanager as pm  # noqa: E402
from mlir_sf._mlir_libs._sfDialectsNanobind import sf  # noqa: E402

SAFETENSORS_PATH = Path(
    "/Users/zeng/.cache/huggingface/hub/models--facebook--opt-125m/"
    "snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/model.safetensors"
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))


def _memref(ptr, ndim, shape):
    strides = tuple(int(np.prod(shape[i + 1 :])) for i in range(ndim))

    class M(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * ndim),
            ("strides", ctypes.c_int64 * ndim),
        ]

    return M(
        ctypes.c_void_p(ptr),
        ctypes.c_void_p(ptr),
        0,
        (ctypes.c_int64 * ndim)(*shape),
        (ctypes.c_int64 * ndim)(*strides),
    )


def _unpack_sret_3d(sret_bytes: bytes):
    sb = bytes(sret_bytes)
    al = struct.unpack_from("<Q", sb, 8)[0]
    sz = tuple(struct.unpack_from("<q", sb, 24 + 8 * i)[0] for i in range(3))
    n = int(np.prod(sz))
    return np.array((ctypes.c_float * n).from_address(al), dtype=np.float32).reshape(sz)


def _compile_dylib(mlir: str, td: str) -> str:
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)
    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(mlir, ctx)
        pman = pm.PassManager.parse(f"builtin.module({SF_LOWERING_PIPELINE})", ctx)
        pman.run(mod.operation)
        lower_linalg_to_llvm_ir(mod)
        _fixup_unrealized_casts_pass(mod)

        m_path = os.path.join(td, "m.mlir")
        ll_path = os.path.join(td, "m.ll")
        o_path = os.path.join(td, "m.o")
        dylib_path = os.path.join(td, "test.dylib")

        with open(m_path, "w") as f:
            f.write(str(mod))
        subprocess.run(
            ["./llvm-project/build/bin/mlir-translate", "--mlir-to-llvmir", m_path, "-o", ll_path],
            capture_output=True,
            check=True,
            timeout=120,
        )
        subprocess.run(
            [_find_llc(), "-filetype=obj", ll_path, "-o", o_path],
            capture_output=True,
            check=True,
            timeout=120,
        )
        free_o = _compile_serveforge_free(td)
        link_dylib([o_path, free_o], dylib_path)
        return dylib_path


# --- Static FFN MLIR (benchmark from existing test) ---

FFN_STATIC_MLIR = """module {{
  func.func @main_0(%x: tensor<2x4x64xf32>, %nw: tensor<64xf32>,
      %w1: tensor<256x64xf32>, %b1: tensor<256xf32>,
      %w2: tensor<64x256xf32>, %b2: tensor<64xf32>) -> tensor<2x4x64xf32> {{
    %n = "sf.rms_norm"(%x, %nw) :
      (tensor<2x4x64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>
    %fc1 = "sf.linear"(%n, %w1, %b1) :
      (tensor<2x4x64xf32>, tensor<256x64xf32>, tensor<256xf32>) -> tensor<2x4x256xf32>
    %act = "sf.silu"(%fc1) : (tensor<2x4x256xf32>) -> tensor<2x4x256xf32>
    %fc2 = "sf.linear"(%act, %w2, %b2) :
      (tensor<2x4x256xf32>, tensor<64x256xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>
    %out = "sf.add"(%fc2, %x) :
      (tensor<2x4x64xf32>, tensor<2x4x64xf32>) -> tensor<2x4x64xf32>
    return %out : tensor<2x4x64xf32>
  }}
}}
"""

# --- Dynamic FFN MLIR (same ops with ?x? dynamic dims) ---

FFN_DYNAMIC_MLIR = """module {{
  func.func @main_0(%x: tensor<?x?x64xf32>, %nw: tensor<64xf32>,
      %w1: tensor<256x64xf32>, %b1: tensor<256xf32>,
      %w2: tensor<64x256xf32>, %b2: tensor<64xf32>) -> tensor<?x?x64xf32> {{
    %n = "sf.rms_norm"(%x, %nw) :
      (tensor<?x?x64xf32>, tensor<64xf32>) -> tensor<?x?x64xf32>
    %fc1 = "sf.linear"(%n, %w1, %b1) :
      (tensor<?x?x64xf32>, tensor<256x64xf32>, tensor<256xf32>) -> tensor<?x?x256xf32>
    %act = "sf.silu"(%fc1) : (tensor<?x?x256xf32>) -> tensor<?x?x256xf32>
    %fc2 = "sf.linear"(%act, %w2, %b2) :
      (tensor<?x?x256xf32>, tensor<64x256xf32>, tensor<64xf32>) -> tensor<?x?x64xf32>
    %out = "sf.add"(%fc2, %x) :
      (tensor<?x?x64xf32>, tensor<?x?x64xf32>) -> tensor<?x?x64xf32>
    return %out : tensor<?x?x64xf32>
  }}
}}
"""


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestDynamicDimIsolation:
    """Isolate dynamic dimension handling in the lowering pipeline."""

    def test_ffn_static_64dim(self):
        """Static FFN: known-good baseline (cos >= 0.9999)."""
        batch, seq, hidden, ffn_dim = 2, 4, 64, 256
        rng = np.random.RandomState(42)
        x = rng.randn(batch, seq, hidden).astype(np.float32)
        nw = rng.randn(hidden).astype(np.float32)
        w1 = rng.randn(ffn_dim, hidden).astype(np.float32)
        b1 = rng.randn(ffn_dim).astype(np.float32)
        w2 = rng.randn(hidden, ffn_dim).astype(np.float32)
        b2 = rng.randn(hidden).astype(np.float32)

        mlir = FFN_STATIC_MLIR.format()
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile_dylib(mlir, td)
            lib = ctypes.CDLL(dylib)
            inputs = [x, nw, w1, b1, w2, b2]
            mrs = [_memref(a.ctypes.data, a.ndim, a.shape) for a in inputs]
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret)] + [ctypes.byref(mr) for mr in mrs]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * len(args)
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*args)
            actual = _unpack_sret_3d(sret)

        norm = (x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
        fc1 = np.dot(norm.reshape(-1, hidden), w1.T) + b1
        silu = fc1 / (1.0 + np.exp(-fc1))
        fc2 = np.dot(silu, w2.T) + b2
        expected = (fc2 + x.reshape(-1, hidden)).reshape(batch, seq, hidden)

        cos_val = _cos(actual, expected)
        print(f"\nStatic FFN: cos={cos_val:.8f}, actual mean={actual.mean():.4f}, expected mean={expected.mean():.4f}")
        assert cos_val >= 0.9999, f"Static FFN baseline failed: cos={cos_val:.8f}"

    def test_ffn_dynamic_small(self):
        """Dynamic FFN (small dims): same data as static FFN but with ?x? type.

        This tests whether the dynamic dimension path in linalg lowering
        produces the same result as the static path.
        """
        batch, seq, hidden, ffn_dim = 2, 4, 64, 256
        rng = np.random.RandomState(42)
        x = rng.randn(batch, seq, hidden).astype(np.float32)
        nw = rng.randn(hidden).astype(np.float32)
        w1 = rng.randn(ffn_dim, hidden).astype(np.float32)
        b1 = rng.randn(ffn_dim).astype(np.float32)
        w2 = rng.randn(hidden, ffn_dim).astype(np.float32)
        b2 = rng.randn(hidden).astype(np.float32)

        mlir = FFN_DYNAMIC_MLIR.format()
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile_dylib(mlir, td)
            lib = ctypes.CDLL(dylib)
            inputs = [x, nw, w1, b1, w2, b2]
            mrs = [_memref(a.ctypes.data, a.ndim, a.shape) for a in inputs]
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret)] + [ctypes.byref(mr) for mr in mrs]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * len(args)
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*args)
            actual = _unpack_sret_3d(sret)

        norm = (x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
        fc1 = np.dot(norm.reshape(-1, hidden), w1.T) + b1
        silu = fc1 / (1.0 + np.exp(-fc1))
        fc2 = np.dot(silu, w2.T) + b2
        expected = (fc2 + x.reshape(-1, hidden)).reshape(batch, seq, hidden)

        cos_val = _cos(actual, expected)
        print(
            f"\nDynamic FFN (small): cos={cos_val:.8f}, actual mean={actual.mean():.4f}, "
            f"expected mean={expected.mean():.4f}"
        )
        assert cos_val >= 0.9999, f"Dynamic FFN failed: cos={cos_val:.8f} — dynamic dim handling has a bug"

    def test_ffn_real_weights(self):
        """Dynamic FFN with real OPT-125m layer-0 weights (real hidden=768).

        Uses safetensors weights and compares with HF layer-0 output.
        """
        st = safetensors.torch.load_file(str(SAFETENSORS_PATH))

        # Layer 0 FFN weights
        prefix = "model.decoder.layers.0."
        nw = st[f"{prefix}self_attn_layer_norm.weight"].numpy().astype(np.float32)
        w1 = st[f"{prefix}fc1.weight"].numpy().astype(np.float32)  # [3072, 768]
        b1 = st[f"{prefix}fc1.bias"].numpy().astype(np.float32)  # [3072]
        w2 = st[f"{prefix}fc2.weight"].numpy().astype(np.float32)  # [768, 3072]
        b2 = st[f"{prefix}fc2.bias"].numpy().astype(np.float32)  # [768]

        hidden = 768
        batch, seq = 2, 4
        rng = np.random.RandomState(42)
        x = rng.randn(batch, seq, hidden).astype(np.float32)

        # Build dynamic MLIR with real dims
        mlir = """module {
  func.func @main_0(%x: tensor<?x?x768xf32>, %nw: tensor<768xf32>,
      %w1: tensor<3072x768xf32>, %b1: tensor<3072xf32>,
      %w2: tensor<768x3072xf32>, %b2: tensor<768xf32>) -> tensor<?x?x768xf32> {
    %n = "sf.rms_norm"(%x, %nw) :
      (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %fc1 = "sf.linear"(%n, %w1, %b1) :
      (tensor<?x?x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) -> tensor<?x?x3072xf32>
    %act = "sf.silu"(%fc1) : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>
    %fc2 = "sf.linear"(%act, %w2, %b2) :
      (tensor<?x?x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %out = "sf.add"(%fc2, %x) :
      (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
    return %out : tensor<?x?x768xf32>
  }
}
"""
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile_dylib(mlir, td)
            lib = ctypes.CDLL(dylib)
            inputs = [x, nw, w1, b1, w2, b2]
            mrs = [_memref(a.ctypes.data, a.ndim, a.shape) for a in inputs]
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret)] + [ctypes.byref(mr) for mr in mrs]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * len(args)
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*args)
            actual = _unpack_sret_3d(sret)

        # HF reference
        x_t = torch.from_numpy(x)
        nw_t = torch.from_numpy(nw)
        w1_t = torch.from_numpy(w1)
        b1_t = torch.from_numpy(b1)
        w2_t = torch.from_numpy(w2)
        b2_t = torch.from_numpy(b2)

        # rms_norm
        rms = torch.sqrt(x_t.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        norm = (x_t / rms) * nw_t
        # fc1
        fc1 = torch.nn.functional.linear(norm, w1_t, b1_t)
        # silu
        silu = torch.nn.functional.silu(fc1)
        # fc2
        fc2 = torch.nn.functional.linear(silu, w2_t, b2_t)
        # add
        expected = (fc2 + x_t).numpy()

        cos_val = _cos(actual, expected)
        mae = float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).mean())
        print(f"\nReal-weights dynamic FFN (768x3072): cos={cos_val:.8f}, MAE={mae:.6e}")
        print(f"  actual mean={actual.mean():.6f}, expected mean={expected.mean():.6f}")
        assert cos_val >= 0.9999, (
            f"Dynamic FFN with real weights failed: cos={cos_val:.8f} < 0.9999\n"
            f"MAE={mae:.2e} — dynamic dimension handling has a bug"
        )
