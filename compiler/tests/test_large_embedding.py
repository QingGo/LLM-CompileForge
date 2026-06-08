"""TDD: large embedding weight lookup regression test.

The full model's embedding weight (50272x768) has corrupted lookups at
high token indices. This test uses a synthetic large embedding to
reproduce the same pattern via the same compilation path.
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


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


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


def _memref(ptr, ndim, shape):
    strides = tuple(int(np.prod(shape[i + 1:])) for i in range(ndim))

    class M(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * ndim), ("strides", ctypes.c_int64 * ndim),
        ]
    return M(ctypes.c_void_p(ptr), ctypes.c_void_p(ptr), 0,
             (ctypes.c_int64 * ndim)(*shape), (ctypes.c_int64 * ndim)(*strides))


def _compile(sf_mlir: str, tmp_dir: str, name: str) -> str:
    import mlir.ir as ir
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf
    from compiler.backend.fixups import _fixup_unrealized_casts_pass
    from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir
    from compiler.pipeline import _apply_sf_to_linalg
    from compiler.backend.compile_utils import _compile_serveforge_free, link_dylib

    lowered = _apply_sf_to_linalg(sf_mlir)
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)
    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(lowered, ctx)
        lower_linalg_to_llvm_ir(mod)
        _fixup_unrealized_casts_pass(mod)
        m = os.path.join(tmp_dir, "m.mlir")
        l = os.path.join(tmp_dir, "m.ll")
        o = os.path.join(tmp_dir, "m.o")
        d = os.path.join(tmp_dir, f"{name}.dylib")
        with open(m, "w") as f:
            f.write(str(mod))
        subprocess.run([_find_tool("mlir-translate"), "--mlir-to-llvmir", m, "-o", l],
                       capture_output=True, text=True, check=True, timeout=60)
        subprocess.run([_find_tool("cc"), "-c", l, "-o", o, "-O0"],
                       capture_output=True, text=True, check=True, timeout=60)
        free_o = _compile_serveforge_free(tmp_dir)
        link_dylib([o, free_o], d)
        return d


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestLargeEmbedding:

    def test_large_embedding_high_indices(self):
        """sf.embedding with 50K vocab — high-index tokens match reference."""
        VOCAB, HIDDEN = 50000, 768
        rng = np.random.RandomState(42)
        emb_w = rng.randn(VOCAB, HIDDEN).astype(np.float32)

        input_ids = np.zeros((2, 4), dtype=np.int64)
        input_ids[0, 0] = 2       # low index
        input_ids[0, 1] = 32826   # high index
        input_ids[0, 2] = 85      # low index
        input_ids[0, 3] = 49999   # max index

        mlir = f"""module {{
  func.func @main_0(%ids: tensor<2x4xi64>, %w: tensor<{VOCAB}x{HIDDEN}xf32>) -> tensor<2x4x{HIDDEN}xf32> {{
    %emb = "sf.embedding"(%w, %ids) : (tensor<{VOCAB}x{HIDDEN}xf32>, tensor<2x4xi64>) -> tensor<2x4x{HIDDEN}xf32>
    return %emb : tensor<2x4x{HIDDEN}xf32>
  }}
}}"""
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(mlir, td, "test_large_emb")
            lib = ctypes.CDLL(dylib)
            in_m = _memref(input_ids.ctypes.data, 2, input_ids.shape)
            w_m = _memref(emb_w.ctypes.data, 2, emb_w.shape)
            sret = (ctypes.c_uint8 * 131072)()
            args = [ctypes.byref(sret), ctypes.byref(in_m), ctypes.byref(w_m)]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * 3
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*args)
            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = tuple(struct.unpack_from("<q", sb, 24 + 8 * i)[0] for i in range(3))
            actual = np.array(
                (ctypes.c_float * int(np.prod(sz))).from_address(al), dtype=np.float32
            ).reshape(sz)

            # Check per-token
            for batch in range(2):
                for seq in range(4):
                    tid = int(input_ids[batch, seq])
                    expected = emb_w[tid % VOCAB]
                    act = actual[batch, seq]
                    cos = _cos(act, expected)
                    status = "✅" if cos >= 0.9999 else "❌"
                    print(f"  token {tid:>5}: cos={cos:.8f} {status}")

            overall_cos = _cos(actual, emb_w[input_ids % VOCAB])
            assert overall_cos >= 0.9999, (
                f"Large embedding cos={overall_cos:.8f} < 0.9999"
            )
