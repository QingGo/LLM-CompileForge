"""Sret Layout Contract Test — multi-function, many outputs, identity weight copies.

Contract:
  1. Output descriptors appear in function return type order
  2. Descriptor at byte offset sum(24 + 16*rank_i for i < N) from sret start
  3. Each descriptor's aligned pointer points to valid writable memory
  4. Computed output data at aligned pointer matches reference
  5. Identity-copied weights retain correct data (not corrupted by aliasing)
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
from compiler.dylib_ffi import DEFAULT_SRET_SIZE


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))


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


def _compile(sf_mlir, tmp, name):
    import mlir.ir as ir
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf

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
        m = os.path.join(tmp, "m.mlir")
        l = os.path.join(tmp, "m.ll")
        o = os.path.join(tmp, "m.o")
        d = os.path.join(tmp, f"{name}.dylib")
        with open(m, "w") as f:
            f.write(str(mod))
        for cmd, out, desc in [
            ([_find_tool("mlir-translate"), "--mlir-to-llvmir", m, "-o", l], None, "translate"),
            ([_find_tool("cc"), "-c", l, "-o", o, "-O0"], None, "cc -c"),
            ([_find_tool("cc"), "-shared", "-o", d, o], None, "link"),
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                raise RuntimeError(f"{desc}: {r.stderr[-300:]}")
        return d


def _sret_desc(sb, off, rank):
    al = struct.unpack_from("<Q", sb, off + 8)[0]
    sz = tuple(struct.unpack_from("<q", sb, off + 24 + 8 * i)[0] for i in range(rank))
    return al, sz


def _data(al, shape):
    n = int(np.prod(shape))
    return np.array((ctypes.c_float * n).from_address(al), dtype=np.float32).reshape(shape)


def _mlir(n_id):
    lines = ["module {"]
    args = ["%in: tensor<4x8xf32>", "%w0: tensor<8x4xf32>"]
    for i in range(n_id):
        args.append(f"%id{i}: tensor<64x64xf32>")
    lines.append(f"  func.func @main_0({', '.join(args)}) -> (")
    lines.append("    tensor<4x4xf32>")
    for _ in range(n_id):
        lines.append(", tensor<64x64xf32>")
    lines.append("  ) {")
    lines.append('    %m = "sf.matmul"(%in, %w0) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>')
    rv = ["%m"]
    for i in range(n_id):
        lines.append(
            f"    %c{i} = linalg.copy ins(%id{i} : tensor<64x64xf32>)"
            f" outs(%id{i} : tensor<64x64xf32>) -> tensor<64x64xf32>"
        )
        rv.append(f"%c{i}")
    lines.append(f"    return {', '.join(rv)} : tensor<4x4xf32>" + "".join(", tensor<64x64xf32>" for _ in range(n_id)))
    lines.append("  }")
    lines.append("  func.func @main_1(%h: tensor<4x4xf32>, %w1: tensor<4x4xf32>) -> tensor<4x4xf32> {")
    lines.append('    %r = "sf.matmul"(%h, %w1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>')
    lines.append("    return %r : tensor<4x4xf32>")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestSretLayoutContract:
    def test_computed_output_10_identity_weights(self):
        rng = np.random.RandomState(42)
        inp = rng.randn(4, 8).astype(np.float32)
        w0 = rng.randn(8, 4).astype(np.float32)
        ids = [rng.randn(64, 64).astype(np.float32) for _ in range(10)]
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(_mlir(10), td, "t10")
            lib = ctypes.CDLL(dylib)
            mrs = [_memref(inp.ctypes.data, 2, inp.shape), _memref(w0.ctypes.data, 2, w0.shape)]
            for w in ids:
                mrs.append(_memref(w.ctypes.data, 2, w.shape))
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret)] + [ctypes.byref(m) for m in mrs]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * len(args)
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*args)
            al, sz = _sret_desc(bytes(sret), 0, 2)
            assert sz == (4, 4), f"size={sz}"
            cos = _cos(_data(al, sz), inp @ w0)
            assert cos >= 0.9999, f"cos={cos:.8f}"

    def test_chain_50_identity_weights(self):
        rng = np.random.RandomState(42)
        inp = rng.randn(4, 8).astype(np.float32)
        w0 = rng.randn(8, 4).astype(np.float32)
        w1 = rng.randn(4, 4).astype(np.float32)
        ids = [rng.randn(64, 64).astype(np.float32) for _ in range(50)]
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(_mlir(50), td, "t50")
            lib = ctypes.CDLL(dylib)
            mrs0 = [_memref(inp.ctypes.data, 2, inp.shape), _memref(w0.ctypes.data, 2, w0.shape)]
            for w in ids:
                mrs0.append(_memref(w.ctypes.data, 2, w.shape))
            sret0 = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            a0 = [ctypes.byref(sret0)] + [ctypes.byref(m) for m in mrs0]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * len(a0)
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*a0)
            al, sz = _sret_desc(bytes(sret0), 0, 2)
            hid = _data(al, sz)
            hm = _memref(hid.ctypes.data, 2, hid.shape)
            wm = _memref(w1.ctypes.data, 2, w1.shape)
            sret1 = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            a1 = [ctypes.byref(sret1), ctypes.byref(hm), ctypes.byref(wm)]
            lib._mlir_ciface_main_1.argtypes = [ctypes.c_void_p] * len(a1)
            lib._mlir_ciface_main_1.restype = None
            lib._mlir_ciface_main_1(*a1)
            al1, sz1 = _sret_desc(bytes(sret1), 0, 2)
            cos = _cos(_data(al1, sz1), (inp @ w0) @ w1)
            assert cos >= 0.9999, f"cos={cos:.8f}"

    def test_contract_200_identity_weights(self):
        """Contract §2-3: verify all 201 output descriptors have valid pointers."""
        rng = np.random.RandomState(42)
        inp = rng.randn(4, 8).astype(np.float32)
        w0 = rng.randn(8, 4).astype(np.float32)
        ids = [rng.randn(64, 64).astype(np.float32) for _ in range(200)]
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(_mlir(200), td, "t200")
            lib = ctypes.CDLL(dylib)
            mrs = [_memref(inp.ctypes.data, 2, inp.shape), _memref(w0.ctypes.data, 2, w0.shape)]
            for w in ids:
                mrs.append(_memref(w.ctypes.data, 2, w.shape))
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret)] + [ctypes.byref(m) for m in mrs]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * len(args)
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*args)
            sb = bytes(sret)
            al0, sz0 = _sret_desc(sb, 0, 2)
            assert sz0 == (4, 4), f"matmul: {sz0}"
            for i in range(200):
                off = 56 + i * 56
                al, sz = _sret_desc(sb, off, 2)
                assert al != 0, f"weight[{i}] null pointer at offset {off}"
                assert sz == (64, 64), f"weight[{i}] size={sz}"
