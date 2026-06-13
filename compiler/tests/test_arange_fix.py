"""TDD tests for sf.arange lowering fix.

sf.arange(start) should produce [start, start+1, ..., start+size-1]
where size is determined by the output tensor's first dimension.
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


def _compile(sf_mlir: str, tmp_dir: str, name: str) -> str:
    import mlir.ir as ir
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf

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
        l = os.path.join(tmp_dir, "m.ll")
        o = os.path.join(tmp_dir, "m.o")
        d = os.path.join(tmp_dir, f"{name}.dylib")
        with open(m, "w") as f:
            f.write(str(mod))
        subprocess.run(
            [_find_tool("mlir-translate"), "--mlir-to-llvmir", m, "-o", l],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        subprocess.run(
            [_find_tool("cc"), "-c", l, "-o", o, "-O0"], capture_output=True, text=True, check=True, timeout=60
        )
        free_o = _compile_serveforge_free(tmp_dir)
        link_dylib([o, free_o], d)
        return d


@pytest.mark.integration
@pytest.mark.timeout(60)
class TestArangeFix:
    def test_arange_static_i64(self):
        """sf.arange(0) with static output tensor<4xi64> → [0,1,2,3]."""
        mlir = """module {
  func.func @main_0() -> tensor<4xi64> {
    %c0 = arith.constant dense<0> : tensor<1xi64>
    %a = "sf.arange"(%c0) : (tensor<1xi64>) -> tensor<4xi64>
    return %a : tensor<4xi64>
  }
}"""
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(mlir, td, "test1")
            lib = ctypes.CDLL(dylib)
            sret = (ctypes.c_uint8 * 4096)()
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p]
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(ctypes.byref(sret))
            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = struct.unpack_from("<q", sb, 24)[0]
            assert sz == 4, f"Expected size 4, got {sz}"
            actual = np.array((ctypes.c_int64 * sz).from_address(al), dtype=np.int64)
            expected = np.array([0, 1, 2, 3], dtype=np.int64)
            assert np.array_equal(actual, expected), (
                f"sf.arange(0)[4]: expected {expected.tolist()}, got {actual.tolist()}"
            )

    def test_arange_nonzero_i64(self):
        """sf.arange(5) with static output tensor<3xi64> → [5,6,7]."""
        mlir = """module {
  func.func @main_0() -> tensor<3xi64> {
    %c5 = arith.constant dense<5> : tensor<1xi64>
    %a = "sf.arange"(%c5) : (tensor<1xi64>) -> tensor<3xi64>
    return %a : tensor<3xi64>
  }
}"""
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(mlir, td, "test2")
            lib = ctypes.CDLL(dylib)
            sret = (ctypes.c_uint8 * 4096)()
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p]
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(ctypes.byref(sret))
            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = struct.unpack_from("<q", sb, 24)[0]
            assert sz == 3, f"Expected size 3, got {sz}"
            actual = np.array((ctypes.c_int64 * sz).from_address(al), dtype=np.int64)
            expected = np.array([5, 6, 7], dtype=np.int64)
            assert np.array_equal(actual, expected), (
                f"sf.arange(5)[3]: expected {expected.tolist()}, got {actual.tolist()}"
            )

    def test_arange_f32_static(self):
        """sf.arange(dense<0.0>:f32) with static output tensor<4xi64> → [0,1,2,3]."""
        mlir = """module {
  func.func @main_0() -> tensor<4xi64> {
    %c0 = arith.constant dense<0.0> : tensor<1xf32>
    %a = "sf.arange"(%c0) : (tensor<1xf32>) -> tensor<4xi64>
    return %a : tensor<4xi64>
  }
}"""
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(mlir, td, "test3")
            lib = ctypes.CDLL(dylib)
            sret = (ctypes.c_uint8 * 4096)()
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p]
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(ctypes.byref(sret))
            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = struct.unpack_from("<q", sb, 24)[0]
            assert sz == 4, f"Expected size 4, got {sz}"
            actual = np.array((ctypes.c_int64 * sz).from_address(al), dtype=np.int64)
            expected = np.array([0, 1, 2, 3], dtype=np.int64)
            assert np.array_equal(actual, expected), (
                f"sf.arange(0.0)[4]: expected {expected.tolist()}, got {actual.tolist()}"
            )

    def test_arange_f32_nonzero_static(self):
        """sf.arange(dense<2.0>:f32) with static output tensor<3xi64> → [2,3,4]."""
        mlir = """module {
  func.func @main_0() -> tensor<3xi64> {
    %c2 = arith.constant dense<2.0> : tensor<1xf32>
    %a = "sf.arange"(%c2) : (tensor<1xf32>) -> tensor<3xi64>
    return %a : tensor<3xi64>
  }
}"""
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(mlir, td, "test4")
            lib = ctypes.CDLL(dylib)
            sret = (ctypes.c_uint8 * 4096)()
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p]
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(ctypes.byref(sret))
            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = struct.unpack_from("<q", sb, 24)[0]
            assert sz == 3, f"Expected size 3, got {sz}"
            actual = np.array((ctypes.c_int64 * sz).from_address(al), dtype=np.int64)
            expected = np.array([2, 3, 4], dtype=np.int64)
            assert np.array_equal(actual, expected), (
                f"sf.arange(2.0)[3]: expected {expected.tolist()}, got {actual.tolist()}"
            )

    def test_arange_dynamic_f32_nonzero(self):
        """sf.arange(2.0) with dynamic output tensor<?xi64> + dyn_shape=[4] → [2,3,4,5]."""
        mlir = """module {
  func.func @main_0() -> tensor<?xi64> {
    %c2 = arith.constant dense<2.0> : tensor<1xf32>
    %sz = arith.constant dense<4> : tensor<1xi64>
    %a = "sf.arange"(%c2, %sz) : (tensor<1xf32>, tensor<1xi64>) -> tensor<?xi64>
    return %a : tensor<?xi64>
  }
}"""
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(mlir, td, "test_dyn1")
            lib = ctypes.CDLL(dylib)
            sret = (ctypes.c_uint8 * 4096)()
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p]
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(ctypes.byref(sret))
            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = struct.unpack_from("<q", sb, 24)[0]
            # Dynamic output: sz is positive (actual size), NOT start value
            actual = np.array((ctypes.c_int64 * sz).from_address(al), dtype=np.int64)
            expected = np.array([2, 3, 4, 5], dtype=np.int64)
            assert sz == 4, f"Expected dynamic size 4, got {sz}"
            assert np.array_equal(actual, expected), (
                f"sf.arange(2.0)[?](dyn=[4]): expected {expected.tolist()}, got {actual.tolist()}"
            )

    def test_arange_dynamic_i64_nonzero(self):
        """sf.arange(5) with dynamic output tensor<?xi64> + dyn_shape=[3] → [5,6,7]."""
        mlir = """module {
  func.func @main_0() -> tensor<?xi64> {
    %c5 = arith.constant dense<5> : tensor<1xi64>
    %sz = arith.constant dense<3> : tensor<1xi64>
    %a = "sf.arange"(%c5, %sz) : (tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
    return %a : tensor<?xi64>
  }
}"""
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(mlir, td, "test_dyn2")
            lib = ctypes.CDLL(dylib)
            sret = (ctypes.c_uint8 * 4096)()
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p]
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(ctypes.byref(sret))
            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = struct.unpack_from("<q", sb, 24)[0]
            actual = np.array((ctypes.c_int64 * sz).from_address(al), dtype=np.int64)
            expected = np.array([5, 6, 7], dtype=np.int64)
            assert sz == 3, f"Expected dynamic size 3, got {sz}"
            assert np.array_equal(actual, expected), (
                f"sf.arange(5)[?](dyn=[3]): expected {expected.tolist()}, got {actual.tolist()}"
            )
