"""Memory safety tests for compiled dylib functions.

Verifies that compiled dylib ciface functions do not write past
allocated buffer boundaries.  Uses macOS malloc diagnostics to
detect heap corruption.

TDD cycle:
  RED:    test detects heap corruption -> fails
  GREEN:  bug fixed -> test passes
"""

from __future__ import annotations

import ctypes
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from compiler.dylib_ffi import DEFAULT_SRET_SIZE  # noqa: E402


def _memref(data_ptr: int, rank: int, shape: tuple[int, ...]) -> ctypes.Structure:
    assert len(shape) == rank or (rank == 0 and shape == ())
    n = max(rank, 1)
    total = 8 + 8 + 8 + n * 8 + n * 8
    buf = bytearray(total)
    struct.pack_into("QQQ", buf, 0, data_ptr, data_ptr, 0)
    off = 24
    for s in (list(shape) + [1] * n)[:n]:
        struct.pack_into("Q", buf, off, s)
        off += 8
    for s in range(1, n + 1):
        val = 1
        for dim in (list(shape) + [1] * n)[s : min(s + 1, n)]:
            val *= max(dim, 1)
        struct.pack_into("Q", buf, off, val * 4)
        off += 8
    raw_memref = type("RawMemRef", (ctypes.Structure,), {"_fields_": [("raw", ctypes.c_uint8 * total)]})
    inst = raw_memref()
    ctypes.memmove(ctypes.addressof(inst), bytes(buf), total)
    return inst


def _compile_simple_model(model_name: str, mlir_text: str, work_dir: str) -> str:
    mlir_path = os.path.join(work_dir, f"{model_name}.mlir")
    lowered_path = os.path.join(work_dir, f"{model_name}.lowered.mlir")
    ll_path = os.path.join(work_dir, f"{model_name}.ll")
    o_path = os.path.join(work_dir, f"{model_name}.o")
    dylib_path = os.path.join(work_dir, f"lib{model_name}.dylib")

    with open(mlir_path, "w") as f:
        f.write(mlir_text)

    env = os.environ.copy()
    sf_opt = str(_PROJECT_ROOT / "sf-dialect" / "build" / "tools" / "sf-opt" / "sf-opt")
    result = subprocess.run(
        [sf_opt, "--sf-promote-weights", "--sf-lower-to-linalg", mlir_path],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sf-opt failed: {result.stderr[:500]}")
    with open(lowered_path, "w") as f:
        f.write(result.stdout)

    mlir_opt = str(_PROJECT_ROOT / "llvm-project" / "build" / "bin" / "mlir-opt")
    mlir_translate = str(_PROJECT_ROOT / "llvm-project" / "build" / "bin" / "mlir-translate")
    llc = str(_PROJECT_ROOT / "llvm-project" / "build" / "bin" / "llc")

    passes = (
        "canonicalize,cse,"
        "one-shot-bufferize{bufferize-function-boundaries},"
        "convert-linalg-to-loops,lower-affine,convert-scf-to-cf,"
        "finalize-memref-to-llvm,convert-math-to-llvm,"
        "convert-arith-to-llvm,convert-func-to-llvm,"
        "convert-cf-to-llvm,reconcile-unrealized-casts"
    )
    result = subprocess.run(
        [mlir_opt, lowered_path, "--pass-pipeline", f"builtin.module({passes})"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mlir-opt failed: {result.stderr[:500]}")

    result2 = subprocess.run(
        [mlir_translate, "--mlir-to-llvmir", "-o", ll_path],
        input=result.stdout,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    if result2.returncode != 0:
        raise RuntimeError(f"mlir-translate failed: {result2.stderr[:500]}")

    result3 = subprocess.run(
        [llc, "-O0", "-filetype=obj", ll_path, "-o", o_path], capture_output=True, text=True, timeout=30, env=env
    )
    if result3.returncode != 0:
        raise RuntimeError(f"llc failed: {result3.stderr[:500]}")

    result4 = subprocess.run(
        ["cc", "-shared", "-o", dylib_path, o_path], capture_output=True, text=True, timeout=30, env=env
    )
    if result4.returncode != 0:
        raise RuntimeError(f"cc failed: {result4.stderr[:500]}")

    return dylib_path


def _malloc_error_count() -> int:
    lib_system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    lib_system.malloc_zone_check.restype = ctypes.c_int
    return lib_system.malloc_zone_check(None)


class TestDylibMemorySafety:
    def test_simple_matmul_no_heap_corruption(self):
        mlir = """
module {
  func.func @main_0(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<4x4xf32>
      attributes {llvm.emit_c_interface} {
    %0 = "sf.matmul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}"""
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile_simple_model("simple_matmul", mlir, td)
            lib = ctypes.CDLL(dylib)

            rng = np.random.RandomState(42)
            a = rng.randn(4, 8).astype(np.float32)
            b = rng.randn(8, 4).astype(np.float32)

            ma = _memref(a.ctypes.data, 2, a.shape)
            mb = _memref(b.ctypes.data, 2, b.shape)
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()

            kernel = lib._mlir_ciface_main_0
            kernel.argtypes = [ctypes.c_void_p] * 3
            kernel.restype = None

            pre_errs = _malloc_error_count()
            kernel(ctypes.byref(sret), ctypes.byref(ma), ctypes.byref(mb))
            post_errs = _malloc_error_count()

            assert post_errs <= pre_errs, f"Heap corruption increased: pre={pre_errs} post={post_errs}"

    @pytest.mark.xfail(reason="requires compiled artifacts — run make build-all")
    def test_full_model_no_heap_corruption(self):
        sys.path.insert(0, str(_PROJECT_ROOT))
        from gen.proto.python import sfa_abi_pb2

        with open(str(_PROJECT_ROOT / "outputs/compiled/opt_125m_test/sfa_abi.c")) as f:
            hex_bytes = re.findall(r"0x[0-9a-fA-F]{2}", f.read())
        raw = bytes(int(h, 16) for h in hex_bytes)
        hdr = sfa_abi_pb2.SfaAbiHeader()
        hdr.ParseFromString(raw)

        dylib_path = str(_PROJECT_ROOT / "outputs/compiled/opt_125m_test/libopt_125m.dylib")
        if not os.path.exists(dylib_path):
            pytest.skip("opt-125m dylib not compiled")

        lib = ctypes.CDLL(dylib_path)
        func0 = hdr.funcs[0]
        kernel = getattr(lib, func0.symbol)

        memrefs = []
        for inf in func0.input_fields:
            rank = max(inf.rank, 1)
            dims = tuple(list(inf.dims) or [1])
            padded = dims + (1,) * (rank - len(dims))
            nelem = int(np.prod(tuple(d if d > 0 else 1 for d in padded)))
            buf = np.zeros(nelem, dtype=np.float32)

            total = 8 + 8 + 8 + rank * 8 + rank * 8
            raw_mr = bytearray(total)
            ptr = buf.ctypes.data
            struct.pack_into("QQQ", raw_mr, 0, ptr, ptr, 0)
            off = 24
            for s in padded[:rank]:
                struct.pack_into("Q", raw_mr, off, s)
                off += 8
            stride = 4
            for s in reversed(padded[:rank]):
                struct.pack_into("Q", raw_mr, off, stride)
                off += 8
                stride *= s if s > 0 else 1
            mr = type("MR", (ctypes.Structure,), {"_fields_": [("raw", ctypes.c_uint8 * total)]})
            inst = mr()
            ctypes.memmove(ctypes.addressof(inst), bytes(raw_mr), total)
            memrefs.append(inst)

        sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
        kernel.argtypes = [ctypes.c_void_p] * (1 + len(memrefs))
        kernel.restype = None

        env = os.environ.copy()
        env.setdefault("MallocGuardEdges", "1")
        pre_errs = _malloc_error_count()
        kernel(ctypes.byref(sret), *(ctypes.byref(m) for m in memrefs))
        post_errs = _malloc_error_count()

        assert post_errs <= pre_errs, f"Heap corruption increased: pre={pre_errs} post={post_errs}"
