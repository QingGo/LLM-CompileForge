"""Regression suite: sf dialect ops compiled to dylib via LLVM lowering pipeline.

Validates that individual ops and their combinations produce numerically
correct output (cos >= 0.999) when compiled through the full sf->linalg->LLVM
lowering pipeline.  All reference computations use fp32 to match dylib
precision.
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

from compiler.backend.compile_utils import (
    _compile_serveforge_free,
    _find_llc,
    _setup_mlir_path,
    link_dylib,
)
from compiler.backend.fixups import _fixup_unrealized_casts_pass
from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir
from compiler.pipeline.lowering import SF_LOWERING_PIPELINE
from compiler.sfcf_parser import DEFAULT_SRET_SIZE
from scripts._cos import cosine_similarity

_setup_mlir_path()
import mlir.ir as ir  # noqa: E402
import mlir.passmanager as pm  # noqa: E402
from mlir_sf._mlir_libs._sfDialectsNanobind import sf  # noqa: E402


def _memref(ptr, ndim, shape):
    strides = tuple(int(np.prod(shape[i + 1:])) for i in range(ndim))

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


def _compile_and_run(mlir, inputs, out_shape):
    with tempfile.TemporaryDirectory() as td:
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        sf.register_dialects(ctx._CAPIPtr, load=True)
        with ir.Location.unknown(ctx):
            mod = ir.Module.parse(mlir, ctx)
            pman = pm.PassManager.parse(
                "builtin.module({})".format(SF_LOWERING_PIPELINE), ctx
            )
            pman.enable_verifier(True)
            pman.run(mod.operation)
            lower_linalg_to_llvm_ir(mod)
            _fixup_unrealized_casts_pass(mod)

            m = os.path.join(td, "m.mlir")
            ll = os.path.join(td, "m.ll")
            o = os.path.join(td, "m.o")
            dylib = os.path.join(td, "test.dylib")

            with open(m, "w") as f:
                f.write(str(mod))

            subprocess.run(
                ["./llvm-project/build/bin/mlir-translate",
                 "--mlir-to-llvmir", m, "-o", ll],
                capture_output=True, check=True, timeout=60,
            )
            subprocess.run(
                [_find_llc(), "-filetype=obj", ll, "-o", o],
                capture_output=True, check=True, timeout=60,
            )
            free_o = _compile_serveforge_free(td)
            link_dylib([o, free_o], dylib)

            lib = ctypes.CDLL(dylib)
            mrs = [_memref(a.ctypes.data, a.ndim, a.shape) for a in inputs]
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret)] + [ctypes.byref(mr) for mr in mrs]
            lib._mlir_ciface_main_0.argtypes = [ctypes.c_void_p] * len(args)
            lib._mlir_ciface_main_0.restype = None
            lib._mlir_ciface_main_0(*args)

            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            rank = len(out_shape)
            sz = tuple(
                struct.unpack_from("<q", sb, 24 + 8 * i)[0]
                for i in range(rank)
            )
            n = int(np.prod(sz))
            return np.array(
                (ctypes.c_float * n).from_address(al), dtype=np.float32
            ).reshape(sz)


def _assert_cos(actual, expected, label):
    cos = cosine_similarity(actual, expected)
    mae = float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).mean())
    assert cos >= 0.9999, (
        "{}: cos={:.8f} < 0.9999, mae={:.2e}\n"
        "  dylib mean={:.6f} ref mean={:.6f}".format(
            label, cos, mae, actual.mean(), expected.mean())
    )


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestDylibMultiOpRegression:

    def test_embedding_plus_add_zero(self):
        batch, seq, vocab, hidden = 2, 4, 100, 64
        rng = np.random.RandomState(42)
        weight = rng.randn(vocab, hidden).astype(np.float32)
        indices = rng.randint(0, vocab, size=(batch, seq), dtype=np.int64)
        zero_buf = np.zeros((batch, seq, hidden), dtype=np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%ids: tensor<2x4xi64>, %emb_w: tensor<100x64xf32>, "
            "%zero: tensor<2x4x64xf32>) -> tensor<2x4x64xf32> {\n"
            '    %emb = "sf.embedding"(%emb_w, %ids) : '
            "(tensor<100x64xf32>, tensor<2x4xi64>) -> tensor<2x4x64xf32>\n"
            '    %1 = "sf.add"(%zero, %emb) : '
            "(tensor<2x4x64xf32>, tensor<2x4x64xf32>) -> tensor<2x4x64xf32>\n"
            "    return %1 : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [indices, weight, zero_buf], (batch, seq, hidden))
        expected = weight[indices % vocab]
        _assert_cos(actual, expected, "embedding+add")

    def test_embedding_plus_mul_identity(self):
        batch, seq, vocab, hidden = 2, 4, 100, 64
        rng = np.random.RandomState(42)
        weight = rng.randn(vocab, hidden).astype(np.float32)
        indices = rng.randint(0, vocab, size=(batch, seq), dtype=np.int64)
        ones = np.ones((batch, seq, hidden), dtype=np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%ids: tensor<2x4xi64>, %emb_w: tensor<100x64xf32>, "
            "%ones: tensor<2x4x64xf32>) -> tensor<2x4x64xf32> {\n"
            '    %emb = "sf.embedding"(%emb_w, %ids) : '
            "(tensor<100x64xf32>, tensor<2x4xi64>) -> tensor<2x4x64xf32>\n"
            '    %1 = "sf.mul"(%emb, %ones) : '
            "(tensor<2x4x64xf32>, tensor<2x4x64xf32>) -> tensor<2x4x64xf32>\n"
            "    return %1 : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [indices, weight, ones], (batch, seq, hidden))
        expected = weight[indices % vocab]
        _assert_cos(actual, expected, "embedding+mul")

    def test_rms_norm(self):
        batch, seq, hidden = 2, 4, 64
        rng = np.random.RandomState(42)
        x = rng.randn(batch, seq, hidden).astype(np.float32)
        nw = rng.randn(hidden).astype(np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%x: tensor<2x4x64xf32>, %w: tensor<64xf32>) "
            "-> tensor<2x4x64xf32> {\n"
            '    %0 = "sf.rms_norm"(%x, %w) : '
            "(tensor<2x4x64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>\n"
            "    return %0 : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [x, nw], (batch, seq, hidden))
        expected = (x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
        _assert_cos(actual, expected, "rms_norm")

    def test_linear(self):
        batch, seq, hidden, ffn_dim = 2, 4, 64, 256
        rng = np.random.RandomState(42)
        x = rng.randn(batch, seq, hidden).astype(np.float32)
        w = rng.randn(ffn_dim, hidden).astype(np.float32)
        b = rng.randn(ffn_dim).astype(np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%x: tensor<2x4x64xf32>, %w: tensor<256x64xf32>, "
            "%b: tensor<256xf32>) -> tensor<2x4x256xf32> {\n"
            '    %0 = "sf.linear"(%x, %w, %b) : '
            "(tensor<2x4x64xf32>, tensor<256x64xf32>, tensor<256xf32>) "
            "-> tensor<2x4x256xf32>\n"
            "    return %0 : tensor<2x4x256xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [x, w, b], (batch, seq, ffn_dim))
        expected = np.dot(x.reshape(-1, hidden), w.T) + b
        expected = expected.reshape(batch, seq, ffn_dim)
        _assert_cos(actual, expected, "linear")

    def test_silu(self):
        batch, seq, ffn_dim = 2, 4, 256
        rng = np.random.RandomState(42)
        x = rng.randn(batch, seq, ffn_dim).astype(np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%x: tensor<2x4x256xf32>) -> tensor<2x4x256xf32> {\n"
            '    %0 = "sf.silu"(%x) : '
            "(tensor<2x4x256xf32>) -> tensor<2x4x256xf32>\n"
            "    return %0 : tensor<2x4x256xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [x], (batch, seq, ffn_dim))
        expected = x / (1.0 + np.exp(-x))
        _assert_cos(actual, expected, "silu")

    def test_ffn_block(self):
        batch, seq, hidden, ffn_dim = 2, 4, 64, 256
        rng = np.random.RandomState(42)
        x = rng.randn(batch, seq, hidden).astype(np.float32)
        nw = rng.randn(hidden).astype(np.float32)
        w1 = rng.randn(ffn_dim, hidden).astype(np.float32)
        b1 = rng.randn(ffn_dim).astype(np.float32)
        w2 = rng.randn(hidden, ffn_dim).astype(np.float32)
        b2 = rng.randn(hidden).astype(np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%x: tensor<2x4x64xf32>, %nw: tensor<64xf32>, "
            "%w1: tensor<256x64xf32>, %b1: tensor<256xf32>, "
            "%w2: tensor<64x256xf32>, %b2: tensor<64xf32>) "
            "-> tensor<2x4x64xf32> {\n"
            '    %n = "sf.rms_norm"(%x, %nw) : '
            "(tensor<2x4x64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>\n"
            '    %fc1 = "sf.linear"(%n, %w1, %b1) : '
            "(tensor<2x4x64xf32>, tensor<256x64xf32>, tensor<256xf32>) "
            "-> tensor<2x4x256xf32>\n"
            '    %act = "sf.silu"(%fc1) : '
            "(tensor<2x4x256xf32>) -> tensor<2x4x256xf32>\n"
            '    %fc2 = "sf.linear"(%act, %w2, %b2) : '
            "(tensor<2x4x256xf32>, tensor<64x256xf32>, tensor<64xf32>) "
            "-> tensor<2x4x64xf32>\n"
            '    %out = "sf.add"(%fc2, %x) : '
            "(tensor<2x4x64xf32>, tensor<2x4x64xf32>) -> tensor<2x4x64xf32>\n"
            "    return %out : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [x, nw, w1, b1, w2, b2], (batch, seq, hidden))

        norm = (x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
        fc1 = np.dot(norm.reshape(-1, hidden), w1.T) + b1
        silu = fc1 / (1.0 + np.exp(-fc1))
        fc2 = np.dot(silu, w2.T) + b2
        expected = fc2 + x.reshape(-1, hidden)
        expected = expected.reshape(batch, seq, hidden)

        _assert_cos(actual, expected, "ffn_block")
