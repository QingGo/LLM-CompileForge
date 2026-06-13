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
from compiler.dylib_ffi import DEFAULT_SRET_SIZE
from scripts._cos import cosine_similarity

_setup_mlir_path()
import mlir.ir as ir  # noqa: E402
import mlir.passmanager as pm  # noqa: E402
from mlir_sf._mlir_libs._sfDialectsNanobind import sf  # noqa: E402


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


def _compile_and_run(mlir, inputs, out_shape):
    with tempfile.TemporaryDirectory() as td:
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        sf.register_dialects(ctx._CAPIPtr, load=True)
        with ir.Location.unknown(ctx):
            mod = ir.Module.parse(mlir, ctx)
            pman = pm.PassManager.parse(f"builtin.module({SF_LOWERING_PIPELINE})", ctx)
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
                ["./llvm-project/build/bin/mlir-translate", "--mlir-to-llvmir", m, "-o", ll],
                capture_output=True,
                check=True,
                timeout=60,
            )
            subprocess.run(
                [_find_llc(), "-filetype=obj", ll, "-o", o],
                capture_output=True,
                check=True,
                timeout=60,
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
            sz = tuple(struct.unpack_from("<q", sb, 24 + 8 * i)[0] for i in range(rank))
            n = int(np.prod(sz))
            return np.array((ctypes.c_float * n).from_address(al), dtype=np.float32).reshape(sz)


def _assert_cos(actual, expected, label):
    cos = cosine_similarity(actual, expected)
    mae = float(np.abs(actual.astype(np.float64) - expected.astype(np.float64)).mean())
    assert cos >= 0.9999, (
        f"{label}: cos={cos:.8f} < 0.9999, mae={mae:.2e}\n"
        f"  dylib mean={actual.mean():.6f} ref mean={expected.mean():.6f}"
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
        expected = (x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
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

        norm = (x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-6)) * nw
        fc1 = np.dot(norm.reshape(-1, hidden), w1.T) + b1
        silu = fc1 / (1.0 + np.exp(-fc1))
        fc2 = np.dot(silu, w2.T) + b2
        expected = fc2 + x.reshape(-1, hidden)
        expected = expected.reshape(batch, seq, hidden)

        _assert_cos(actual, expected, "ffn_block")

    def test_scaled_dot_product_attention(self):
        batch, heads, seq, d_k = 1, 2, 4, 8
        rng = np.random.RandomState(99)
        q = rng.randn(batch, heads, seq, d_k).astype(np.float32)
        k = rng.randn(batch, heads, seq, d_k).astype(np.float32)
        v = rng.randn(batch, heads, seq, d_k).astype(np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%q: tensor<1x2x4x8xf32>,"
            " %k: tensor<1x2x4x8xf32>, %v: tensor<1x2x4x8xf32>)"
            " -> tensor<1x2x4x8xf32> {\n"
            '    %0 = "sf.scaled_dot_product_attention"(%q, %k, %v) : '
            "(tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>)"
            " -> tensor<1x2x4x8xf32>\n"
            "    return %0 : tensor<1x2x4x8xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [q, k, v], (batch, heads, seq, d_k))

        scale = 1.0 / np.sqrt(d_k)
        attn = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn_max = attn.max(axis=-1, keepdims=True)
        attn = np.exp(attn - attn_max)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        expected = np.matmul(attn, v)

        _assert_cos(actual, expected, "scaled_dot_product_attention")

    def test_layer_norm(self):
        batch, seq, hidden = 2, 4, 64
        rng = np.random.RandomState(77)
        x = rng.randn(batch, seq, hidden).astype(np.float32)
        weight = rng.randn(hidden).astype(np.float32)
        bias = rng.randn(hidden).astype(np.float32)
        eps = 1e-5

        mlir = (
            "module {\n"
            "  func.func @main_0(%x: tensor<2x4x64xf32>, %w: tensor<64xf32>,"
            " %b: tensor<64xf32>) -> tensor<2x4x64xf32> {\n"
            '    %0 = "sf.layer_norm"(%x, %w, %b) {axis = 2 : i64, eps = 1.0e-5 : f64} : '
            "(tensor<2x4x64xf32>, tensor<64xf32>, tensor<64xf32>)"
            " -> tensor<2x4x64xf32>\n"
            "    return %0 : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [x, weight, bias], (batch, seq, hidden))

        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + eps) * weight + bias

        _assert_cos(actual, expected, "layer_norm")

    @pytest.mark.xfail(reason="sf.softmax is not lowered by sf-lower-to-linalg (softmax is handled inside SDPA decomposition)")
    def test_softmax(self):
        batch, seq, hidden = 2, 4, 64
        rng = np.random.RandomState(55)
        x = rng.randn(batch, seq, hidden).astype(np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%x: tensor<2x4x64xf32>) -> tensor<2x4x64xf32> {\n"
            '    %0 = "sf.softmax"(%x) : '
            "(tensor<2x4x64xf32>) -> tensor<2x4x64xf32>\n"
            "    return %0 : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [x], (batch, seq, hidden))

        x_max = x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        expected = exp_x / exp_x.sum(axis=-1, keepdims=True)

        _assert_cos(actual, expected, "softmax")

    def test_transpose(self):
        batch, m, n = 2, 4, 8
        rng = np.random.RandomState(33)
        x = rng.randn(batch, m, n).astype(np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%x: tensor<2x4x8xf32>) -> tensor<2x8x4xf32> {\n"
            '    %0 = "sf.transpose"(%x) {dim0 = 1 : i64, dim1 = 2 : i64} : '
            "(tensor<2x4x8xf32>) -> tensor<2x8x4xf32>\n"
            "    return %0 : tensor<2x8x4xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [x], (batch, n, m))
        expected = x.transpose(0, 2, 1)
        _assert_cos(actual, expected, "transpose")

    def test_mul(self):
        batch, seq, hidden = 2, 4, 64
        rng = np.random.RandomState(44)
        a = rng.randn(batch, seq, hidden).astype(np.float32)
        b = rng.randn(batch, seq, hidden).astype(np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%a: tensor<2x4x64xf32>, %b: tensor<2x4x64xf32>)"
            " -> tensor<2x4x64xf32> {\n"
            '    %0 = "sf.mul"(%a, %b) : '
            "(tensor<2x4x64xf32>, tensor<2x4x64xf32>) -> tensor<2x4x64xf32>\n"
            "    return %0 : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [a, b], (batch, seq, hidden))
        expected = a * b
        _assert_cos(actual, expected, "mul")

    def test_relu(self):
        batch, seq, hidden = 2, 4, 64
        rng = np.random.RandomState(66)
        x = rng.randn(batch, seq, hidden).astype(np.float32)

        mlir = (
            "module {\n"
            "  func.func @main_0(%x: tensor<2x4x64xf32>) -> tensor<2x4x64xf32> {\n"
            '    %0 = "sf.relu"(%x) : '
            "(tensor<2x4x64xf32>) -> tensor<2x4x64xf32>\n"
            "    return %0 : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )
        actual = _compile_and_run(mlir, [x], (batch, seq, hidden))
        expected = np.maximum(x, 0)
        _assert_cos(actual, expected, "relu")

    def test_mini_transformer_layer(self):
        """Full transformer layer: emb + ln + linear + SDPA(4D) + silu + add."""
        batch, seq, hidden, vocab = 2, 4, 64, 100
        rng = np.random.RandomState(123)
        emb_w = rng.randn(vocab, hidden).astype(np.float32)
        ids = rng.randint(0, vocab, size=(batch, seq), dtype=np.int64)

        # Use 4D SDPA directly (1 head, d_k=hidden)
        ln1_w = rng.randn(hidden).astype(np.float32); ln1_b = rng.randn(hidden).astype(np.float32)
        q_w = rng.randn(hidden, hidden).astype(np.float32)
        k_w = rng.randn(hidden, hidden).astype(np.float32)
        v_w = rng.randn(hidden, hidden).astype(np.float32)
        o_w = rng.randn(hidden, hidden).astype(np.float32)
        ln2_w = rng.randn(hidden).astype(np.float32); ln2_b = rng.randn(hidden).astype(np.float32)
        ffn = 4 * hidden
        fc1_w = rng.randn(ffn, hidden).astype(np.float32); fc1_b = rng.randn(ffn).astype(np.float32)
        fc2_w = rng.randn(hidden, ffn).astype(np.float32); fc2_b = rng.randn(hidden).astype(np.float32)

        all_inputs = [ids, emb_w, ln1_w, ln1_b, q_w, k_w, v_w, o_w,
                      ln2_w, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b]

        # SDPA with 4D tensors (B, 1, S, H) — unsqueeze then sf.view to squeeze back
        mlir = (
            "module {\n"
            "  func.func @main_0(%ids: tensor<2x4xi64>,\n"
            "      %emb: tensor<100x64xf32>,\n"
            "      %ln1w: tensor<64xf32>, %ln1b: tensor<64xf32>,\n"
            "      %qw: tensor<64x64xf32>, %kw: tensor<64x64xf32>,\n"
            "      %vw: tensor<64x64xf32>, %ow: tensor<64x64xf32>,\n"
            "      %ln2w: tensor<64xf32>, %ln2b: tensor<64xf32>,\n"
            "      %fc1w: tensor<256x64xf32>, %fc1b: tensor<256xf32>,\n"
            "      %fc2w: tensor<64x256xf32>, %fc2b: tensor<64xf32>)\n"
            "      -> tensor<2x4x64xf32> {\n"
            '    %h = "sf.embedding"(%emb, %ids) {num_buckets = 100 : i64} : '
            "(tensor<100x64xf32>, tensor<2x4xi64>) -> tensor<2x4x64xf32>\n"
            '    %ln1 = "sf.layer_norm"(%h, %ln1w, %ln1b) '
            "{axis = 2 : i64, eps = 1.0e-5 : f64} : "
            "(tensor<2x4x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>\n"
            '    %q = "sf.linear"(%ln1, %qw) {use_bias = false} : '
            "(tensor<2x4x64xf32>, tensor<64x64xf32>) -> tensor<2x4x64xf32>\n"
            '    %k = "sf.linear"(%ln1, %kw) {use_bias = false} : '
            "(tensor<2x4x64xf32>, tensor<64x64xf32>) -> tensor<2x4x64xf32>\n"
            '    %v = "sf.linear"(%ln1, %vw) {use_bias = false} : '
            "(tensor<2x4x64xf32>, tensor<64x64xf32>) -> tensor<2x4x64xf32>\n"
            # SDPA with explicit 4D shapes via unsqueeze
            '    %q4 = "sf.unsqueeze"(%q) {dim = 1 : i64} : '
            "(tensor<2x4x64xf32>) -> tensor<2x1x4x64xf32>\n"
            '    %k4 = "sf.unsqueeze"(%k) {dim = 1 : i64} : '
            "(tensor<2x4x64xf32>) -> tensor<2x1x4x64xf32>\n"
            '    %v4 = "sf.unsqueeze"(%v) {dim = 1 : i64} : '
            "(tensor<2x4x64xf32>) -> tensor<2x1x4x64xf32>\n"
            '    %attn = "sf.scaled_dot_product_attention"(%q4, %k4, %v4) : '
            "(tensor<2x1x4x64xf32>, tensor<2x1x4x64xf32>, tensor<2x1x4x64xf32>) "
            "-> tensor<2x1x4x64xf32>\n"
            # Squeeze SDPA output from (2,1,4,64) → (2,4,64) via sf.view
            '    %s = "sf.view"(%attn) {shape = [2, 4, 64]} : '
            "(tensor<2x1x4x64xf32>) -> tensor<2x4x64xf32>\n"
            '    %attn_o = "sf.linear"(%s, %ow) {use_bias = false} : '
            "(tensor<2x4x64xf32>, tensor<64x64xf32>) -> tensor<2x4x64xf32>\n"
            '    %r1 = "sf.add"(%h, %attn_o) : '
            "(tensor<2x4x64xf32>, tensor<2x4x64xf32>) -> tensor<2x4x64xf32>\n"
            '    %ln2 = "sf.layer_norm"(%r1, %ln2w, %ln2b) '
            "{axis = 2 : i64, eps = 1.0e-5 : f64} : "
            "(tensor<2x4x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>\n"
            '    %fc1 = "sf.linear"(%ln2, %fc1w, %fc1b) : '
            "(tensor<2x4x64xf32>, tensor<256x64xf32>, tensor<256xf32>) -> tensor<2x4x256xf32>\n"
            '    %act = "sf.silu"(%fc1) : '
            "(tensor<2x4x256xf32>) -> tensor<2x4x256xf32>\n"
            '    %fc2 = "sf.linear"(%act, %fc2w, %fc2b) : '
            "(tensor<2x4x256xf32>, tensor<64x256xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>\n"
            '    %out = "sf.add"(%r1, %fc2) : '
            "(tensor<2x4x64xf32>, tensor<2x4x64xf32>) -> tensor<2x4x64xf32>\n"
            "    return %out : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )

        actual = _compile_and_run(mlir, all_inputs, (batch, seq, hidden))

        eps = 1e-5
        h = emb_w[ids % vocab]
        m1 = h.mean(axis=-1, keepdims=True)
        v1 = ((h - m1) ** 2).mean(axis=-1, keepdims=True)
        ln1 = (h - m1) / np.sqrt(v1 + eps) * ln1_w + ln1_b
        q = ln1.reshape(-1, hidden).dot(q_w.T)
        k = ln1.reshape(-1, hidden).dot(k_w.T)
        v = ln1.reshape(-1, hidden).dot(v_w.T)
        # SDPA
        q4 = q.reshape(batch, 1, seq, hidden)
        k4 = k.reshape(batch, 1, seq, hidden)
        v4 = v.reshape(batch, 1, seq, hidden)
        scale = 1.0 / np.sqrt(hidden)
        aw = np.matmul(q4, k4.transpose(0, 1, 3, 2)) * scale
        aw = np.exp(aw - aw.max(axis=-1, keepdims=True))
        aw = aw / aw.sum(axis=-1, keepdims=True)
        attn = np.matmul(aw, v4).reshape(batch, seq, hidden)
        attn_o = attn.reshape(-1, hidden).dot(o_w.T).reshape(batch, seq, hidden)
        r1 = h + attn_o
        m2 = r1.mean(axis=-1, keepdims=True)
        v2 = ((r1 - m2) ** 2).mean(axis=-1, keepdims=True)
        ln2 = (r1 - m2) / np.sqrt(v2 + eps) * ln2_w + ln2_b
        fc1 = ln2.reshape(-1, hidden).dot(fc1_w.T) + fc1_b
        act = fc1 / (1.0 + np.exp(-fc1))
        fc2 = act.dot(fc2_w.T) + fc2_b
        expected = r1 + fc2.reshape(batch, seq, hidden)

        _assert_cos(actual, expected, "mini_transformer_layer")
