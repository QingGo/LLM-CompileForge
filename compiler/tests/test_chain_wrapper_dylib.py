"""Chain-wrapper dylib contract test: multi-function module → dylib ctypes call.

Validates that a multi-function sf dialect module compiled through the full
pipeline (sf→linalg→LLVM→dylib) produces correct output when called via
the chain-wrapper's _mlir_ciface_main entry point.

This tests the code path that the E2E uses — single _mlir_ciface_main call
covering all sub-functions — which was previously untested at unit level.
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


def _compile_and_run_main(mlir, inputs, out_shape):
    """Compile sf MLIR → dylib, call _mlir_ciface_main via ctypes."""
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
            lib._mlir_ciface_main.argtypes = [ctypes.c_void_p] * len(args)
            lib._mlir_ciface_main.restype = None
            lib._mlir_ciface_main(*args)

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
class TestChainWrapperDylib:
    def test_two_func_embedding_rms_norm_nonzero(self) -> None:
        """2-function chain (embedding → rms_norm) produces non-zero output.

        This is a minimal contract test: the chain-wrapper's _mlir_ciface_main
        must produce correct output (not all zeros) for a known input.
        """
        batch, seq, vocab, hidden = 2, 4, 100, 64
        rng = np.random.RandomState(42)

        emb_weight = rng.randn(vocab, hidden).astype(np.float32)
        rms_weight = rng.randn(hidden).astype(np.float32)
        input_ids = rng.randint(0, vocab, size=(batch, seq), dtype=np.int64)

        mlir = (
            'module attributes {sf.chain_order = ["main_0", "main_1"],\n'
            "                   sf.exec_plan_data = [2, 3,\n"
            "                     3, 0, 0, 0, 0, 1, 0, 0, 2, 0,\n"
            "                     2, 1, 0, 0, 1, 0, 1]} {\n"
            "  func.func @main_0(%ids: tensor<2x4xi64>, %emb: tensor<100x64xf32>, "
            "%rms: tensor<64xf32>) -> (tensor<2x4x64xf32>, tensor<64xf32>) {\n"
            '    %0 = "sf.embedding"(%emb, %ids) {num_buckets = 100 : i64} : '
            "(tensor<100x64xf32>, tensor<2x4xi64>) -> tensor<2x4x64xf32>\n"
            "    return %0, %rms : tensor<2x4x64xf32>, tensor<64xf32>\n"
            "  }\n"
            "  func.func @main_1(%hidden: tensor<2x4x64xf32>, %rms: tensor<64xf32>) "
            "-> tensor<2x4x64xf32> {\n"
            '    %0 = "sf.rms_norm"(%hidden, %rms) : '
            "(tensor<2x4x64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>\n"
            "    return %0 : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )

        actual = _compile_and_run_main(mlir, [input_ids, emb_weight, rms_weight], (batch, seq, hidden))

        # Reference: embedding → rms_norm
        embedded = emb_weight[input_ids % vocab]
        eps = 1e-5
        expected = (embedded / np.sqrt((embedded**2).mean(axis=-1, keepdims=True) + eps)) * rms_weight

        _assert_cos(actual, expected, "embedding→rms_norm chain")

        # Smoke: output must not be all zeros
        assert np.abs(actual).max() > 0.0, "Chain wrapper produced all zeros — likely LLVM lowering bug"

        # Verify each position's output is non-zero
        for b in range(batch):
            for s in range(seq):
                pos_abs = np.abs(actual[b, s]).max()
                assert pos_abs > 0.0, f"Position [{b},{s}] is all zeros (abs_max={pos_abs})"


def _build_n_func_chain(n_funcs: int) -> str:
    """Build MLIR for an N-function chain: main_0 embeds, rest do rms_norm.

    main_0:  (input_ids, emb_weight, rms_weight) → (hidden, rms_weight)
    main_i:  (hidden, rms_weight) → rms_norm(hidden, rms_weight)
    """
    lines: list[str] = []
    # exec_plan_data
    plan = [n_funcs, 3]  # num_steps, num_global
    # step0: 3 GLOBAL_INPUT edges
    plan.extend([3, 0, 0, 0, 0, 1, 0, 0, 2, 0])
    # step1..stepN-1: 2 edges each (hidden from prev, rms_weight from step0.out1)
    for i in range(1, n_funcs):
        plan.extend([2, 1, i - 1, 0, 1, 0, 1])
    plan_str = ", ".join(str(x) for x in plan)

    chain_names = ", ".join(f'"{n}"' for n in [f"main_{i}" for i in range(n_funcs)])
    lines.append(
        f"module attributes {{sf.chain_order = [{chain_names}],\n"
        f"                   sf.exec_plan_data = [{plan_str}]}} {{"
    )
    lines.append(
        "  func.func @main_0(%ids: tensor<2x4xi64>, %emb: tensor<100x64xf32>, "
        "%rms: tensor<64xf32>) -> (tensor<2x4x64xf32>, tensor<64xf32>) {"
    )
    lines.append(
        '    %0 = "sf.embedding"(%emb, %ids) {num_buckets = 100 : i64} : '
        "(tensor<100x64xf32>, tensor<2x4xi64>) -> tensor<2x4x64xf32>"
    )
    lines.append("    return %0, %rms : tensor<2x4x64xf32>, tensor<64xf32>")
    lines.append("  }")
    for i in range(1, n_funcs):
        lines.append(f"  func.func @main_{i}(%h: tensor<2x4x64xf32>, %w: tensor<64xf32>) -> tensor<2x4x64xf32> {{")
        lines.append('    %0 = "sf.rms_norm"(%h, %w) : (tensor<2x4x64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>')
        lines.append("    return %0 : tensor<2x4x64xf32>")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


@pytest.mark.integration
@pytest.mark.timeout(180)
class TestChainWrapperDylibScaling:
    """Verify N-function chains work correctly — find the breaking point."""

    @pytest.mark.parametrize("n_funcs", [3, 4, 6, 10, 16])
    def test_n_func_chain_nonzero(self, n_funcs: int) -> None:
        """N-function rms_norm chain produces non-zero output."""
        batch, seq, vocab, hidden = 2, 4, 100, 64
        rng = np.random.RandomState(42 + n_funcs)

        emb_weight = rng.randn(vocab, hidden).astype(np.float32)
        rms_weight = rng.randn(hidden).astype(np.float32)
        input_ids = rng.randint(0, vocab, size=(batch, seq), dtype=np.int64)

        mlir = _build_n_func_chain(n_funcs)
        actual = _compile_and_run_main(mlir, [input_ids, emb_weight, rms_weight], (batch, seq, hidden))

        eps = 1e-5
        embedded = emb_weight[input_ids % vocab]
        expected = embedded
        for _ in range(n_funcs - 1):
            expected = (expected / np.sqrt((expected**2).mean(axis=-1, keepdims=True) + eps)) * rms_weight

        _assert_cos(actual, expected, f"{n_funcs}-func rms_norm chain")
        assert np.abs(actual).max() > 0.0, f"{n_funcs}-func chain produced all zeros"


def _build_n_func_chain_with_weights(n_funcs: int) -> str:
    """Build MLIR with sf.weight ops (pre-promotion form).

    Uses sf.weight ops that the sf-promote-weights pass will convert to
    function arguments. This tests the full pipeline including promotion.
    """
    lines = ["module {"]
    # main_0: embedding using sf.weight + chain_order / exec_plan_data added later
    lines.append("  func.func @main_0(%ids: tensor<2x4xi64>) -> (tensor<2x4x64xf32>, tensor<64xf32>) {")
    lines.append('    %emb = "sf.weight"() {name = "emb", type = tensor<100x64xf32>} : () -> tensor<100x64xf32>')
    lines.append('    %rms = "sf.weight"() {name = "rms", type = tensor<64xf32>} : () -> tensor<64xf32>')
    lines.append('    %0 = "sf.embedding"(%emb, %ids) {num_buckets = 100 : i64} : ')
    lines.append("        (tensor<100x64xf32>, tensor<2x4xi64>) -> tensor<2x4x64xf32>")
    lines.append("    return %0, %rms : tensor<2x4x64xf32>, tensor<64xf32>")
    lines.append("  }")
    for i in range(1, n_funcs):
        lines.append(f"  func.func @main_{i}(%h: tensor<2x4x64xf32>, %w: tensor<64xf32>) -> tensor<2x4x64xf32> {{")
        lines.append('    %0 = "sf.rms_norm"(%h, %w) : ')
        lines.append("        (tensor<2x4x64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>")
        lines.append("    return %0 : tensor<2x4x64xf32>")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _add_exec_plan_attrs(mlir_text: str, n_funcs: int) -> str:
    """Insert sf.chain_order and sf.exec_plan_data into the module line."""
    chain_names = ", ".join(f'"{n}"' for n in [f"main_{i}" for i in range(n_funcs)])
    plan = [n_funcs, 3]
    plan.extend([3, 0, 0, 0, 0, 1, 0, 0, 2, 0])
    for i in range(1, n_funcs):
        plan.extend([2, 1, i - 1, 0, 1, 0, 1])
    plan_str = ", ".join(str(x) for x in plan)
    attrs = f"module attributes {{sf.chain_order = [{chain_names}], sf.exec_plan_data = [{plan_str}]}} {{"
    return mlir_text.replace("module {", attrs, 1)


@pytest.mark.integration
@pytest.mark.timeout(180)
class TestChainWrapperWithPromotion:
    """Chain-wrapper after sf-promote-weights (matching real pipeline order)."""

    @pytest.mark.parametrize("n_funcs", [3, 10, 16])
    def test_n_func_with_promotion_nonzero(self, n_funcs: int) -> None:
        """N-function chain with weight promotion produces non-zero output."""
        batch, seq, vocab, hidden = 2, 4, 100, 64
        rng = np.random.RandomState(100 + n_funcs)

        emb_weight = rng.randn(vocab, hidden).astype(np.float32)
        rms_weight = rng.randn(hidden).astype(np.float32)
        input_ids = rng.randint(0, vocab, size=(batch, seq), dtype=np.int64)

        mlir_base = _build_n_func_chain_with_weights(n_funcs)
        mlir = _add_exec_plan_attrs(mlir_base, n_funcs)

        actual = _compile_and_run_main(mlir, [input_ids, emb_weight, rms_weight], (batch, seq, hidden))

        eps = 1e-5
        embedded = emb_weight[input_ids % vocab]
        expected = embedded
        for _ in range(n_funcs - 1):
            expected = (expected / np.sqrt((expected**2).mean(axis=-1, keepdims=True) + eps)) * rms_weight

        _assert_cos(actual, expected, f"{n_funcs}-func rms_norm chain (with promotion)")
        assert np.abs(actual).max() > 0.0, f"{n_funcs}-func chain (with promotion) produced all zeros"


def _build_many_global_inputs_mlir(n_weights: int) -> str:
    """Build MLIR with many promoted weights as function arguments."""
    w_names = [f"w{i}" for i in range(n_weights)]

    lines = []
    lines.append("module {")

    # Build main_0 arg list: input_ids + w0(embedding weight) + w1..wN(rms weights)
    args = (
        ["%ids: tensor<2x4xi64>"]
        + [
            "%w0: tensor<100x64xf32>"  # embedding weight (vocab=100, hidden=64)
        ]
        + [f"%{n}: tensor<64xf32>" for n in w_names[1:]]
    )
    rms_types = ["tensor<64xf32>" for _ in range(n_weights - 1)]
    ret_types = ["tensor<2x4x64xf32>", "tensor<100x64xf32>"] + rms_types
    ret_str = ",\n      ".join(ret_types)
    ret_names = "%emb, %w0, " + ", ".join(f"%{n}" for n in w_names[1:])

    lines.append("  func.func @main_0(")
    for i, arg in enumerate(args):
        comma = "," if i < len(args) - 1 else ""
        lines.append(f"      {arg}{comma}")
    lines.append(f"  ) -> ({ret_str}) {{")

    lines.append('    %emb = "sf.embedding"(%w0, %ids) {num_buckets = 100 : i64} : ')
    lines.append("        (tensor<100x64xf32>, tensor<2x4xi64>) -> tensor<2x4x64xf32>")
    lines.append(f"    return {ret_names} :")
    for i, rt in enumerate(ret_types):
        comma = "," if i < len(ret_types) - 1 else ""
        lines.append(f"      {rt}{comma}")
    lines.append("  }")
    lines.append("  func.func @main_1(%h: tensor<2x4x64xf32>, %w: tensor<64xf32>) ")
    lines.append("      -> tensor<2x4x64xf32> {")
    lines.append('    %0 = "sf.rms_norm"(%h, %w) : ')
    lines.append("        (tensor<2x4x64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>")
    lines.append("    return %0 : tensor<2x4x64xf32>")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)
    lines.append("  func.func @main_1(%h: tensor<2x4x64xf32>, %w: tensor<64xf32>) ")
    lines.append("      -> tensor<2x4x64xf32> {")
    lines.append('    %0 = "sf.rms_norm"(%h, %w) : ')
    lines.append("        (tensor<2x4x64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>")
    lines.append("    return %0 : tensor<2x4x64xf32>")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _compile_and_run_main_with_plan(
    mlir_text: str,
    inputs: list,
    out_shape: tuple,
    exec_plan_data: list[int] | None = None,
    chain_order: list[str] | None = None,
) -> np.ndarray:
    """Like _compile_and_run_main but attaches exec_plan_data programmatically.

    Avoids MLIR text line-length limits for large exec_plan_data arrays.
    """
    import os
    import struct
    import subprocess
    import tempfile

    from compiler.backend.compile_utils import _compile_serveforge_free, _find_llc, link_dylib
    from compiler.backend.fixups import _fixup_unrealized_casts_pass
    from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir
    from compiler.pipeline.lowering import SF_LOWERING_PIPELINE
    from compiler.dylib_ffi import DEFAULT_SRET_SIZE

    with tempfile.TemporaryDirectory() as td:
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        sf.register_dialects(ctx._CAPIPtr, load=True)
        with ir.Location.unknown(ctx):
            mod = ir.Module.parse(mlir_text, ctx)

            # Attach exec_plan_data and chain_order programmatically
            if exec_plan_data:
                int_attrs = [ir.IntegerAttr.get(ir.IntegerType.get_signless(64), v) for v in exec_plan_data]
                mod.operation.attributes["sf.exec_plan_data"] = ir.ArrayAttr.get(int_attrs)
            if chain_order:
                name_attrs = [ir.StringAttr.get(n) for n in chain_order]
                mod.operation.attributes["sf.chain_order"] = ir.ArrayAttr.get(name_attrs)

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
            lib._mlir_ciface_main.argtypes = [ctypes.c_void_p] * len(args)
            lib._mlir_ciface_main.restype = None
            lib._mlir_ciface_main(*args)

            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            rank = len(out_shape)
            sz = tuple(struct.unpack_from("<q", sb, 24 + 8 * i)[0] for i in range(rank))
            n = int(np.prod(sz))
            return np.array((ctypes.c_float * n).from_address(al), dtype=np.float32).reshape(sz)


def _make_many_globals_plan(n_weights: int) -> tuple[list[int], list[str]]:
    """Build exec_plan_data and chain_order for many-global-inputs test."""
    chain_order = ["main_0", "main_1"]
    n_global = n_weights + 1  # weights + input_ids
    plan = [2, n_global]
    plan.append(n_global)
    for i in range(n_global):
        plan.extend([0, i, 0])
    plan.extend([2, 1, 0, 0, 1, 0, 2])  # hidden from step0.out0, rms_weight from step0.out2
    return plan, chain_order


@pytest.mark.integration
@pytest.mark.timeout(180)
class TestChainWrapperRealisticOps:
    """Chain-wrapper with realistic op sequences (FFN blocks, multi-output)."""

    def test_two_ffn_block_chain_nonzero(self) -> None:
        """2-function chain of FFN blocks (matching real transformer layers)."""
        batch, seq, hidden, ffn_dim = 2, 4, 64, 256
        rng = np.random.RandomState(500)

        x = rng.randn(batch, seq, hidden).astype(np.float32)
        nw0 = rng.randn(hidden).astype(np.float32)
        w1_0 = rng.randn(ffn_dim, hidden).astype(np.float32)
        b1_0 = rng.randn(ffn_dim).astype(np.float32)
        w2_0 = rng.randn(hidden, ffn_dim).astype(np.float32)
        b2_0 = rng.randn(hidden).astype(np.float32)
        nw1 = rng.randn(hidden).astype(np.float32)
        w1_1 = rng.randn(ffn_dim, hidden).astype(np.float32)
        b1_1 = rng.randn(ffn_dim).astype(np.float32)
        w2_1 = rng.randn(hidden, ffn_dim).astype(np.float32)
        b2_1 = rng.randn(hidden).astype(np.float32)

        # 2 functions: main_0 does FFN and exports (hidden + 5 weights),
        # main_1 does FFN with the same weights.
        # 6 global inputs: x + nw0 + w1_0 + b1_0 + w2_0 + b2_0
        mlir = (
            'module attributes {sf.chain_order = ["main_0", "main_1"],\n'
            "                   sf.exec_plan_data = [2, 6,\n"
            "                     6, 0,0,0, 0,1,0, 0,2,0, 0,3,0, 0,4,0, 0,5,0,\n"
            "                     6, 1,0,0, 1,0,1, 1,0,2, 1,0,3, 1,0,4, 1,0,5]} {\n"
            "  func.func @main_0(%x: tensor<2x4x64xf32>,\n"
            "      %nw: tensor<64xf32>, %w1: tensor<256x64xf32>,\n"
            "      %b1: tensor<256xf32>, %w2: tensor<64x256xf32>,\n"
            "      %b2: tensor<64xf32>)\n"
            "      -> (tensor<2x4x64xf32>, tensor<64xf32>, tensor<256x64xf32>,\n"
            "          tensor<256xf32>, tensor<64x256xf32>, tensor<64xf32>) {\n"
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
            "    return %out, %nw, %w1, %b1, %w2, %b2 : "
            "tensor<2x4x64xf32>, tensor<64xf32>, tensor<256x64xf32>, "
            "tensor<256xf32>, tensor<64x256xf32>, tensor<64xf32>\n"
            "  }\n"
            "  func.func @main_1(%h: tensor<2x4x64xf32>,\n"
            "      %nw: tensor<64xf32>, %w1: tensor<256x64xf32>,\n"
            "      %b1: tensor<256xf32>, %w2: tensor<64x256xf32>,\n"
            "      %b2: tensor<64xf32>)\n"
            "      -> tensor<2x4x64xf32> {\n"
            '    %n = "sf.rms_norm"(%h, %nw) : '
            "(tensor<2x4x64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>\n"
            '    %fc1 = "sf.linear"(%n, %w1, %b1) : '
            "(tensor<2x4x64xf32>, tensor<256x64xf32>, tensor<256xf32>) "
            "-> tensor<2x4x256xf32>\n"
            '    %act = "sf.silu"(%fc1) : '
            "(tensor<2x4x256xf32>) -> tensor<2x4x256xf32>\n"
            '    %fc2 = "sf.linear"(%act, %w2, %b2) : '
            "(tensor<2x4x256xf32>, tensor<64x256xf32>, tensor<64xf32>) "
            "-> tensor<2x4x64xf32>\n"
            '    %out = "sf.add"(%fc2, %h) : '
            "(tensor<2x4x64xf32>, tensor<2x4x64xf32>) -> tensor<2x4x64xf32>\n"
            "    return %out : tensor<2x4x64xf32>\n"
            "  }\n"
            "}\n"
        )

        all_inputs = [x, nw0, w1_0, b1_0, w2_0, b2_0]
        actual = _compile_and_run_main(mlir, all_inputs, (batch, seq, hidden))

        eps = 1e-5
        norm0 = (x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)) * nw0
        fc1_0 = np.dot(norm0.reshape(-1, hidden), w1_0.T) + b1_0
        silu_0 = fc1_0 / (1.0 + np.exp(-fc1_0))
        fc2_0 = np.dot(silu_0, w2_0.T) + b2_0
        after_layer0 = fc2_0 + x.reshape(-1, hidden)

        norm1 = (after_layer0 / np.sqrt((after_layer0**2).mean(axis=-1, keepdims=True) + eps)) * nw0
        fc1_1 = np.dot(norm1, w1_0.T) + b1_0
        silu_1 = fc1_1 / (1.0 + np.exp(-fc1_1))
        fc2_1 = np.dot(silu_1, w2_0.T) + b2_0
        expected = fc2_1 + after_layer0
        expected = expected.reshape(batch, seq, hidden)

        _assert_cos(actual, expected, "2×FFN block chain")
        assert np.abs(actual).max() > 0.0, "2×FFN block chain produced all zeros"

    """Test chain-wrapper with many global inputs (matching real model scale)."""

    @pytest.mark.parametrize("n_weights", [10, 50, 100, 198])
    def test_many_global_inputs_nonzero(self, n_weights: int) -> None:
        """2-function chain with N global weight inputs produces non-zero output."""
        batch, seq, hidden = 2, 4, 64
        rng = np.random.RandomState(200 + n_weights)

        input_ids = np.zeros((batch, seq), dtype=np.int64)
        vocab = 100
        emb_weight = rng.randn(vocab, hidden).astype(np.float32)
        rms_weights = [rng.randn(hidden).astype(np.float32) for _ in range(n_weights - 1)]

        mlir = _build_many_global_inputs_mlir(n_weights)
        plan, chain_order = _make_many_globals_plan(n_weights)

        all_inputs = [input_ids, emb_weight] + rms_weights
        actual = _compile_and_run_main_with_plan(
            mlir, all_inputs, (batch, seq, hidden), exec_plan_data=plan, chain_order=chain_order
        )

        eps = 1e-5
        embedded = emb_weight[np.zeros((batch, seq), dtype=np.int64) % vocab]
        expected = (embedded / np.sqrt((embedded**2).mean(axis=-1, keepdims=True) + eps)) * rms_weights[0]

        _assert_cos(actual, expected, f"2-func {n_weights}-global chain")
        assert np.abs(actual).max() > 0.0, f"2-func {n_weights}-global chain produced all zeros"
