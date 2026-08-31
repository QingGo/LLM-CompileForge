"""TDD RED test: sf.embedding + sf.weight promotion → bufferize corruption.

Uses the REAL sf-promote-weights path (sf.weight ops → promoted to args)
to reproduce the full-model embedding corruption.
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


def _build_sf_mlir_with_weights(
    vocab: int,
    hidden: int,
    num_extra_weights: int,
) -> tuple[str, list[str], list[tuple[int, ...]]]:
    """Build sf dialect MLIR with sf.weight ops (promoted by sf-promote-weights)."""
    lines = ["module {"]
    ret_types = []
    ret_vals = []

    # Token embedding weight
    weight_names = ["tok_embed_weight"]
    weight_shapes = [(vocab, hidden)]
    lines.append(
        f'    %tok_embed_weight = "sf.weight"() {{name = "tok_embed_weight"}} : () -> tensor<{vocab}x{hidden}xf32>'
    )

    # Extra weights
    for i in range(num_extra_weights):
        if i % 4 == 0:
            s = (hidden, hidden)
        elif i % 4 == 1:
            s = (hidden * 4, hidden)
        elif i % 4 == 2:
            s = (hidden,)
        else:
            s = (hidden, hidden * 4)
        name = f"w{i}"
        weight_names.append(name)
        weight_shapes.append(s)
        shape_str = "x".join(str(d) for d in s)
        lines.append(f'    %{name} = "sf.weight"() {{name = "{name}"}} : () -> tensor<{shape_str}xf32>')

    # All weights are also returned as identity copies
    for name, s in zip(weight_names, weight_shapes, strict=True):
        shape_str = "x".join(str(d) for d in s)
        ret_types.append(f"tensor<{shape_str}xf32>")
        ret_vals.append(f"%{name}")

    # Embedding op
    lines.append(
        f'    %emb = "sf.embedding"(%tok_embed_weight, %ids) : '
        f"(tensor<{vocab}x{hidden}xf32>, tensor<2x4xi64>) -> tensor<2x4x{hidden}xf32>"
    )

    emb_type = f"tensor<2x4x{hidden}xf32>"
    ret_types.insert(0, emb_type)  # emb output first
    ret_vals.insert(0, "%emb")

    lines.append(f"    return {', '.join(ret_vals)} : {', '.join(ret_types)}")
    lines.append("  }")
    lines.append("}")

    # Build function signature with correct return types
    ret_type_str = ", ".join(ret_types)
    func_sig = f"  func.func @main_0(%ids: tensor<2x4xi64>) -> ({ret_type_str}) {{"
    lines.insert(1, func_sig)
    return "\n".join(lines), weight_names, weight_shapes


def _compile_sf_to_dylib(mlir_text: str, tmp_dir: str, name: str) -> str:
    """Compile sf MLIR → dylib (used by baseline test)."""
    from compiler.pipeline import _apply_sf_to_linalg

    lowered = _apply_sf_to_linalg(mlir_text)
    return _compile_lowered_to_dylib(lowered, tmp_dir, name)


def _compile_lowered_to_dylib(lowered_mlir: str, tmp_dir: str, name: str) -> str:
    """Compile already-lowered (sf→linalg) MLIR to dylib."""
    import mlir.ir as ir
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf

    from compiler.backend.compile_utils import _compile_serveforge_free
    from compiler.backend.fixups import _fixup_unrealized_casts_pass
    from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)
    mod = ir.Module.parse(lowered_mlir, ctx)
    lower_linalg_to_llvm_ir(mod)
    _fixup_unrealized_casts_pass(mod)
    m = os.path.join(tmp_dir, "m.mlir")
    ll = os.path.join(tmp_dir, "m.ll")
    o = os.path.join(tmp_dir, "m.o")
    d = os.path.join(tmp_dir, f"{name}.dylib")
    with open(m, "w") as f:
        f.write(str(mod))
    cc = _find_tool("cc")
    mt = _find_tool("mlir-translate")
    subprocess.run([mt, "--mlir-to-llvmir", m, "-o", ll], capture_output=True, text=True, check=True, timeout=60)
    subprocess.run([cc, "-c", ll, "-o", o, "-O0"], capture_output=True, text=True, check=True, timeout=60)
    free_o = _compile_serveforge_free(tmp_dir)
    subprocess.run([cc, "-shared", "-o", d, o, free_o], capture_output=True, text=True, check=True, timeout=60)
    return d


@pytest.mark.integration
@pytest.mark.timeout(180)
class TestEmbeddingWithWeightPromotion:
    @pytest.mark.parametrize("num_weights", [25, 50, 100])
    def test_embedding_not_corrupted(self, num_weights):
        """sf.embedding + sf.weight promotion: high-index tokens must be correct."""
        vocab, hidden = 1000, 64
        rng = np.random.RandomState(42)

        mlir, w_names, w_shapes = _build_sf_mlir_with_weights(vocab, hidden, num_weights)

        w_arrays = {}
        for name, shape in zip(w_names, w_shapes, strict=True):
            w_arrays[name] = rng.randn(*shape).astype(np.float32) * 0.02
        emb_w = w_arrays["tok_embed_weight"]

        input_ids = np.zeros((2, 4), dtype=np.int64)
        input_ids[0, 0] = 2
        input_ids[0, 1] = 500
        input_ids[0, 2] = 85
        input_ids[0, 3] = 999
        input_ids[1, 0] = 998
        input_ids[1, 1] = 0
        input_ids[1, 2] = 750
        input_ids[1, 3] = 1

        with tempfile.TemporaryDirectory() as td:
            from compiler.pipeline import _apply_sf_to_linalg

            # Apply sf→linalg once, parse weight names from result
            lowered = _apply_sf_to_linalg(mlir)
            wm = re.search(r"weight_names[^]]*\[(.*?)\]", lowered, re.DOTALL)
            promoted = [w.strip().strip('"') for w in wm.group(1).split(",")]
            w_arrs = [w_arrays.get(n, np.zeros((1,), dtype=np.float32)) for n in promoted]
            all_inputs = [input_ids] + w_arrs

            # Compile from lowered text (no redundant sf→linalg call)
            dylib = _compile_lowered_to_dylib(lowered, td, f"embed_{num_weights}")

            lib = ctypes.CDLL(dylib)
            mrs = [_memref(a.ctypes.data, a.ndim, a.shape) for a in all_inputs]
            sret = (ctypes.c_uint8 * 524288)()
            args = [ctypes.byref(sret)] + [ctypes.byref(m) for m in mrs]
            k = lib._mlir_ciface_main_0
            k.argtypes = [ctypes.c_void_p] * len(args)
            k.restype = None
            k(*args)

            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = tuple(struct.unpack_from("<q", sb, 24 + 8 * i)[0] for i in range(3))
            n = int(np.prod(sz))
            actual = np.array((ctypes.c_float * n).from_address(al), dtype=np.float32).reshape(sz)

            failures = []
            for batch in range(2):
                for seq in range(4):
                    tid = int(input_ids[batch, seq])
                    expected = emb_w[tid % vocab]
                    cos_val = _cos(actual[batch, seq], expected)
                    if cos_val < 0.9999:
                        failures.append(f"  token {tid:>4}: cos={cos_val:.8f}")

            if failures:
                pytest.fail(f"sf.embedding corrupted with {num_weights} sf.weight ops:\n" + "\n".join(failures))
