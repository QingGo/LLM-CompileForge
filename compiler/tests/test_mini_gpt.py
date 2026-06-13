"""Mini Transformer Decoder TDD test.

Compiles a 2-layer GPT-style decoder (hidden=64, vocab=100, heads=4)
from sf dialect MLIR → dylib → ctypes call → compare with numpy reference.

All weights are synthesized (no external dependencies), compilation is fast,
and feedback loop is short.
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


def _compile_sf_to_dylib(sf_mlir: str, tmp_dir: str, name: str) -> str:
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
            [_find_tool("cc"), "-c", l, "-o", o, "-O0"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        free_o = _compile_serveforge_free(tmp_dir)
        link_dylib([o, free_o], d)
        return d


# ══════════════════════════════════════════════════════════════════════
#  Mini GPT configuration
# ══════════════════════════════════════════════════════════════════════

BATCH, SEQ, VOCAB, HIDDEN, HEADS = 2, 4, 100, 64, 4
D_K = HIDDEN // HEADS  # 16


def _make_mini_gpt_mlir(num_layers: int = 2) -> tuple[str, list[str], list[tuple[int, ...]]]:
    """Generate sf dialect MLIR for a 2-layer GPT decoder.

    Returns (mlir_text, weight_names, weight_shapes) so tests can
    generate matching random weights and call the dylib.
    """
    lines = ["module {"]
    weight_names: list[str] = []
    weight_shapes: list[tuple[int, ...]] = []

    def w(name: str, shape: tuple[int, ...]) -> str:
        weight_names.append(name)
        weight_shapes.append(shape)
        shape_str = "x".join(str(d) for d in shape)
        return f'%{name} = "sf.weight"() {{name = "{name}"}} : () -> tensor<{shape_str}xf32>'

    h = str(HIDDEN)  # "64"
    v = str(VOCAB)  # "100"
    dk = str(D_K)  # "16"
    hd = str(HEADS)  # "4"
    b, s = str(BATCH), str(SEQ)

    args = [f"%input_ids: tensor<{b}x{s}xi64>"]
    ret_types = [f"tensor<{b}x{s}x{h}xf32>"]
    ret_vals = []

    lines.append(f"  func.func @main_0({', '.join(args)}) -> ({', '.join(ret_types)}) {{")

    # Token embedding
    lines.append(f"    {w('tok_embed', (VOCAB, HIDDEN))}")
    lines.append(
        f'    %h = "sf.embedding"(%tok_embed, %input_ids) : (tensor<{v}x{h}xf32>, tensor<{b}x{s}xi64>) -> tensor<{b}x{s}x{h}xf32>'
    )

    # Position embedding — use sf.embedding for simplicity
    lines.append(f"    {w('pos_embed', (SEQ, HIDDEN))}")
    lines.append("    %pos_ids_0 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi64>")
    lines.append(
        f'    %pos = "sf.embedding"(%pos_embed, %pos_ids_0) : (tensor<{s}x{h}xf32>, tensor<4xi64>) -> tensor<4x{h}xf32>'
    )
    lines.append(
        f'    %pos_bc = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}} ins(%pos : tensor<4x{h}xf32>) outs(%h : tensor<{b}x{s}x{h}xf32>) {{'
    )
    lines.append("    ^bb0(%pv: f32, %o: f32):")
    lines.append("      linalg.yield %pv : f32")
    lines.append(f"    }} -> tensor<{b}x{s}x{h}xf32>")
    lines.append(
        f'    %h1 = "sf.add"(%h, %pos_bc) : (tensor<{b}x{s}x{h}xf32>, tensor<{b}x{s}x{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
    )

    h_var = "%h1"
    for layer in range(num_layers):
        p = f"l{layer}"
        # Layer norm 1 (pre-attn)
        lines.append(f"    {w(f'{p}_ln1_w', (HIDDEN,))}")
        lines.append(f"    {w(f'{p}_ln1_b', (HIDDEN,))}")
        lines.append(
            f'    %{p}_ln1 = "sf.layer_norm"({h_var}, %{p}_ln1_w, %{p}_ln1_b) : (tensor<{b}x{s}x{h}xf32>, tensor<{h}xf32>, tensor<{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
        )

        # Self-attention: Q, K, V projections
        lines.append(f"    {w(f'{p}_q_w', (HIDDEN, HIDDEN))}")
        lines.append(f"    {w(f'{p}_k_w', (HIDDEN, HIDDEN))}")
        lines.append(f"    {w(f'{p}_v_w', (HIDDEN, HIDDEN))}")
        lines.append(f"    {w(f'{p}_o_w', (HIDDEN, HIDDEN))}")
        lines.append(
            f'    %{p}_q = "sf.linear"(%{p}_ln1, %{p}_q_w) : (tensor<{b}x{s}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
        )
        lines.append(
            f'    %{p}_k = "sf.linear"(%{p}_ln1, %{p}_k_w) : (tensor<{b}x{s}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
        )
        lines.append(
            f'    %{p}_v = "sf.linear"(%{p}_ln1, %{p}_v_w) : (tensor<{b}x{s}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
        )

        # Reshape for multi-head: (B,S,H) → (B,S,HEADS,D_K) — skip for simplicity, use single head
        # Actually the full model reshapes. For mini GPT, keep shapes flat

        # SDPA
        scale_val = 1.0 / np.sqrt(D_K)
        lines.append(
            f'    %{p}_attn = "sf.scaled_dot_product_attention"(%{p}_q, %{p}_k, %{p}_v) {{scale = {scale_val} : f64}} : (tensor<{b}x{s}x{h}xf32>, tensor<{b}x{s}x{h}xf32>, tensor<{b}x{s}x{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
        )

        # Output projection
        lines.append(
            f'    %{p}_attn_out = "sf.linear"(%{p}_attn, %{p}_o_w) : (tensor<{b}x{s}x{h}xf32>, tensor<{h}x{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
        )

        # Residual
        lines.append(
            f'    %{p}_res1 = "sf.add"({h_var}, %{p}_attn_out) : (tensor<{b}x{s}x{h}xf32>, tensor<{b}x{s}x{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
        )

        # Layer norm 2 (pre-FFN)
        lines.append(f"    {w(f'{p}_ln2_w', (HIDDEN,))}")
        lines.append(f"    {w(f'{p}_ln2_b', (HIDDEN,))}")
        lines.append(
            f'    %{p}_ln2 = "sf.layer_norm"(%{p}_res1, %{p}_ln2_w, %{p}_ln2_b) : (tensor<{b}x{s}x{h}xf32>, tensor<{h}xf32>, tensor<{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
        )

        # FFN: fc1 → silu → fc2
        ffn_hidden = HIDDEN * 4  # 256
        fh = str(ffn_hidden)
        lines.append(f"    {w(f'{p}_fc1_w', (ffn_hidden, HIDDEN))}")
        lines.append(f"    {w(f'{p}_fc1_b', (ffn_hidden,))}")
        lines.append(f"    {w(f'{p}_fc2_w', (HIDDEN, ffn_hidden))}")
        lines.append(f"    {w(f'{p}_fc2_b', (HIDDEN,))}")
        lines.append(
            f'    %{p}_fc1 = "sf.linear"(%{p}_ln2, %{p}_fc1_w, %{p}_fc1_b) : (tensor<{b}x{s}x{h}xf32>, tensor<{fh}x{h}xf32>, tensor<{fh}xf32>) -> tensor<{b}x{s}x{fh}xf32>'
        )
        lines.append(f'    %{p}_act = "sf.silu"(%{p}_fc1) : (tensor<{b}x{s}x{fh}xf32>) -> tensor<{b}x{s}x{fh}xf32>')
        lines.append(
            f'    %{p}_fc2 = "sf.linear"(%{p}_act, %{p}_fc2_w, %{p}_fc2_b) : (tensor<{b}x{s}x{fh}xf32>, tensor<{h}x{fh}xf32>, tensor<{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
        )

        # Residual
        lines.append(
            f'    %{p}_res2 = "sf.add"(%{p}_res1, %{p}_fc2) : (tensor<{b}x{s}x{h}xf32>, tensor<{b}x{s}x{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
        )

        h_var = f"%{p}_res2"

    # Final layer norm
    lines.append(f"    {w('final_ln_w', (HIDDEN,))}")
    lines.append(f"    {w('final_ln_b', (HIDDEN,))}")
    lines.append(
        f'    %final_ln = "sf.layer_norm"({h_var}, %final_ln_w, %final_ln_b) : (tensor<{b}x{s}x{h}xf32>, tensor<{h}xf32>, tensor<{h}xf32>) -> tensor<{b}x{s}x{h}xf32>'
    )

    ret_vals.append("%final_ln")

    lines.append(f"    return {', '.join(ret_vals)} : {', '.join(ret_types)}")
    lines.append("  }")
    lines.append("}")

    return "\n".join(lines), weight_names, weight_shapes


# ══════════════════════════════════════════════════════════════════════
#  Numpy reference
# ══════════════════════════════════════════════════════════════════════


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
    scores = q @ k.swapaxes(-1, -2) * scale
    attn = _softmax(scores, axis=-1)
    return attn @ v


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    # w is (out, in), x @ w.T
    y = x @ w.T
    if b is not None:
        y = y + b
    return y


def _mini_gpt_ref(
    input_ids: np.ndarray,
    weights: dict[str, np.ndarray],
    num_layers: int = 2,
) -> np.ndarray:
    """Pure numpy reference for the mini GPT decoder."""
    x = weights["tok_embed"][input_ids]  # (B,S,H)
    pos = weights["pos_embed"][np.arange(SEQ)]  # (S,H)
    x = x + pos[np.newaxis, :, :]  # (B,S,H)

    for layer in range(num_layers):
        p = f"l{layer}"
        # Pre-attn LN
        ln1 = _layer_norm(x, weights[f"{p}_ln1_w"], weights[f"{p}_ln1_b"])
        # Self-attention
        q = _linear(ln1, weights[f"{p}_q_w"])
        k = _linear(ln1, weights[f"{p}_k_w"])
        v = _linear(ln1, weights[f"{p}_v_w"])
        scale = 1.0 / np.sqrt(D_K)
        attn = _sdpa(q, k, v, scale)
        attn_out = _linear(attn, weights[f"{p}_o_w"])
        x = x + attn_out
        # Pre-FFN LN
        ln2 = _layer_norm(x, weights[f"{p}_ln2_w"], weights[f"{p}_ln2_b"])
        # FFN
        fc1 = _linear(ln2, weights[f"{p}_fc1_w"], weights[f"{p}_fc1_b"])
        act = _silu(fc1)
        fc2 = _linear(act, weights[f"{p}_fc2_w"], weights[f"{p}_fc2_b"])
        x = x + fc2

    x = _layer_norm(x, weights["final_ln_w"], weights["final_ln_b"])
    return x


# ══════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestMiniGpt:
    @pytest.mark.parametrize("num_layers", [2, 4, 8, 12])
    def test_mini_gpt_compiles_and_runs(self, num_layers):
        """Mini GPT compiles to dylib and produces output matching numpy ref."""
        rng = np.random.RandomState(42)
        mlir, w_names, w_shapes = _make_mini_gpt_mlir(num_layers=num_layers)
        weights = {n: rng.randn(*s).astype(np.float32) * 0.02 for n, s in zip(w_names, w_shapes)}
        input_ids = np.array([[2, 3, 1, 5], [0, 0, 0, 0]], dtype=np.int64)

        # Compile to dylib
        with tempfile.TemporaryDirectory() as td:
            dylib = _compile_sf_to_dylib(mlir, td, f"mini_gpt_l{num_layers}")

            # Get weight order from lowered MLIR (after sf-promote-weights)
            import re

            from compiler.pipeline import _apply_sf_to_linalg

            lowered = _apply_sf_to_linalg(mlir)
            wm = re.search(r"sf\.weight_names\s*=\s*\[(.*?)\]", lowered, re.DOTALL)
            promoted_names = [w.strip().strip('"') for w in wm.group(1).split(",")]

            # Build input args: input_ids + weights in promoted order
            w_arrs = [weights[n] for n in promoted_names]
            all_inputs = [input_ids] + w_arrs

            # Call ciface
            lib = ctypes.CDLL(dylib)
            mrs = [_memref(a.ctypes.data, a.ndim, a.shape) for a in all_inputs]
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret)] + [ctypes.byref(m) for m in mrs]
            k = lib._mlir_ciface_main_0
            k.argtypes = [ctypes.c_void_p] * len(args)
            k.restype = None
            k(*args)

            # Read output: single rank-3 tensor in sret
            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = tuple(struct.unpack_from("<q", sb, 24 + 8 * i)[0] for i in range(3))
            assert sz == (BATCH, SEQ, HIDDEN), f"Wrong output shape: {sz}"
            n = int(np.prod(sz))
            actual = np.array((ctypes.c_float * n).from_address(al), dtype=np.float32).reshape(sz)

            # Numpy reference
            expected = _mini_gpt_ref(input_ids, weights, num_layers=num_layers)

            cos = _cos(actual, expected)
            assert cos >= 0.9999, (
                f"Mini GPT L={num_layers} cos={cos:.8f} < 0.9999\n"
                f"Expected[:5]={expected.ravel()[:5].tolist()}\n"
                f"Actual[:5]={actual.ravel()[:5].tolist()}"
            )
