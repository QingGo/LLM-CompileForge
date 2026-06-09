"""Mini GPT test framework — composable, configurable, maintainable.

Architecture:
  ModelConfig       — model hyperparameters
  WeightRegistry    — auto-tracking weight names + shapes
  MlirBuilder       — sf dialect MLIR generation per op
  NumpyReference    — mirror of MlirBuilder for ground-truth computation
  TransformerLayer  — compose attention + FFN from ops
  MiniGpt           — orchestrate: MLIR → compile → compare

Adding a new attention mechanism or position encoding means adding new
TransformerLayer or PosEncoding classes — no string concatenation.
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from compiler.sfcf_parser import DEFAULT_SRET_SIZE


# ══════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    batch: int = 2
    seq: int = 4
    vocab: int = 100
    hidden: int = 64
    heads: int = 4
    layers: int = 2
    ffn_mult: int = 4
    causal_mask: bool = False
    dtype: str = "f32"

    @property
    def d_k(self) -> int:
        return self.hidden // self.heads


# ══════════════════════════════════════════════════════════════════════
#  Weight registry
# ══════════════════════════════════════════════════════════════════════

class WeightRegistry:
    def __init__(self):
        self._names: list[str] = []
        self._shapes: list[tuple[int, ...]] = []

    def register(self, name: str, shape: tuple[int, ...]) -> str:
        self._names.append(name)
        self._shapes.append(shape)
        shape_str = "x".join(str(d) for d in shape)
        return f'%{name} = "sf.weight"() {{name = "{name}"}} : () -> tensor<{shape_str}xf32>'

    @property
    def names(self) -> list[str]:
        return self._names

    @property
    def shapes(self) -> list[tuple[int, ...]]:
        return self._shapes


# ══════════════════════════════════════════════════════════════════════
#  MLIR builder
# ══════════════════════════════════════════════════════════════════════

class MlirBuilder:
    def __init__(self, config: ModelConfig, weights: WeightRegistry):
        self._c = config
        self._w = weights
        self._lines: list[str] = []
        self._counter = 0

    def _new_var(self, prefix: str = "v") -> str:
        self._counter += 1
        return f"%{prefix}_{self._counter}"

    def _shape_str(self, *dims: int) -> str:
        return "x".join(str(d) for d in dims)

    def _tensor_type(self, *dims: int) -> str:
        return f"tensor<{self._shape_str(*dims)}xf32>"

    # ── Ops ──

    def embedding(self, weight_name: str, weight_shape: tuple[int, ...],
                  indices_var: str, indices_dims: tuple[int, ...]) -> str:
        v = self._new_var("emb")
        w_reg = self._w.register(weight_name, weight_shape)
        w_var = w_reg.split("=")[0].strip()
        self._lines.append(f"    {w_reg}")
        wt = self._tensor_type(*weight_shape)
        it = f"tensor<{self._shape_str(*indices_dims)}xi64>"
        ot = self._tensor_type(*indices_dims, weight_shape[-1])
        self._lines.append(
            f'    {v} = "sf.embedding"({w_var}, {indices_var}) '
            f": ({wt}, {it}) -> {ot}"
        )
        return v

    def linear(self, input_var: str, weight_shape: tuple[int, ...],
               bias_shape: tuple[int, ...] | None = None,
               prefix: str = "lin") -> str:
        v = self._new_var(prefix)
        w_name = f"w_{v[1:]}"
        w_reg = self._w.register(w_name, weight_shape)
        w_var = w_reg.split("=")[0].strip()
        self._lines.append(f"    {w_reg}")
        wt = self._tensor_type(*weight_shape)
        # Determine input rank from input type context — caller provides out_dims
        # We need output shape. For now, out_dims is inferred from weight.
        # Caller must manage shapes.
        # Actually let's just do simple 3-input and 2-input forms.
        # For simplicity, handle only f32 for now.
        # We'll let the caller provide output type.
        return w_var  # caller must emit the sf.linear call

    def layer_norm(self, input_var: str, input_dims: tuple[int, ...],
                   w_name: str, b_name: str) -> str:
        v = self._new_var("ln")
        last_dim = input_dims[-1]
        w_reg = self._w.register(w_name, (last_dim,))
        b_reg = self._w.register(b_name, (last_dim,))
        w_var = w_reg.split("=")[0].strip()
        b_var = b_reg.split("=")[0].strip()
        self._lines.append(f"    {w_reg}")
        self._lines.append(f"    {b_reg}")
        it = self._tensor_type(*input_dims)
        ot = it
        self._lines.append(
            f'    {v} = "sf.layer_norm"({input_var}, {w_var}, {b_var}) '
            f": ({it}, tensor<{last_dim}xf32>, tensor<{last_dim}xf32>) -> {ot}"
        )
        return v

    def sdpa(self, q_var: str, k_var: str, v_var: str,
             dims: tuple[int, ...],
             mask_var: str | None = None) -> str:
        v = self._new_var("attn")
        scale = 1.0 / np.sqrt(self._c.d_k)
        it = self._tensor_type(*dims)

        if mask_var:
            mask_dims = (1, dims[1], dims[1])
            mt = f"tensor<{self._shape_str(*mask_dims)}xf32>"
            self._lines.append(
                f'    {v} = "sf.scaled_dot_product_attention"'
                f'({q_var}, {k_var}, {v_var}, {mask_var})'
                f' {{scale = {scale} : f64}}'
                f' : ({it}, {it}, {it}, {mt}) -> {it}'
            )
        else:
            self._lines.append(
                f'    {v} = "sf.scaled_dot_product_attention"'
                f'({q_var}, {k_var}, {v_var})'
                f' {{scale = {scale} : f64}}'
                f' : ({it}, {it}, {it}) -> {it}'
            )
        return v

    def add(self, a_var: str, b_var: str, dims: tuple[int, ...]) -> str:
        v = self._new_var("add")
        it = self._tensor_type(*dims)
        self._lines.append(
            f'    {v} = "sf.add"({a_var}, {b_var}) : ({it}, {it}) -> {it}'
        )
        return v

    def silu(self, input_var: str, dims: tuple[int, ...]) -> str:
        v = self._new_var("act")
        it = self._tensor_type(*dims)
        self._lines.append(
            f'    {v} = "sf.silu"({input_var}) : ({it}) -> {it}'
        )
        return v

    def arange(self, start_val: int, size: int, dtype: str = "i64") -> str:
        v = self._new_var("arange")
        if dtype == "i64":
            self._lines.append(
                f"    %{v[1:]}_cst = arith.constant dense<{start_val}> : tensor<1xi64>"
            )
            self._lines.append(
                f'    {v} = "sf.arange"(%{v[1:]}_cst) '
                f": (tensor<1xi64>) -> tensor<{size}xi64>"
            )
        else:
            self._lines.append(
                f"    %{v[1:]}_cst = arith.constant dense<{float(start_val)}> : tensor<1xf32>"
            )
            self._lines.append(
                f'    {v} = "sf.arange"(%{v[1:]}_cst) '
                f": (tensor<1xf32>) -> tensor<{size}xi64>"
            )
        return v

    def causal_mask(self, seq: int) -> str:
        v = self._new_var("mask")
        self._lines.extend([
            f"    %{v[1:]}_ones = arith.constant dense<1.000000e+00> : tensor<1x{seq}x{seq}xf32>",
            f"    %{v[1:]}_tril = linalg.generic {{indexing_maps = [",
            f"        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,",
            f"        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],",
            f'        iterator_types = ["parallel", "parallel", "parallel"]}}',
            f"        ins(%{v[1:]}_ones : tensor<1x{seq}x{seq}xf32>)",
            f"        outs(%{v[1:]}_ones : tensor<1x{seq}x{seq}xf32>) {{",
            f"    ^bb0(%in: f32, %out: f32):",
            f"      %{v[1:]}_d1 = linalg.index 1 : index",
            f"      %{v[1:]}_d2 = linalg.index 2 : index",
            f"      %{v[1:]}_cmp = arith.cmpi uge, %{v[1:]}_d1, %{v[1:]}_d2 : index",
            f"      %{v[1:]}_sel = arith.select %{v[1:]}_cmp, %in, %out : f32",
            f"      linalg.yield %{v[1:]}_sel : f32",
            f"    }} -> tensor<1x{seq}x{seq}xf32>",
        ])
        return f"%{v[1:]}_tril"

    def gather(self, weight_name: str, weight_shape: tuple[int, ...],
               indices_var: str, indices_dims: tuple[int, ...]) -> str:
        v = self._new_var("gather")
        w_reg = self._w.register(weight_name, weight_shape)
        w_var = w_reg.split("=")[0].strip()
        self._lines.append(f"    {w_reg}")
        wt = self._tensor_type(*weight_shape)
        idt = f"tensor<{self._shape_str(*indices_dims)}xi64>"
        out_dims = (*indices_dims, weight_shape[-1])
        ot = self._tensor_type(*out_dims)
        self._lines.append(
            f'    {v} = "sf.embedding"({w_var}, {indices_var}) '
            f": ({wt}, {idt}) -> {ot}"
        )
        return v

    def build_module(self, func_name: str, func_args: list[str],
                     func_body: list[str],
                     ret_types: list[str], ret_vals: list[str]) -> str:
        lines = ["module {"]
        lines.append(
            f"  func.func @{func_name}({', '.join(func_args)}) -> "
            f"({', '.join(ret_types)}) {{"
        )
        lines.extend(func_body)
        lines.append(f"    return {', '.join(ret_vals)} : {', '.join(ret_types)}")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    @property
    def body_lines(self) -> list[str]:
        return self._lines


# ══════════════════════════════════════════════════════════════════════
#  Linear helper — emits weight register + sf.linear call
# ══════════════════════════════════════════════════════════════════════

def _emit_linear(
    b: MlirBuilder,
    input_var: str,
    input_dims: tuple[int, ...],
    out_features: int,
    prefix: str,
    has_bias: bool = False,
) -> str:
    v = b._new_var(prefix)
    in_features = input_dims[-1]
    w_shape = (out_features, in_features)
    w_name = f"w_{prefix}"
    w_reg = b._w.register(w_name, w_shape)
    w_var = w_reg.split("=")[0].strip()
    b.body_lines.append(f"    {w_reg}")

    it = b._tensor_type(*input_dims)
    ot = b._tensor_type(*input_dims[:-1], out_features)

    if has_bias:
        b_shape = (out_features,)
        b_name = f"b_{prefix}"
        b_reg = b._w.register(b_name, b_shape)
        b_var = b_reg.split("=")[0].strip()
        b.body_lines.append(f"    {b_reg}")
        b.body_lines.append(
            f'    {v} = "sf.linear"({input_var}, {w_var}, {b_var}) '
            f": ({it}, tensor<{out_features}x{in_features}xf32>, "
            f"tensor<{out_features}xf32>) -> {ot}"
        )
    else:
        b.body_lines.append(
            f'    {v} = "sf.linear"({input_var}, {w_var}) '
            f": ({it}, tensor<{out_features}x{in_features}xf32>) -> {ot}"
        )
    return v


# ══════════════════════════════════════════════════════════════════════
#  Transformer layer (composable)
# ══════════════════════════════════════════════════════════════════════

def _build_transformer_layer_mlir(
    b: MlirBuilder,
    hidden_var: str,
    layer_idx: int,
    config: ModelConfig,
    mask_var: str | None,
) -> str:
    """Build one transformer layer. Returns output variable."""
    p = f"l{layer_idx}"
    h = config.hidden
    dims = (config.batch, config.seq, h)

    # Pre-attention layer norm
    ln1 = b.layer_norm(hidden_var, dims, f"{p}_ln1_w", f"{p}_ln1_b")

    # Q/K/V projections
    q = _emit_linear(b, ln1, dims, h, prefix=f"{p}_q")
    k = _emit_linear(b, ln1, dims, h, prefix=f"{p}_k")
    v = _emit_linear(b, ln1, dims, h, prefix=f"{p}_v")

    # SDPA
    attn = b.sdpa(q, k, v, dims, mask_var=mask_var)

    # Output projection
    attn_out = _emit_linear(b, attn, dims, h, prefix=f"{p}_o")

    # Residual
    res1 = b.add(hidden_var, attn_out, dims)

    # Pre-FFN layer norm
    ln2 = b.layer_norm(res1, dims, f"{p}_ln2_w", f"{p}_ln2_b")

    # FFN
    ffn_hidden = h * config.ffn_mult
    ffn_dims = (config.batch, config.seq, ffn_hidden)
    fc1 = _emit_linear(b, ln2, dims, ffn_hidden, prefix=f"{p}_fc1", has_bias=True)
    act = b.silu(fc1, ffn_dims)
    fc2 = _emit_linear(b, act, ffn_dims, h, prefix=f"{p}_fc2", has_bias=True)

    return b.add(res1, fc2, dims)


# ══════════════════════════════════════════════════════════════════════
#  Numpy reference (mirrors MLIR builder)
# ══════════════════════════════════════════════════════════════════════

class NumpyReference:
    @staticmethod
    def embedding(weight: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return weight[indices]

    @staticmethod
    def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
        y = x @ w.T
        return y + b if b is not None else y

    @staticmethod
    def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                   eps: float = 1e-5) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * w + b

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1,
                mask: np.ndarray | None = None) -> np.ndarray:
        if mask is not None:
            x = np.where(mask > 0.5, x, -1e10)
        x_max = x.max(axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=axis, keepdims=True)

    @staticmethod
    def sdpa(q: np.ndarray, k: np.ndarray, v: np.ndarray,
             scale: float, mask: np.ndarray | None = None) -> np.ndarray:
        scores = q @ k.swapaxes(-1, -2) * scale
        attn = NumpyReference.softmax(scores, axis=-1, mask=mask)
        return attn @ v

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))


def _compute_transformer_layer_ref(
    ref: type[NumpyReference],
    x: np.ndarray,
    weights: dict[str, np.ndarray],
    layer_idx: int,
    config: ModelConfig,
    mask: np.ndarray | None,
) -> np.ndarray:
    p = f"l{layer_idx}"
    ln1 = ref.layer_norm(x, weights[f"{p}_ln1_w"], weights[f"{p}_ln1_b"])
    q = ref.linear(ln1, weights[f"w_{p}_q"])
    k = ref.linear(ln1, weights[f"w_{p}_k"])
    v = ref.linear(ln1, weights[f"w_{p}_v"])
    scale = 1.0 / np.sqrt(config.d_k)
    attn = ref.sdpa(q, k, v, scale, mask=mask)
    attn_out = ref.linear(attn, weights[f"w_{p}_o"])
    x = x + attn_out
    ln2 = ref.layer_norm(x, weights[f"{p}_ln2_w"], weights[f"{p}_ln2_b"])
    fc1 = ref.linear(ln2, weights[f"w_{p}_fc1"], weights[f"b_{p}_fc1"])
    act = ref.silu(fc1)
    fc2 = ref.linear(act, weights[f"w_{p}_fc2"], weights[f"b_{p}_fc2"])
    return x + fc2


# ══════════════════════════════════════════════════════════════════════
#  Mini GPT orchestrator
# ══════════════════════════════════════════════════════════════════════

class MiniGpt:
    def __init__(self, config: ModelConfig):
        self._config = config
        self._weights = WeightRegistry()

    def build_mlir(self) -> str:
        config = self._config
        b = MlirBuilder(config, self._weights)
        h, v, s, bs = config.hidden, config.vocab, config.seq, config.batch
        dims = (bs, s, h)

        # Input
        args = [f"%input_ids: tensor<{bs}x{s}xi64>"]
        ret_types = [b._tensor_type(*dims)]

        # Token embedding
        tok = b.embedding("tok_embed", (v, h), "%input_ids", (bs, s))

        # Position embedding
        pos_ids = b.arange(0, s, dtype="i64")
        pos_var = b.gather("pos_embed", (s, h), pos_ids, (s,))

        # Broadcast position embedding to batch
        pos_bc = _emit_simple_broadcast(b, pos_var, (s, h), tok, dims)

        # Add
        hidden = b.add(tok, pos_bc, dims)

        # Causal mask
        mask_var = b.causal_mask(s) if config.causal_mask else None

        # Layers
        for li in range(config.layers):
            hidden = _build_transformer_layer_mlir(
                b, hidden, li, config, mask_var,
            )

        # Final layer norm is in the layer already (or add separately)
        # For simplicity, add final LN
        final_ln = b.layer_norm(hidden, dims, "final_ln_w", "final_ln_b")

        return b.build_module(
            "main_0", args, b.body_lines, ret_types, [final_ln]
        )

    def compute_reference(
        self, input_ids: np.ndarray, weights: dict[str, np.ndarray]
    ) -> np.ndarray:
        config = self._config
        ref = NumpyReference
        mask = np.tril(np.ones((1, config.seq, config.seq))) if config.causal_mask else None

        x = ref.embedding(weights["tok_embed"], input_ids)
        pos = weights["pos_embed"][np.arange(config.seq)]
        x = x + pos[np.newaxis, :, :]

        for li in range(config.layers):
            x = _compute_transformer_layer_ref(ref, x, weights, li, config, mask)
        x = ref.layer_norm(x, weights["final_ln_w"], weights["final_ln_b"])
        return x

    @property
    def weight_names(self) -> list[str]:
        return self._weights.names

    @property
    def weight_shapes(self) -> list[tuple[int, ...]]:
        return self._weights.shapes


def _emit_simple_broadcast(
    b: MlirBuilder, src_var: str, src_dims: tuple[int, ...],
    dst_var: str, dst_dims: tuple[int, ...],
) -> str:
    """Broadcast src from (S,H) to (B,S,H) using linalg.generic."""
    v = b._new_var("bc")
    st = b._tensor_type(*src_dims)
    dt = b._tensor_type(*dst_dims)
    b.body_lines.extend([
        f"    {v} = linalg.generic {{indexing_maps = [",
        f"        affine_map<(d0, d1, d2) -> (d1, d2)>,",
        f"        affine_map<(d0, d1, d2) -> (d0, d1, d2)>],",
        f'        iterator_types = ["parallel", "parallel", "parallel"]}}',
        f"        ins({src_var} : {st})",
        f"        outs({dst_var} : {dt}) {{",
        "    ^bb0(%pv: f32, %o: f32):",
        "      linalg.yield %pv : f32",
        f"    }} -> {dt}",
    ])
    return v


# ══════════════════════════════════════════════════════════════════════
#  Compilation helpers
# ══════════════════════════════════════════════════════════════════════

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
    from compiler.backend.compile_utils import _compile_serveforge_free

    lowered = _apply_sf_to_linalg(sf_mlir)
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    sf.register_dialects(ctx._CAPIPtr, load=True)
    mod = ir.Module.parse(lowered, ctx)
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
    subprocess.run([mt, "--mlir-to-llvmir", m, "-o", ll],
                   capture_output=True, text=True, check=True, timeout=60)
    subprocess.run([cc, "-c", ll, "-o", o, "-O0"],
                   capture_output=True, text=True, check=True, timeout=60)
    free_o = _compile_serveforge_free(tmp_dir)
    subprocess.run([cc, "-shared", "-o", d, o, free_o],
                   capture_output=True, text=True, check=True, timeout=60)
    return d


# ══════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.integration
@pytest.mark.timeout(120)
class TestMiniGptV2:

    @pytest.mark.parametrize("layers", [2, 4, 8, 12])
    def test_basic(self, layers):
        config = ModelConfig(layers=layers, causal_mask=False)
        self._run_test(config)

    def test_causal_mask(self):
        config = ModelConfig(layers=2, causal_mask=True)
        self._run_test(config)

    def _run_test(self, config: ModelConfig):
        rng = np.random.RandomState(42)
        model = MiniGpt(config)
        mlir = model.build_mlir()

        w_shapes = model.weight_shapes
        weights = {
            n: rng.randn(*s).astype(np.float32) * 0.02
            for n, s in zip(model.weight_names, w_shapes)
        }
        input_ids = np.array([[2, 3, 1, 5], [0, 0, 0, 0]], dtype=np.int64)

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile(mlir, td, f"test_v2_l{config.layers}")

            from compiler.pipeline import _apply_sf_to_linalg
            import re
            lowered = _apply_sf_to_linalg(mlir)
            wm = re.search(r'sf\.weight_names\s*=\s*\[(.*?)\]', lowered, re.DOTALL)
            promoted = [w.strip().strip('"') for w in wm.group(1).split(',')]
            w_arrs = [weights[n] for n in promoted]
            all_in = [input_ids] + w_arrs

            lib = ctypes.CDLL(dylib)
            mrs = [_memref(a.ctypes.data, a.ndim, a.shape) for a in all_in]
            sret = (ctypes.c_uint8 * DEFAULT_SRET_SIZE)()
            args = [ctypes.byref(sret)] + [ctypes.byref(m) for m in mrs]
            k = getattr(lib, "_mlir_ciface_main_0")
            k.argtypes = [ctypes.c_void_p] * len(args)
            k.restype = None
            k(*args)

            sb = bytes(sret)
            al = struct.unpack_from("<Q", sb, 8)[0]
            sz = tuple(struct.unpack_from("<q", sb, 24 + 8 * i)[0] for i in range(3))
            actual = np.array(
                (ctypes.c_float * int(np.prod(sz))).from_address(al), dtype=np.float32
            ).reshape(sz)

            expected = model.compute_reference(input_ids, weights)
            cos = _cos(actual, expected)
            assert cos >= 0.9999, (
                f"Mini GPT L={config.layers} mask={config.causal_mask}: "
                f"cos={cos:.8f} < 0.9999"
            )
