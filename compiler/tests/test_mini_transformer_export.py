"""Mini Transformer — PyTorch export → sf → dylib → ctypes → compare.

Config-driven, composable, real torch.export path.
Tests the ACTUAL fx_graph_to_mlir pipeline.
"""

from __future__ import annotations

import ctypes
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════
@dataclass
class MiniConfig:
    batch: int = 2
    seq: int = 4
    vocab: int = 100
    hidden: int = 64
    heads: int = 4
    layers: int = 2
    ffn_mult: int = 4
    causal_mask: bool = False
    pos_encoding: str = "learned"

    @property
    def d_k(self) -> int:
        return self.hidden // self.heads


# ══════════════════════════════════════════════════════════════════════
class _Attention(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.hidden = config.hidden
        self.heads = config.heads
        self.d_k = config.d_k
        self.q_proj = nn.Linear(config.hidden, config.hidden, bias=False)
        self.k_proj = nn.Linear(config.hidden, config.hidden, bias=False)
        self.v_proj = nn.Linear(config.hidden, config.hidden, bias=False)
        self.out_proj = nn.Linear(config.hidden, config.hidden, bias=False)
        self.causal = config.causal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        q = self.q_proj(x).view(B, S, self.heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.heads, self.d_k).transpose(1, 2)
        scale = 1.0 / np.sqrt(self.d_k)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, scale=scale)
        attn = attn.transpose(1, 2).contiguous().view(B, S, H)
        return self.out_proj(attn)


class _FFN(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden, config.hidden * config.ffn_mult)
        self.fc2 = nn.Linear(config.hidden * config.ffn_mult, config.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class _TransformerLayer(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden, eps=1e-5)
        self.attn = _Attention(config)
        self.ln2 = nn.LayerNorm(config.hidden, eps=1e-5)
        self.ffn = _FFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class MiniTransformer(nn.Module):
    def __init__(self, config: MiniConfig):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab, config.hidden)
        self.layers = nn.ModuleList([
            _TransformerLayer(config) for _ in range(config.layers)
        ])
        self.final_ln = nn.LayerNorm(config.hidden, eps=1e-5)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.final_ln(x)


# ══════════════════════════════════════════════════════════════════════
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel().astype(np.float64)
    bf = b.ravel().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


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


def _compile_lowered_to_dylib(lowered_mlir: str, tmp_dir: str, name: str) -> str:
    """Compile sf→linalg lowered MLIR to dylib. Does NOT re-apply sf→linalg."""
    import mlir.ir as ir
    import subprocess
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf
    from compiler.backend.fixups import _fixup_unrealized_casts_pass
    from compiler.backend.llvm_backend import lower_linalg_to_llvm_ir
    from compiler.backend.compile_utils import _compile_serveforge_free
    from compiler.tests.test_precision_contract import _find_tool

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
    subprocess.run([mt, "--mlir-to-llvmir", m, "-o", ll],
                   capture_output=True, text=True, check=True, timeout=60)
    subprocess.run([cc, "-c", ll, "-o", o, "-O0"],
                   capture_output=True, text=True, check=True, timeout=60)
    free_o = _compile_serveforge_free(tmp_dir)
    subprocess.run([cc, "-shared", "-o", d, o, free_o],
                   capture_output=True, text=True, check=True, timeout=60)
    return d


# ══════════════════════════════════════════════════════════════════════
#  Low-level ciface helpers
# ══════════════════════════════════════════════════════════════════════

def _call_ciface(lib, symbol, input_arrays):
    mrs = [_memref(a.ctypes.data, a.ndim, a.shape) for a in input_arrays]
    sret = (ctypes.c_uint8 * 131072)()
    args = [ctypes.byref(sret)] + [ctypes.byref(m) for m in mrs]
    k = getattr(lib, symbol)
    k.argtypes = [ctypes.c_void_p] * len(args)
    k.restype = None
    k(*args)
    return bytes(sret)


def _parse_sret_outputs(sret_bytes, output_ranks):
    results = []
    off = 0
    for rank in output_ranks:
        desc_size = 24 + 16 * rank
        al = struct.unpack_from("<Q", sret_bytes, off + 8)[0]
        sz = tuple(struct.unpack_from("<q", sret_bytes, off + 24 + 8 * i)[0] for i in range(rank))
        n = int(np.prod(sz))
        arr = np.array((ctypes.c_float * n).from_address(al), dtype=np.float32).reshape(sz) if al and n > 0 else np.array([], dtype=np.float32)
        results.append(arr)
        off += desc_size
    return results



# ══════════════════════════════════════════════════════════════════════
@pytest.mark.integration
@pytest.mark.timeout(120)
class TestMiniTransformerExport:

    @pytest.mark.parametrize("layers", [1, 2, 4, 12])
    def test_export_compile_compare(self, layers):
        """Full path: torch.export → fx_graph_to_mlir → dylib → ctypes → compare."""
        config = MiniConfig(layers=layers, causal_mask=False)
        torch.manual_seed(42)
        model = MiniTransformer(config).eval()
        input_ids = torch.tensor([[2, 3, 1, 5], [0, 0, 0, 0]], dtype=torch.int64)

        with torch.no_grad():
            expected = model(input_ids).numpy().astype(np.float32)

        # Export → MlirModule → MLIR text
        with torch.no_grad():
            exported = torch.export.export(model, (input_ids,))
        from compiler.fx.converter import fx_graph_to_mlir
        from compiler.pipeline import _apply_sf_to_linalg
        mlir_mod = fx_graph_to_mlir(exported)

        from compiler.artifact.ir import mlir_module_to_ir_module
        import mlir.ir as ir
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        sf.register_dialects(ctx._CAPIPtr, load=True)
        ir_mod = mlir_module_to_ir_module(mlir_mod, ctx)
        mlir_text = str(ir_mod)

        # Map state_dict names to promoted weight names
        # state_dict: "tok_embed.weight" → promoted: "tok_embed_weight"
        weight_map: dict[str, np.ndarray] = {
            k.replace(".", "_"): v.numpy().astype(np.float32)
            for k, v in model.state_dict().items()
        }

        # Apply sf→linalg lowering once (shared for weight-name extraction + compilation)
        lowered = _apply_sf_to_linalg(mlir_text)
        wm = re.search(r'(?:debug_weight_names|sf\.weight_names)\s*=\s*\[(.*?)\]', lowered, re.DOTALL)
        promoted = list(dict.fromkeys(w.strip().strip('"') for w in wm.group(1).split(',')))  # deduplicate
        w_arrs = [weight_map.get(n, np.zeros((1,), dtype=np.float32)) for n in promoted]
        all_in = [input_ids.numpy().astype(np.int64)] + w_arrs

        with tempfile.TemporaryDirectory() as td:
            dylib = _compile_lowered_to_dylib(lowered, td, f"export_l{layers}")
            lib = ctypes.CDLL(dylib)

            # Single entry point via chain-wrapper's main() function.
            sret = _call_ciface(lib, "_mlir_ciface_main", all_in)
            actual = _parse_sret_outputs(sret, [3])[0]

            cos = _cos(actual, expected)
            assert cos >= 0.9999, (
                f"Export L={layers}: cos={cos:.8f} < 0.9999\n"
                f"Expected[:5]={expected.ravel()[:5].tolist()}\n"
                f"Actual[:5]={actual.ravel()[:5].tolist()}"
            )


class TestChainOrder:
    """sf.chain_order module attribute correctness."""

    @pytest.mark.parametrize("layers", [1, 2, 4])
    def test_chain_order_correct(self, layers):
        """chain_order must list functions in topological execution order."""
        config = MiniConfig(layers=layers, causal_mask=False)
        torch.manual_seed(42)
        model = MiniTransformer(config).eval()
        input_ids = torch.tensor([[2, 3, 1, 5], [0, 0, 0, 0]], dtype=torch.int64)

        with torch.no_grad():
            exported = torch.export.export(model, (input_ids,))
        from compiler.fx.converter import fx_graph_to_mlir
        from compiler.artifact import mlir_module_to_ir_module
        import mlir.ir as ir
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf

        mlir_mod = fx_graph_to_mlir(exported)
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        sf.register_dialects(ctx._CAPIPtr, load=True)
        ir_mod = mlir_module_to_ir_module(mlir_mod, ctx)
        mlir_text = str(ir_mod)

        # chain_order must be present and sorted numerically
        order_attr = ir_mod.operation.attributes.get("sf.chain_order")
        assert order_attr is not None, "sf.chain_order must be set on multi-function modules"

        chain = [str(attr) for attr in order_attr]
        # Verify numeric order: extract suffix numbers and check monotonicity
        import re
        nums = []
        for name in chain:
            m = re.search(r'main_(\d+)(?:[ab])?$', name)
            if m:
                nums.append(int(m.group(1)))
        for i in range(1, len(nums)):
            assert nums[i] >= nums[i-1], (
                f"chain_order not monotonic at index {i}: "
                f"{chain[i-1]} (#{nums[i-1]}) → {chain[i]} (#{nums[i]})"
            )
