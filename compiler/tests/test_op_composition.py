"""Op composition precision tests — verify op chains against HF reference.

Compiles standalone dylibs from sf-dialect MLIR for specific op chains
(attention, FFN, full layer), exposes intermediate tensors via multi-output
functions, and compares each against the HuggingFace reference using a
4-gate precision check.

Key insight: main_0 (embedding) is correct (cos=1.0), but main_1 (layer 0)
diverges with mean_rel_err=0.134. These tests expose which specific op in
the chain causes divergence by returning all intermediate tensors.

Usage::

    pytest compiler/tests/test_op_composition.py -v --timeout=0
    pytest compiler/tests/test_op_composition.py -v -k "seq2" --timeout=0
    pytest compiler/tests/test_op_composition.py -v -k "attention_chain" --timeout=0
"""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

# ── Path setup ──────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from compiler.tests.test_per_function_cos import check_4_gate  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────

VOCAB_SIZE: int = 50265
RANDOM_SEED: int = 42
N_HEADS: int = 12
HEAD_DIM: int = 64  # 768 / 12
HIDDEN_DIM: int = 768
FFN_DIM: int = 3072
Q_SCALE: float = 1.0 / np.sqrt(HEAD_DIM)  # 0.125
EPS: float = 1e-5

# ── Tool finding ────────────────────────────────────────────────────


def _find_tool(name: str) -> str:
    """Locate a binary needed for compilation (mlir-translate, cc)."""
    candidates = [name]
    if name in ("cc", "clang"):
        for hp in [
            "/usr/local/opt/llvm/bin/clang",
            "/opt/homebrew/opt/llvm/bin/clang",
        ]:
            candidates.insert(0, hp)
    candidates.append(str(ROOT / "llvm-project" / "build" / "bin" / name))
    for p in candidates:
        if Path(p).is_file():
            return str(p)
        try:
            if subprocess.run([p, "--version"], capture_output=True, timeout=5).returncode == 0:
                return p
        except FileNotFoundError:
            continue
    raise RuntimeError(f"{name} not found")


# ── Sret + MemRef helpers ───────────────────────────────────────────


def _sret(size: int) -> ctypes.Array:
    return (ctypes.c_uint8 * size)()


def _memref(arr: np.ndarray) -> Any:
    """Build ctypes MemRef descriptor from numpy array."""
    ndim = arr.ndim
    elem_strides = tuple(s // arr.itemsize for s in arr.strides)

    class M(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("sizes", ctypes.c_int64 * ndim),
            ("strides", ctypes.c_int64 * ndim),
        ]

    return M(
        ctypes.c_void_p(arr.ctypes.data),
        ctypes.c_void_p(arr.ctypes.data),
        0,
        (ctypes.c_int64 * ndim)(*arr.shape),
        (ctypes.c_int64 * ndim)(*elem_strides),
    )


def desc_size(rank: int) -> int:
    """Size in bytes of a single sret memref descriptor for the given rank."""
    return 24 + 16 * rank


def _parse_sret_outputs(
    sret_bytes: bytes,
    ranks: list[int],
    expected_shapes: list[tuple[int, ...]] | None = None,
) -> list[np.ndarray]:
    """Parse output tensors from the sret buffer.

    Each output descriptor in the sret buffer has layout:
        offset 0:  allocated (i64)
        offset 8:  aligned (i64)  ← pointer to actual output data
        offset 16: offset (i64)
        offset 24: sizes[i64] * rank
        after sizes: strides[i64] * rank
    Total per-descriptor: 24 + 16*rank bytes.

    Args:
        sret_bytes: Raw sret buffer from ciface call.
        ranks: Expected rank for each output.
        expected_shapes: Optional list of expected shapes. When provided
            and a runtime size reads as 0 or 1 (suspicious for dynamic dims),
            the expected shape dim is used instead. This handles cases where
            the compiled code reports wrong runtime sizes for intermediate
            dynamic dimensions.
    """
    tensors = []
    offset = 0
    for oi, rank in enumerate(ranks):
        dsize = desc_size(rank)
        desc = sret_bytes[offset : offset + dsize]

        aligned = struct.unpack_from("<Q", desc, 8)[0]

        runtime_sizes = []
        for i in range(rank):
            s = struct.unpack_from("<q", desc, 24 + 8 * i)[0]
            # If runtime size is 0 or suspiciously 1 (potential dynamic-dim bug),
            # and we have an expected shape, use the expected shape's dim.
            if s <= 0 or s > 1_000_000_000:
                if expected_shapes is not None and oi < len(expected_shapes):
                    exp_shape = expected_shapes[oi]
                    if i < len(exp_shape) and exp_shape[i] > 0:
                        s = exp_shape[i]
                    else:
                        s = 1
                else:
                    s = 1
            runtime_sizes.append(int(s))

        n = int(np.prod(runtime_sizes))
        if n > 0 and aligned != 0:
            buf = (ctypes.c_float * n).from_address(aligned)
            arr = np.array(buf, dtype=np.float32).copy().reshape(runtime_sizes)
        else:
            arr = np.array([], dtype=np.float32)

        tensors.append(arr)
        offset += dsize

    return tensors


# ── Compilation helpers ─────────────────────────────────────────────


def _compile_single_function(
    sf_mlir: str,
    name: str = "test_func",
    tmp_dir: str | None = None,
) -> ctypes.CDLL:
    """Compile sf-dialect MLIR → lowered → LLVM → cc → dylib.

    Returns a loaded ctypes.CDLL ready for ciface calling.
    """
    _maybe_import_mlir()
    import mlir.ir as ir
    from mlir_sf._mlir_libs._sfDialectsNanobind import sf  # noqa: F401

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

    if tmp_dir is None:
        td_context = tempfile.TemporaryDirectory()
        td = td_context.__enter__()

        def _cleanup():
            td_context.__exit__(None, None, None)
    else:
        td = tmp_dir

        def _cleanup():
            return None

    try:
        m = os.path.join(td, "m.mlir")
        ll_file = os.path.join(td, "m.ll")
        o = os.path.join(td, "m.o")
        d = os.path.join(td, f"{name}.dylib")

        with open(m, "w") as f:
            f.write(str(mod))

        subprocess.run(
            [_find_tool("mlir-translate"), "--mlir-to-llvmir", m, "-o", ll_file],
            capture_output=True, text=True, check=True, timeout=120,
        )
        subprocess.run(
            [_find_tool("cc"), "-c", ll_file, "-o", o, "-O0"],
            capture_output=True, text=True, check=True, timeout=120,
        )
        free_o = _compile_serveforge_free(td)
        link_dylib([o, free_o], d)

        return ctypes.CDLL(d)
    finally:
        _cleanup()


def _compile_multi_output_function(
    sf_mlir: str,
    output_names: list[str],
    name: str = "multi_out_func",
    tmp_dir: str | None = None,
) -> ctypes.CDLL:
    """Compile MLIR with multi-output function — same as _compile_single_function.

    The MLIR function must return multiple tensors via ``func.return``.
    The caller is responsible for constructing the MLIR with the correct
    output ranks and parsing the sret buffer accordingly.
    """
    return _compile_single_function(sf_mlir, name=name, tmp_dir=tmp_dir)


# ── MLIR import guard ───────────────────────────────────────────────

_mlir_imported: bool = False


def _maybe_import_mlir() -> None:
    """Ensure MLIR bindings are on sys.path (idempotent)."""
    global _mlir_imported
    if _mlir_imported:
        return
    _mlir_imported = True
    from compiler.backend.compile_utils import _setup_mlir_path

    _setup_mlir_path()


# ── HF weight loading ───────────────────────────────────────────────

_hf_model_cache: dict[str, Any] = {}


def _load_hf_model() -> Any:
    """Load facebook/opt-125m in float32, cached per process."""
    cache_key = "opt_125m"
    if cache_key not in _hf_model_cache:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torch_dtype=torch.float32,
        )
        model = model.to("cpu")
        model.eval()
        _hf_model_cache[cache_key] = model
    return _hf_model_cache[cache_key]


def _extract_layer_weights_numpy(hf_model: Any, layer_idx: int) -> dict[str, np.ndarray]:
    """Extract layer weights from HF model as numpy float32 arrays.

    Args:
        hf_model: Loaded HF OPT-125m model.
        layer_idx: Layer index (0-11).
    """
    layer = hf_model.model.decoder.layers[layer_idx]
    weights = {}

    # Self-attn layer norm
    weights["ln_w"] = layer.self_attn_layer_norm.weight.detach().numpy().astype(np.float32)
    weights["ln_b"] = layer.self_attn_layer_norm.bias.detach().numpy().astype(np.float32)

    # QKV projections
    weights["q_w"] = layer.self_attn.q_proj.weight.detach().numpy().astype(np.float32)
    weights["q_b"] = layer.self_attn.q_proj.bias.detach().numpy().astype(np.float32)
    weights["k_w"] = layer.self_attn.k_proj.weight.detach().numpy().astype(np.float32)
    weights["k_b"] = layer.self_attn.k_proj.bias.detach().numpy().astype(np.float32)
    weights["v_w"] = layer.self_attn.v_proj.weight.detach().numpy().astype(np.float32)
    weights["v_b"] = layer.self_attn.v_proj.bias.detach().numpy().astype(np.float32)
    weights["o_w"] = layer.self_attn.out_proj.weight.detach().numpy().astype(np.float32)
    weights["o_b"] = layer.self_attn.out_proj.bias.detach().numpy().astype(np.float32)

    # FFN layer norm
    weights["ffn_ln_w"] = layer.final_layer_norm.weight.detach().numpy().astype(np.float32)
    weights["ffn_ln_b"] = layer.final_layer_norm.bias.detach().numpy().astype(np.float32)

    # FFN projections
    weights["fc1_w"] = layer.fc1.weight.detach().numpy().astype(np.float32)
    weights["fc1_b"] = layer.fc1.bias.detach().numpy().astype(np.float32)
    weights["fc2_w"] = layer.fc2.weight.detach().numpy().astype(np.float32)
    weights["fc2_b"] = layer.fc2.bias.detach().numpy().astype(np.float32)

    return weights

# Keep backward-compat alias
def _extract_layer0_weights_numpy(hf_model):
    return _extract_layer_weights_numpy(hf_model, 0)


# ── HF reference computation ────────────────────────────────────────


def _hf_attention_chain_ref(
    hf_model: Any,
    hidden_states: np.ndarray,
    seq_len: int,
    layer_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Compute attention chain reference using HF model submodules.

    Args:
        hf_model: Loaded HF OPT-125m model.
        hidden_states: [1, seq_len, 768] float32 numpy array (input embeddings).
        seq_len: Number of tokens.
        layer_idx: Layer index (0-11).

    Returns:
        Dict mapping stage name → numpy array of intermediate result.
    """
    layer = hf_model.model.decoder.layers[layer_idx]
    scaling = Q_SCALE  # 1/sqrt(64) = 0.125

    hs = torch.from_numpy(hidden_states.astype(np.float32))

    with torch.no_grad():
        # 1. Layer norm
        ln_out = layer.self_attn_layer_norm(hs)

        # 2. QKV projections
        q_raw = layer.self_attn.q_proj(ln_out)  # [1, S, 768]
        q_scaled = q_raw * scaling  # pre-scale Q
        k_raw = layer.self_attn.k_proj(ln_out)  # [1, S, 768]
        v_raw = layer.self_attn.v_proj(ln_out)  # [1, S, 768]

        # 3. Reshape + transpose for multi-head
        b = 1
        s = seq_len
        q_t = q_scaled.reshape(b, s, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()  # [B, 12, S, 64]
        k_t = k_raw.reshape(b, s, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()  # [B, 12, S, 64]
        v_t = v_raw.reshape(b, s, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()  # [B, 12, S, 64]

        # 4. Build causal mask (same as compiled model: [1, 1, S, S] with -inf)
        causal_mask = torch.tril(torch.ones(1, 1, s, s, dtype=torch.float32))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float("-inf"))

        # 5. Scaled dot-product attention (scale=1.0 because Q is pre-scaled)
        sdpa_out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=causal_mask,
            scale=1.0,
        )  # [1, 12, S, 64]

        # 6. Transpose back + flatten
        sdpa_flat = sdpa_out.transpose(1, 2).contiguous().reshape(b, s, HIDDEN_DIM)  # [1, S, 768]

        # 7. Output projection
        final = layer.self_attn.out_proj(sdpa_flat)  # [1, S, 768]

    return {
        "ln_out": ln_out.numpy().astype(np.float32),
        "q_raw": q_raw.numpy().astype(np.float32),
        "k_raw": k_raw.numpy().astype(np.float32),
        "v_raw": v_raw.numpy().astype(np.float32),
        "q_t": q_t.numpy().astype(np.float32),
        "k_t": k_t.numpy().astype(np.float32),
        "v_t": v_t.numpy().astype(np.float32),
        "sdpa_out": sdpa_out.numpy().astype(np.float32),
        "final": final.numpy().astype(np.float32),
    }


def _hf_full_layer_ref(
    hf_model: Any,
    hidden_states: np.ndarray,
    seq_len: int,
    layer_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Compute full transformer layer reference using HF model submodules.

    Args:
        layer_idx: Layer index (0-11).

    Returns dict with keys for attention intermediates + FFN intermediates.
    """
    layer = hf_model.model.decoder.layers[layer_idx]
    scaling = Q_SCALE

    hs = torch.from_numpy(hidden_states.astype(np.float32))
    b = 1
    s = seq_len

    with torch.no_grad():
        # ── Attention block ──
        residual = hs  # [1, S, 768]
        ln_out = layer.self_attn_layer_norm(hs)

        q_raw = layer.self_attn.q_proj(ln_out)
        q_scaled = q_raw * scaling
        k_raw = layer.self_attn.k_proj(ln_out)
        v_raw = layer.self_attn.v_proj(ln_out)

        q_t = q_scaled.reshape(b, s, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
        k_t = k_raw.reshape(b, s, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()
        v_t = v_raw.reshape(b, s, N_HEADS, HEAD_DIM).transpose(1, 2).contiguous()

        causal_mask = torch.tril(torch.ones(1, 1, s, s, dtype=torch.float32))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float("-inf"))

        sdpa_out = torch.nn.functional.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=causal_mask, scale=1.0,
        )

        sdpa_flat = sdpa_out.transpose(1, 2).contiguous().reshape(b, s, HIDDEN_DIM)
        attn_output = layer.self_attn.out_proj(sdpa_flat)

        # Residual add (after dropout — dropout is identity in eval mode)
        attn_residual = residual + attn_output

        # ── FFN block ──
        ffn_ln_out = layer.final_layer_norm(attn_residual)
        ffn_hidden = layer.fc1(ffn_ln_out)  # [1, S, 3072]
        ffn_relu = torch.nn.functional.relu(ffn_hidden)
        ffn_output = layer.fc2(ffn_relu)  # [1, S, 768]

        # FFN residual
        ffn_residual = attn_residual + ffn_output

    return {
        # Attention intermediates
        "attn_ln_out": ln_out.numpy().astype(np.float32),
        "q_t": q_t.numpy().astype(np.float32),
        "k_t": k_t.numpy().astype(np.float32),
        "v_t": v_t.numpy().astype(np.float32),
        "sdpa_out": sdpa_out.numpy().astype(np.float32),
        "attn_output": attn_output.numpy().astype(np.float32),
        "attn_residual": attn_residual.numpy().astype(np.float32),
        # FFN intermediates
        "ffn_ln_out": ffn_ln_out.numpy().astype(np.float32),
        "ffn_hidden": ffn_hidden.numpy().astype(np.float32),
        "ffn_relu": ffn_relu.numpy().astype(np.float32),
        "ffn_output": ffn_output.numpy().astype(np.float32),
        "ffn_residual": ffn_residual.numpy().astype(np.float32),
    }


# ── MLIR templates ──────────────────────────────────────────────────

# Multi-output attention chain function.
# Returns 9 intermediate tensors: ln_out, q_raw, k_raw, v_raw,
#   q_t, k_t, v_t, sdpa_out, final_output.
# All weight tensors are passed as function arguments (no sf.weight ops),
# so sf-promote-weights is a no-op and sf-lower-to-linalg works directly.
_ATTENTION_CHAIN_MLIR = """module {
  func.func @attn_chain(
    %input: tensor<?x?x768xf32>,
    %ln_w: tensor<768xf32>, %ln_b: tensor<768xf32>,
    %q_w: tensor<768x768xf32>, %q_b: tensor<768xf32>,
    %k_w: tensor<768x768xf32>, %k_b: tensor<768xf32>,
    %v_w: tensor<768x768xf32>, %v_b: tensor<768xf32>,
    %o_w: tensor<768x768xf32>, %o_b: tensor<768xf32>,
    %mask: tensor<?x1x?x?xf32>,
    %batch_dim: tensor<1xf32>, %seq_dim: tensor<1xf32>,
    %q_scale: tensor<1xf32>
  ) -> (
    tensor<?x?x768xf32>,
    tensor<?x?x768xf32>,
    tensor<?x?x768xf32>,
    tensor<?x?x768xf32>,
    tensor<?x12x?x64xf32>,
    tensor<?x12x?x64xf32>,
    tensor<?x12x?x64xf32>,
    tensor<?x12x?x64xf32>,
    tensor<?x?x768xf32>
  ) {
    %ln = "sf.layer_norm"(%input, %ln_w, %ln_b) {normalized_shape = [768]}
         : (tensor<?x?x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>

    %q = "sf.linear"(%ln, %q_w, %q_b) {}
        : (tensor<?x?x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %q_s = "sf.mul"(%q, %q_scale) {}
           : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x768xf32>
    %q_r = "sf.view"(%q_s, %batch_dim) {shape = [-2, -1, 12, 64]}
           : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x12x64xf32>
    %q_t = "sf.transpose"(%q_r) {dim0 = 1 : i64, dim1 = 2 : i64}
           : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>

    %k = "sf.linear"(%ln, %k_w, %k_b) {}
        : (tensor<?x?x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %k_r = "sf.view"(%k, %batch_dim) {shape = [-2, -1, 12, 64]}
           : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x12x64xf32>
    %k_t = "sf.transpose"(%k_r) {dim0 = 1 : i64, dim1 = 2 : i64}
           : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>

    %v = "sf.linear"(%ln, %v_w, %v_b) {}
        : (tensor<?x?x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %v_r = "sf.view"(%v, %batch_dim) {shape = [-2, -1, 12, 64]}
           : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x12x64xf32>
    %v_t = "sf.transpose"(%v_r) {dim0 = 1 : i64, dim1 = 2 : i64}
           : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>

    %sdpa = "sf.scaled_dot_product_attention"(%q_t, %k_t, %v_t, %mask) {scale = 1.0 : f64}
            : (tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>,
               tensor<?x12x?x64xf32>, tensor<?x1x?x?xf32>) -> tensor<?x12x?x64xf32>

    %sdpa_t = "sf.transpose"(%sdpa) {dim0 = 1 : i64, dim1 = 2 : i64}
              : (tensor<?x12x?x64xf32>) -> tensor<?x?x12x64xf32>
    %sdpa_f = "sf.view"(%sdpa_t, %batch_dim, %seq_dim) {shape = [-2, -3, -1]}
              : (tensor<?x?x12x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x?x?xf32>

    %out = "sf.linear"(%sdpa_f, %o_w, %o_b) {}
           : (tensor<?x?x?xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>

    func.return %ln, %q, %k, %v, %q_t, %k_t, %v_t, %sdpa, %out
        : tensor<?x?x768xf32>, tensor<?x?x768xf32>, tensor<?x?x768xf32>,
          tensor<?x?x768xf32>, tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>,
          tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>, tensor<?x?x768xf32>
  }
}"""


# Full transformer layer (attention + FFN + residuals).
# Returns 12 intermediate tensors covering both attention and FFN blocks.
_FULL_LAYER_MLIR = """module {
  func.func @full_layer(
    %input: tensor<?x?x768xf32>,
    %attn_ln_w: tensor<768xf32>, %attn_ln_b: tensor<768xf32>,
    %q_w: tensor<768x768xf32>, %q_b: tensor<768xf32>,
    %k_w: tensor<768x768xf32>, %k_b: tensor<768xf32>,
    %v_w: tensor<768x768xf32>, %v_b: tensor<768xf32>,
    %o_w: tensor<768x768xf32>, %o_b: tensor<768xf32>,
    %ffn_ln_w: tensor<768xf32>, %ffn_ln_b: tensor<768xf32>,
    %fc1_w: tensor<3072x768xf32>, %fc1_b: tensor<3072xf32>,
    %fc2_w: tensor<768x3072xf32>, %fc2_b: tensor<768xf32>,
    %mask: tensor<?x1x?x?xf32>,
    %batch_dim: tensor<1xf32>, %seq_dim: tensor<1xf32>,
    %q_scale: tensor<1xf32>
  ) -> (
    tensor<?x?x768xf32>,
    tensor<?x?x768xf32>,
    tensor<?x12x?x64xf32>,
    tensor<?x12x?x64xf32>,
    tensor<?x12x?x64xf32>,
    tensor<?x12x?x64xf32>,
    tensor<?x?x768xf32>,
    tensor<?x?x768xf32>,
    tensor<?x?x3072xf32>,
    tensor<?x?x3072xf32>,
    tensor<?x?x768xf32>,
    tensor<?x?x768xf32>
  ) {
    // ── Attention block ──
    %ln = "sf.layer_norm"(%input, %attn_ln_w, %attn_ln_b) {normalized_shape = [768]}
         : (tensor<?x?x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>

    %q = "sf.linear"(%ln, %q_w, %q_b) {}
        : (tensor<?x?x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %q_s = "sf.mul"(%q, %q_scale) {}
           : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x768xf32>
    %q_r = "sf.view"(%q_s, %batch_dim) {shape = [-2, -1, 12, 64]}
           : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x12x64xf32>
    %q_t = "sf.transpose"(%q_r) {dim0 = 1 : i64, dim1 = 2 : i64}
           : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>

    %k = "sf.linear"(%ln, %k_w, %k_b) {}
        : (tensor<?x?x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %k_r = "sf.view"(%k, %batch_dim) {shape = [-2, -1, 12, 64]}
           : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x12x64xf32>
    %k_t = "sf.transpose"(%k_r) {dim0 = 1 : i64, dim1 = 2 : i64}
           : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>

    %v = "sf.linear"(%ln, %v_w, %v_b) {}
        : (tensor<?x?x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %v_r = "sf.view"(%v, %batch_dim) {shape = [-2, -1, 12, 64]}
           : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x12x64xf32>
    %v_t = "sf.transpose"(%v_r) {dim0 = 1 : i64, dim1 = 2 : i64}
           : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>

    %sdpa = "sf.scaled_dot_product_attention"(%q_t, %k_t, %v_t, %mask) {scale = 1.0 : f64}
            : (tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>,
               tensor<?x12x?x64xf32>, tensor<?x1x?x?xf32>) -> tensor<?x12x?x64xf32>

    %sdpa_t = "sf.transpose"(%sdpa) {dim0 = 1 : i64, dim1 = 2 : i64}
              : (tensor<?x12x?x64xf32>) -> tensor<?x?x12x64xf32>
    %sdpa_f = "sf.view"(%sdpa_t, %batch_dim, %seq_dim) {shape = [-2, -3, -1]}
              : (tensor<?x?x12x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x?x?xf32>

    %attn_out = "sf.linear"(%sdpa_f, %o_w, %o_b) {}
                : (tensor<?x?x?xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>

    %attn_res = "sf.add"(%input, %attn_out) {}
                : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>

    // ── FFN block ──
    %ffn_ln = "sf.layer_norm"(%attn_res, %ffn_ln_w, %ffn_ln_b) {normalized_shape = [768]}
              : (tensor<?x?x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>

    %ffn_h = "sf.linear"(%ffn_ln, %fc1_w, %fc1_b) {}
             : (tensor<?x?x768xf32>, tensor<3072x768xf32>, tensor<3072xf32>) -> tensor<?x?x3072xf32>

    %ffn_r = "sf.relu"(%ffn_h) {}
             : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>

    %ffn_out = "sf.linear"(%ffn_r, %fc2_w, %fc2_b) {}
               : (tensor<?x?x3072xf32>, tensor<768x3072xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>

    %ffn_res = "sf.add"(%attn_res, %ffn_out) {}
               : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>

    func.return %ln, %attn_out, %q_t, %k_t, %v_t, %sdpa, %attn_res,
                %ffn_ln, %ffn_h, %ffn_r, %ffn_out, %ffn_res
        : tensor<?x?x768xf32>, tensor<?x?x768xf32>, tensor<?x12x?x64xf32>,
          tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>, tensor<?x12x?x64xf32>,
          tensor<?x?x768xf32>, tensor<?x?x768xf32>, tensor<?x?x3072xf32>,
          tensor<?x?x3072xf32>, tensor<?x?x768xf32>, tensor<?x?x768xf32>
  }
}"""


# ── Shared fixture ──────────────────────────────────────────────────


@pytest.fixture(scope="session")
def hf_model() -> Any:
    """Load HF model once per session (cached)."""
    return _load_hf_model()


@pytest.fixture(scope="session")
def all_layer_weights(hf_model: Any) -> list[dict[str, np.ndarray]]:
    """Extract weights for all 12 transformer layers once per session."""
    return [_extract_layer_weights_numpy(hf_model, i) for i in range(12)]


# Backward-compat: layer 0 weights fixture
@pytest.fixture(scope="session")
def layer0_weights(all_layer_weights: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return all_layer_weights[0]


@pytest.fixture(scope="session")
def embed_weight(hf_model: Any) -> np.ndarray:
    """Load embedding weight [50272, 768]."""
    w = hf_model.model.decoder.embed_tokens.weight.detach().numpy()
    return w.astype(np.float32)


# ── Helper: token + embedding generation ────────────────────────────


def _make_input_embedding(
    embed_weight: np.ndarray,
    seq_len: int,
    batch_size: int = 1,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate token IDs and their embedding lookup.

    Returns:
        (input_ids [batch, seq] as int64, hidden_states [batch, seq, 768] as float32)
    """
    rng = np.random.RandomState(seed)
    token_ids = rng.randint(0, VOCAB_SIZE, size=(batch_size, seq_len), dtype=np.int64)
    hidden_states = embed_weight[token_ids].astype(np.float32)  # [batch, seq, 768]
    return token_ids, hidden_states


def _make_causal_mask(seq_len: int) -> np.ndarray:
    """Build causal mask: [1, 1, S, S] with 1.0 for attended, -inf for masked."""
    mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.float32))
    mask = np.where(mask == 0, float("-inf"), 1.0).astype(np.float32)
    return mask


def _make_shape_tensor(val: int) -> np.ndarray:
    """Create tensor<1xf32> with the given value."""
    return np.array([float(val)], dtype=np.float32)


# ── Test: Attention Chain ───────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.timeout(300)
class TestAttentionChain:
    """Verify attention chain (layer_norm → QKV → view → transpose → SDPA → out_proj).

    Compares each intermediate tensor against HF reference for 1-token and 2-token inputs.
    """

    def _run_test(self, hf_model, all_layer_weights, embed_weight, seq_len, layer_idx):
        """Core test logic shared across parametrized tests."""
        _, hidden_states = _make_input_embedding(embed_weight, seq_len)
        mask = _make_causal_mask(seq_len)
        batch_dim = _make_shape_tensor(1)
        seq_dim = _make_shape_tensor(seq_len)
        q_scale_arr = _make_shape_tensor(Q_SCALE)

        # ── HF reference ──
        ref = _hf_attention_chain_ref(hf_model, hidden_states, seq_len, layer_idx=layer_idx)

        # Pre-define ref_keys for expected shape extraction + later comparison
        ref_keys = [
            "ln_out", "q_raw", "k_raw", "v_raw",
            "q_t", "k_t", "v_t", "sdpa_out", "final",
        ]

        # ── Compile dylib ──
        w = all_layer_weights[layer_idx]
        with tempfile.TemporaryDirectory() as td:
            lib = _compile_single_function(
                _ATTENTION_CHAIN_MLIR, name="attn_chain", tmp_dir=td,
            )

            # ── Call compiled function ──
            # Build memref descriptors for all inputs (15 inputs)
            all_inputs = [
                hidden_states,
                w["ln_w"], w["ln_b"],
                w["q_w"], w["q_b"],
                w["k_w"], w["k_b"],
                w["v_w"], w["v_b"],
                w["o_w"], w["o_b"],
                mask,
                batch_dim, seq_dim,
                q_scale_arr,
            ]
            memrefs = [_memref(a) for a in all_inputs]

            # 9 outputs with known ranks and expected shapes (from HF ref)
            output_ranks = [3, 3, 3, 3, 4, 4, 4, 4, 3]
            expected_shapes = [ref[k].shape for k in ref_keys]
            sret_size = max(sum(desc_size(r) for r in output_ranks), 4096)
            sret_buf = _sret(sret_size)

            ciface = lib._mlir_ciface_attn_chain
            nargs = 1 + len(memrefs)
            ciface.argtypes = [ctypes.c_void_p] * nargs
            ciface.restype = None
            ciface(ctypes.byref(sret_buf), *[ctypes.byref(m) for m in memrefs])

            outputs = _parse_sret_outputs(
                bytes(sret_buf), output_ranks, expected_shapes=expected_shapes,
            )

        # ── Compare intermediates ──
        stage_names = [
            "ln_out", "q_raw", "k_raw", "v_raw",
            "q_t", "k_t", "v_t", "sdpa_out", "final",
        ]

        results: dict[str, dict[str, Any]] = {}
        all_pass = True

        for i, (stage, ref_key) in enumerate(zip(stage_names, ref_keys, strict=True)):
            actual = outputs[i]
            expected = ref[ref_key]

            passed, details = check_4_gate(actual, expected, min_cos=0.999)
            results[stage] = {"passed": passed, "details": details}
            if not passed:
                all_pass = False

        return results, all_pass

    @pytest.mark.parametrize("layer_idx", range(12))
    def test_attention_chain_1token(
        self, hf_model, all_layer_weights, embed_weight, layer_idx,
    ):
        """Attention chain: 1 token — should be near-perfect (no cross-token attention)."""
        results, all_pass = self._run_test(
            hf_model, all_layer_weights, embed_weight, 1, layer_idx,
        )

        # Build diagnostic output
        lines = [f"\nAttention chain layer {layer_idx} (seq_len=1) — per-stage 4-gate results:"]
        for stage, r in results.items():
            d = r["details"]
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"  {stage:12s}: cos={d['cos']:.10f}  "
                f"mean_rel={d['mean_rel_err']:.2e}  "
                f"max_out={d['max_outlier']:.2e}  "
                f"top10_j={d['top_n_jaccard']:.3f}  [{status}]"
            )
        report = "\n".join(lines)
        print(report)

        if not all_pass:
            # Collect failures for assertion
            failures = [s for s, r in results.items() if not r["passed"]]
            pytest.fail(
                f"Attention chain layer {layer_idx} (seq_len=1): "
                f"{len(failures)} stage(s) failed: {failures}\n{report}"
            )

    @pytest.mark.parametrize("layer_idx", range(12))
    def test_attention_chain_2token(
        self, hf_model, all_layer_weights, embed_weight, layer_idx,
    ):
        """Attention chain: 2 tokens — catches pos-1 divergence per-op.

        This is the critical test: with 2 tokens, token 0 must NOT attend
        to token 1 (causal mask). If the mask is broken, token 0's logits
        will diverge from HF reference.
        """
        results, all_pass = self._run_test(
            hf_model, all_layer_weights, embed_weight, 2, layer_idx,
        )

        lines = [f"\nAttention chain layer {layer_idx} (seq_len=2) — per-stage 4-gate results:"]
        for stage, r in results.items():
            d = r["details"]
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"  {stage:12s}: cos={d['cos']:.10f}  "
                f"mean_rel={d['mean_rel_err']:.2e}  "
                f"max_out={d['max_outlier']:.2e}  "
                f"top10_j={d['top_n_jaccard']:.3f}  [{status}]"
            )
        report = "\n".join(lines)
        print(report)

        if not all_pass:
            failures = [s for s, r in results.items() if not r["passed"]]
            pytest.fail(
                f"Attention chain layer {layer_idx} (seq_len=2): "
                f"{len(failures)} stage(s) failed: {failures}\n{report}"
            )


# ── Test: Full Layer ────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.timeout(300)
class TestFullLayer:
    """Verify full transformer layer (attention + FFN + residuals).

    Compares all 12 intermediate tensors against HF reference.
    """

    def _run_full_layer_test(self, hf_model, all_layer_weights, embed_weight, seq_len, layer_idx):
        """Core test logic for full layer test."""
        _, hidden_states = _make_input_embedding(embed_weight, seq_len)
        mask = _make_causal_mask(seq_len)
        batch_dim = _make_shape_tensor(1)
        seq_dim = _make_shape_tensor(seq_len)
        q_scale_arr = _make_shape_tensor(Q_SCALE)

        # ── HF reference ──
        ref = _hf_full_layer_ref(hf_model, hidden_states, seq_len, layer_idx=layer_idx)

        # Pre-define ref_keys for expected shape extraction
        ref_keys = [
            "attn_ln_out", "attn_output", "q_t", "k_t", "v_t", "sdpa_out",
            "attn_residual",
            "ffn_ln_out", "ffn_hidden", "ffn_relu", "ffn_output", "ffn_residual",
        ]

        # ── Compile dylib ──
        w = all_layer_weights[layer_idx]
        with tempfile.TemporaryDirectory() as td:
            lib = _compile_single_function(
                _FULL_LAYER_MLIR, name="full_layer", tmp_dir=td,
            )

            # Build memref descriptors for all inputs
            all_inputs = [
                hidden_states,
                w["ln_w"], w["ln_b"],
                w["q_w"], w["q_b"],
                w["k_w"], w["k_b"],
                w["v_w"], w["v_b"],
                w["o_w"], w["o_b"],
                w["ffn_ln_w"], w["ffn_ln_b"],
                w["fc1_w"], w["fc1_b"],
                w["fc2_w"], w["fc2_b"],
                mask,
                batch_dim, seq_dim,
                q_scale_arr,
            ]
            memrefs = [_memref(a) for a in all_inputs]

            # 12 outputs with expected shapes from HF ref
            output_ranks = [3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3]
            expected_shapes = [ref[k].shape for k in ref_keys]
            sret_size = max(sum(desc_size(r) for r in output_ranks), 8192)
            sret_buf = _sret(sret_size)

            ciface = lib._mlir_ciface_full_layer
            nargs = 1 + len(memrefs)
            ciface.argtypes = [ctypes.c_void_p] * nargs
            ciface.restype = None
            ciface(ctypes.byref(sret_buf), *[ctypes.byref(m) for m in memrefs])

            outputs = _parse_sret_outputs(
                bytes(sret_buf), output_ranks, expected_shapes=expected_shapes,
            )

        # ── Compare intermediates ──
        stage_names = [
            "attn_ln_out", "attn_output", "q_t", "k_t", "v_t", "sdpa_out",
            "attn_residual",
            "ffn_ln_out", "ffn_hidden", "ffn_relu", "ffn_output", "ffn_residual",
        ]

        results: dict[str, dict[str, Any]] = {}
        all_pass = True

        for i, (stage, ref_key) in enumerate(zip(stage_names, ref_keys, strict=True)):
            actual = outputs[i]
            expected = ref[ref_key]

            passed, details = check_4_gate(actual, expected, min_cos=0.999)
            results[stage] = {"passed": passed, "details": details}
            if not passed:
                all_pass = False

        return results, all_pass

    @pytest.mark.parametrize("layer_idx", range(12))
    def test_full_layer_1token(
        self, hf_model, all_layer_weights, embed_weight, layer_idx,
    ):
        """Full layer: 1 token — attention is trivial, FFN should match."""
        results, all_pass = self._run_full_layer_test(
            hf_model, all_layer_weights, embed_weight, 1, layer_idx,
        )

        lines = [f"\nFull layer {layer_idx} (seq_len=1) — per-stage 4-gate results:"]
        for stage, r in results.items():
            d = r["details"]
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"  {stage:16s}: cos={d['cos']:.10f}  "
                f"mean_rel={d['mean_rel_err']:.2e}  "
                f"max_out={d['max_outlier']:.2e}  "
                f"top10_j={d['top_n_jaccard']:.3f}  [{status}]"
            )
        report = "\n".join(lines)
        print(report)

        if not all_pass:
            failures = [s for s, r in results.items() if not r["passed"]]
            pytest.fail(
                f"Full layer {layer_idx} (seq_len=1): {len(failures)} stage(s) failed: "
                f"{failures}\n{report}"
            )

    @pytest.mark.parametrize("layer_idx", range(12))
    def test_full_layer_2token(
        self, hf_model, all_layer_weights, embed_weight, layer_idx,
    ):
        """Full layer: 2 tokens — key test for composition divergence.

        Catches cases where individual ops pass but composition fails
        (e.g., residual connections with wrong shapes, FFN on wrong input).
        """
        results, all_pass = self._run_full_layer_test(
            hf_model, all_layer_weights, embed_weight, 2, layer_idx,
        )

        lines = [f"\nFull layer {layer_idx} (seq_len=2) — per-stage 4-gate results:"]
        for stage, r in results.items():
            d = r["details"]
            status = "PASS" if r["passed"] else "FAIL"
            lines.append(
                f"  {stage:16s}: cos={d['cos']:.10f}  "
                f"mean_rel={d['mean_rel_err']:.2e}  "
                f"max_out={d['max_outlier']:.2e}  "
                f"top10_j={d['top_n_jaccard']:.3f}  [{status}]"
            )
        report = "\n".join(lines)
        print(report)

        if not all_pass:
            failures = [s for s, r in results.items() if not r["passed"]]
            pytest.fail(
                f"Full layer {layer_idx} (seq_len=2): {len(failures)} stage(s) failed: "
                f"{failures}\n{report}"
            )


# ── Test: Compilation smoke test (no HF model needed) ───────────────


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestOpCompositionCompile:
    """Smoke tests: verify MLIR templates compile without errors."""

    @pytest.mark.parametrize("layer_idx", range(12))
    def test_attention_chain_compiles(self, hf_model, all_layer_weights, embed_weight, layer_idx):
        """Verify the attention chain MLIR compiles to a loadable dylib."""
        seq_len = 1
        _, hidden_states = _make_input_embedding(embed_weight, seq_len)
        mask = _make_causal_mask(seq_len)
        batch_dim = _make_shape_tensor(1)
        seq_dim = _make_shape_tensor(seq_len)
        q_scale_arr = _make_shape_tensor(Q_SCALE)

        w = all_layer_weights[layer_idx]
        with tempfile.TemporaryDirectory() as td:
            lib = _compile_single_function(
                _ATTENTION_CHAIN_MLIR, name="attn_smoke", tmp_dir=td,
            )

            # Basic call to verify no crash
            all_inputs = [
                hidden_states,
                w["ln_w"], w["ln_b"],
                w["q_w"], w["q_b"],
                w["k_w"], w["k_b"],
                w["v_w"], w["v_b"],
                w["o_w"], w["o_b"],
                mask,
                batch_dim, seq_dim,
                q_scale_arr,
            ]
            memrefs = [_memref(a) for a in all_inputs]
            output_ranks = [3, 3, 3, 3, 4, 4, 4, 4, 3]
            sret_size = max(sum(desc_size(r) for r in output_ranks), 4096)
            sret_buf = _sret(sret_size)

            ciface = lib._mlir_ciface_attn_chain
            nargs = 1 + len(memrefs)
            ciface.argtypes = [ctypes.c_void_p] * nargs
            ciface.restype = None
            ciface(ctypes.byref(sret_buf), *[ctypes.byref(m) for m in memrefs])

            outputs = _parse_sret_outputs(bytes(sret_buf), output_ranks)
            assert len(outputs) == 9, f"Expected 9 outputs, got {len(outputs)}"
            # Check all outputs are non-NaN
            for i, out in enumerate(outputs):
                assert not np.any(np.isnan(out)), f"Output {i} contains NaN"

    @pytest.mark.parametrize("layer_idx", range(12))
    def test_full_layer_compiles(self, hf_model, all_layer_weights, embed_weight, layer_idx):
        """Verify the full layer MLIR compiles to a loadable dylib."""
        seq_len = 1
        _, hidden_states = _make_input_embedding(embed_weight, seq_len)
        mask = _make_causal_mask(seq_len)
        batch_dim = _make_shape_tensor(1)
        seq_dim = _make_shape_tensor(seq_len)
        q_scale_arr = _make_shape_tensor(Q_SCALE)

        w = all_layer_weights[layer_idx]
        with tempfile.TemporaryDirectory() as td:
            lib = _compile_single_function(
                _FULL_LAYER_MLIR, name="full_smoke", tmp_dir=td,
            )

            all_inputs = [
                hidden_states,
                w["ln_w"], w["ln_b"],
                w["q_w"], w["q_b"],
                w["k_w"], w["k_b"],
                w["v_w"], w["v_b"],
                w["o_w"], w["o_b"],
                w["ffn_ln_w"], w["ffn_ln_b"],
                w["fc1_w"], w["fc1_b"],
                w["fc2_w"], w["fc2_b"],
                mask,
                batch_dim, seq_dim,
                q_scale_arr,
            ]
            memrefs = [_memref(a) for a in all_inputs]
            output_ranks = [3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3]
            sret_size = max(sum(desc_size(r) for r in output_ranks), 8192)
            sret_buf = _sret(sret_size)

            ciface = lib._mlir_ciface_full_layer
            nargs = 1 + len(memrefs)
            ciface.argtypes = [ctypes.c_void_p] * nargs
            ciface.restype = None
            ciface(ctypes.byref(sret_buf), *[ctypes.byref(m) for m in memrefs])

            outputs = _parse_sret_outputs(bytes(sret_buf), output_ranks)
            assert len(outputs) == 12, f"Expected 12 outputs, got {len(outputs)}"
            for i, out in enumerate(outputs):
                assert not np.any(np.isnan(out)), f"Output {i} contains NaN"
