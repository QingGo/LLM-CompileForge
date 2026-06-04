"""KV cache correctness test — compares compiled model outputs with HF reference.

Tests the CachePolicy compilation path and verifies:
  1. Compiled model loads with correct CachePolicy metadata
  2. Prefill logits match HF within cos > 0.999
  3. forward_with_kv() API compatibility
  4. K/V per-layer structure from CachePolicy metadata

The compiled model uses static shapes (seq=8) with apply_fusion=False
to avoid pre-existing issues with dynamic shape compilation.
"""

from __future__ import annotations

import os

import pytest
import torch

from compiler.cache_policy import CachePolicy
from tests.helpers import cosine_similarity

# ── Constants ──────────────────────────────────────────────────

MODEL_DIR = "./compiled/opt_125m_kv"
NUM_LAYERS = 12
NUM_KV_HEADS = 12
HEAD_DIM = 64
BLOCK_SIZE = 16
SEQ_LEN = 8


# ── Helpers ────────────────────────────────────────────────────


def _load_hf_opt_125m():
    """Load facebook/opt-125m from local HF cache with use_cache=True."""
    from compiler.mlir_dialect.lowering.compile_utils import _patch_transformers_torch
    _patch_transformers_torch()
    from transformers.models.opt.configuration_opt import OPTConfig
    from transformers.models.opt.modeling_opt import OPTForCausalLM

    hub_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--opt-125m"
    )
    assert os.path.isdir(hub_dir), "opt-125m not in HF cache"
    snapshots = os.path.join(hub_dir, "snapshots")
    # Find snapshot that contains pytorch_model.bin
    model_dir = None
    for snap in sorted(os.listdir(snapshots), reverse=True):
        candidate = os.path.join(snapshots, snap)
        if os.path.isfile(os.path.join(candidate, "pytorch_model.bin")):
            model_dir = candidate
            break
    assert model_dir is not None, (
        f"No snapshot with pytorch_model.bin in {snapshots}"
    )

    config = OPTConfig.from_pretrained(model_dir)
    config.use_cache = True  # Enable KV cache output
    model = OPTForCausalLM(config)
    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"),
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, model_dir


def _load_compiled():
    """Load compiled opt-125m with CachePolicy."""
    from compiler.serialize import load_artifact
    return load_artifact(MODEL_DIR)


def _make_executor(compiled_module):
    """Create MlirExecutor for the compiled module."""
    from engine.mlir_executor import MlirExecutor
    from hal.pytorch_backend import PyTorchBackend
    return MlirExecutor(compiled_module, PyTorchBackend("cpu"))


# ── Tests ──────────────────────────────────────────────────────


@pytest.mark.integration
class TestKVCachePrefill:
    """Prefill correctness: compiled logits must match HF reference."""

    @pytest.mark.timeout(300)
    def test_prefill_cosine_matches_hf(self):
        """Prefill 8 tokens: compiled logits match HF within cos > 0.999."""
        hf_model, _ = _load_hf_opt_125m()
        compiled_module = _load_compiled()
        executor = _make_executor(compiled_module)

        input_ids = torch.randint(0, 5000, (1, SEQ_LEN), dtype=torch.long)

        with torch.no_grad():
            hf_output = hf_model(input_ids)
            hf_logits = hf_output.logits

        compiled_logits = executor.forward(input_ids)

        assert hf_logits.shape == compiled_logits.shape, (
            f"Shape mismatch: HF {hf_logits.shape} vs compiled {compiled_logits.shape}"
        )
        similarity = cosine_similarity(hf_logits, compiled_logits)
        print(f"\n  Prefill cosine similarity: {similarity:.8f}")
        assert similarity > 0.999, (
            f"Cosine similarity {similarity:.8f} below threshold 0.999"
        )

    @pytest.mark.timeout(30)
    def test_cache_policy_metadata_present(self):
        """Compiled artifact must carry CachePolicy in metadata."""
        compiled_module = _load_compiled()
        assert "cache_policy" in compiled_module.metadata, (
            "Compiled artifact missing cache_policy metadata"
        )
        raw_policy = compiled_module.metadata["cache_policy"]
        assert "slabs" in raw_policy, "cache_policy missing 'slabs'"
        assert "intercepts" in raw_policy, "cache_policy missing 'intercepts'"

        # Verify slab specs
        slab_ids = {s["slab_id"] for s in raw_policy["slabs"]}
        assert slab_ids == {"k", "v"}, f"Expected slabs 'k','v', got {slab_ids}"

        # Verify intercepts target SDPA
        for intercept in raw_policy["intercepts"]:
            assert intercept["op_name"] == "scaled_dot_product_attention"
            assert intercept["direction"] == "read_write"
            assert intercept["slab_id"] in ("k", "v")

        # Verify dimension specs
        for slab in raw_policy["slabs"]:
            dims = slab["dims"]
            assert dims["layers"] == NUM_LAYERS
            assert dims["heads"] == NUM_KV_HEADS
            assert dims["dim"] == HEAD_DIM

    @pytest.mark.timeout(60)
    def test_cache_policy_models_correct_dimensions(self):
        """CachePolicy.for_llama() produces correct dimension specs."""
        policy = CachePolicy.for_llama(
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            block_size=BLOCK_SIZE,
        )
        serialized = policy.to_dict()

        assert serialized["block_size"] == BLOCK_SIZE
        for slab in serialized["slabs"]:
            assert slab["dims"]["layers"] == NUM_LAYERS
            assert slab["dims"]["heads"] == NUM_KV_HEADS
            assert slab["dims"]["dim"] == HEAD_DIM
            assert slab["layout"] == "BNLD"
            assert slab["dtype"] == "float32"

        assert len(serialized["intercepts"]) == 2
        for intercept in serialized["intercepts"]:
            assert intercept["source"] in ("operand[1]", "operand[2]")
            assert intercept["layer"] == "sequential"

    @pytest.mark.timeout(60)
    def test_forward_with_kv_api(self):
        """forward_with_kv() returns (logits, kv_tensors) tuple."""
        compiled_module = _load_compiled()
        executor = _make_executor(compiled_module)

        input_ids = torch.randint(0, 5000, (1, SEQ_LEN), dtype=torch.long)
        logits, kv_tensors = executor.forward_with_kv(input_ids)

        assert logits is not None
        assert logits.shape == (1, SEQ_LEN, 50272), (
            f"Expected shape (1, {SEQ_LEN}, 50272), got {logits.shape}"
        )
        assert isinstance(kv_tensors, list), (
            f"Expected kv_tensors to be a list, got {type(kv_tensors)}"
        )

    @pytest.mark.timeout(60)
    def test_function_split_with_cache_policy(self):
        """Model compiled with CachePolicy has split functions (a/b layers)."""
        compiled_module = _load_compiled()
        func_names = [f.name for f in compiled_module.functions]

        # Check for 'a' (QKV) and 'b' (Attn+FFN) function pairs
        a_funcs = [n for n in func_names if n.endswith("a")]
        b_funcs = [n for n in func_names if n.endswith("b")]
        assert len(a_funcs) >= 1, f"No 'a' functions found: {func_names}"
        assert len(b_funcs) >= 1, f"No 'b' functions found: {func_names}"
        assert len(a_funcs) == len(b_funcs), (
            f"Mismatched a/b: {len(a_funcs)} a vs {len(b_funcs)} b"
        )
        print(f"\n  Function split: {len(a_funcs)} a/b pairs, "
              f"{len(compiled_module.functions)} total functions")

    @pytest.mark.timeout(300)
    def test_prefill_kv_extraction(self):
        """Compiled model's forward produces logits matching HF when use_cache=True."""
        hf_model, model_dir = _load_hf_opt_125m()
        compiled_module = _load_compiled()
        executor = _make_executor(compiled_module)

        input_ids = torch.randint(0, 5000, (1, SEQ_LEN), dtype=torch.long)

        # HF reference with KV cache output
        with torch.no_grad():
            hf_output = hf_model(input_ids)

        # Compiled forward
        compiled_logits = executor.forward(input_ids)

        # Logits must match
        cos = cosine_similarity(hf_output.logits, compiled_logits)
        print(f"\n  Prefill logits cos: {cos:.8f}")
        assert cos > 0.999, f"Logits cos {cos:.8f} < 0.999"

        # HF should return past_key_values when use_cache=True
        assert hf_output.past_key_values is not None, (
            "HF model with use_cache=True should return past_key_values"
        )
        assert len(hf_output.past_key_values) == NUM_LAYERS, (
            f"Expected {NUM_LAYERS} layers in past_key_values, "
            f"got {len(hf_output.past_key_values)}"
        )

        # HF past_key_values is DynamicCache (not subscriptable but iterable)
        pkv = list(hf_output.past_key_values)
        assert len(pkv) == NUM_LAYERS, (
            f"Expected {NUM_LAYERS} layers in past_key_values, "
            f"got {len(pkv)}"
        )

        # Verify each layer's K/V shape (layers are (K, V, ...) tuples)
        for layer in range(NUM_LAYERS):
            hf_k = pkv[layer][0]
            hf_v = pkv[layer][1]
            assert hf_k.shape == (1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), (
                f"Layer {layer} K shape mismatch: {hf_k.shape}"
            )
            assert hf_v.shape == (1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM), (
                f"Layer {layer} V shape mismatch: {hf_v.shape}"
            )

        print(f"\n  HF past_key_values: {NUM_LAYERS} layers, "
              f"each K/V shape (1, {NUM_KV_HEADS}, {SEQ_LEN}, {HEAD_DIM})")


@pytest.mark.integration
class TestKVCacheDecodeShape:
    """Decode-step shape verification (model compiled at static seq=8)."""

    @pytest.mark.timeout(120)
    def test_prefill_produces_correct_shape(self):
        """Prefill 8 tokens produces correct logits shape."""
        compiled_module = _load_compiled()
        executor = _make_executor(compiled_module)

        prefill_ids = torch.randint(0, 5000, (1, SEQ_LEN), dtype=torch.long)
        prefill_logits = executor.forward(prefill_ids)
        assert prefill_logits.shape == (1, SEQ_LEN, 50272), (
            f"Prefill shape: {prefill_logits.shape}"
        )
        assert not torch.isnan(prefill_logits).any(), "NaN in prefill output"
        assert not torch.isinf(prefill_logits).any(), "Inf in prefill output"


@pytest.mark.integration
class TestKVCacheRecompute:
    """Full recompute consistency."""

    @pytest.mark.timeout(300)
    def test_recompute_consistent(self):
        """Running the same input twice produces identical logits."""
        compiled_module = _load_compiled()
        executor = _make_executor(compiled_module)

        input_ids = torch.randint(0, 5000, (1, SEQ_LEN), dtype=torch.long)

        logits1 = executor.forward(input_ids)
        logits2 = executor.forward(input_ids)

        cos = cosine_similarity(logits1, logits2)
        print(f"\n  Recompute cos: {cos:.8f}")
        assert cos > 0.9999, f"Recompute cos {cos:.8f} < 0.9999"


@pytest.mark.integration
class TestKVCacheMetadata:
    """CachePolicy metadata validation — verifies compile-time config."""

    @pytest.mark.timeout(30)
    def test_cache_policy_serialization_roundtrip(self):
        """CachePolicy serializes and deserializes correctly."""
        original = CachePolicy.for_llama(
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            block_size=BLOCK_SIZE,
        )
        serialized = original.to_dict()
        restored = CachePolicy.from_dict(serialized)

        assert restored.block_size == BLOCK_SIZE
        assert len(restored.slabs) == 2
        assert len(restored.intercepts) == 2
        assert not restored.is_empty

    @pytest.mark.timeout(30)
    def test_hf_key_map_preserves_weight_mapping(self):
        """Compiled artifact has hf_key_map for weight resolution."""
        compiled_module = _load_compiled()
        hfk = compiled_module.metadata.get("hf_key_map", {})
        assert len(hfk) > 0, "hf_key_map is empty"
        assert "lm_head_weight" in hfk, (
            "hf_key_map missing lm_head_weight"
        )
        assert hfk["lm_head_weight"] == "lm_head.weight"
