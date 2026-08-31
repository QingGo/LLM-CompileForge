"""Tests for config-driven CachePolicy generation (S3-pre E1).

The compiler must derive slab dimensions from the model config; it must
never hard-code per-model layer/kv-head/head-dim triplets in compile.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import compiler.compile as compile_mod
from compiler.cache_policy import CachePolicy


def _llama_1b_config() -> dict[str, object]:
    return {
        "model_type": "llama",
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "hidden_size": 2048,
        "layer_types": None,
    }


def _llama_3b_config() -> dict[str, object]:
    return {
        "model_type": "llama",
        "num_hidden_layers": 28,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 3072,
        "layer_types": None,
    }


def _qwen_text_config() -> dict[str, object]:
    return {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "hidden_size": 1024,
        "layer_types": ["linear_attention"] * 18 + ["full_attention"] * 6,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }


class TestCachePolicyForConfig:
    """CachePolicy.for_config reads standard decoder attention dims from config."""

    @pytest.mark.parametrize(
        ("config", "layers", "kv_heads", "head_dim"),
        [
            (_llama_1b_config(), 16, 8, 64),
            (_llama_3b_config(), 28, 8, 128),
        ],
    )
    def test_llama_dims_match_config(
        self, config: dict[str, object], layers: int, kv_heads: int, head_dim: int
    ) -> None:
        policy = CachePolicy.for_config(config)
        assert not policy.is_empty
        assert len(policy.slabs) == 2
        for slab in policy.slabs:
            assert slab.slab_id in ("k", "v")
            assert slab.dims["layers"] == layers
            assert slab.dims["heads"] == kv_heads
            assert slab.dims["dim"] == head_dim

    def test_dense_qwen_text_config_extracts_correct_dims(self) -> None:
        """A dense Qwen text config has 24 layers / 2 KV heads / 256 dim."""
        dense = _qwen_text_config()
        dense["layer_types"] = ["full_attention"] * 24
        policy = CachePolicy.for_config(dense)
        for slab in policy.slabs:
            assert slab.dims["layers"] == 24
            assert slab.dims["heads"] == 2
            assert slab.dims["dim"] == 256

    def test_multimodal_text_config_is_used(self) -> None:
        """A Qwen3.5 multimodal config must resolve dims through text_config."""
        multimodal = SimpleNamespace(text_config=SimpleNamespace(**_qwen_text_config()))
        policy = CachePolicy.for_config(multimodal)
        assert len(policy.slabs) == 4
        assert {s.slab_id for s in policy.slabs} == {
            "k", "v", "recurrent_state", "conv_state",
        }

    def test_mixed_attention_gets_recurrent_and_conv_slabs(self) -> None:
        """Gated DeltaNet needs recurrent/conv state in addition to K/V."""
        policy = CachePolicy.for_config(_qwen_text_config())
        assert len(policy.slabs) == 4
        rec = next(s for s in policy.slabs if s.slab_id == "recurrent_state")
        conv = next(s for s in policy.slabs if s.slab_id == "conv_state")
        assert rec.dims["layers"] == 18
        assert rec.dims["heads"] == 16
        assert rec.dims["key_dim"] == 128
        assert rec.dims["value_dim"] == 128
        assert conv.dims["layers"] == 18
        assert conv.dims["channels"] == 6144
        assert conv.dims["kernel"] == 4
        assert policy.intercepts[-2].slab_id == "recurrent_state"
        assert policy.intercepts[-1].slab_id == "conv_state"

    def test_falls_back_to_attention_heads_and_hidden_size(self) -> None:
        config = {
            "model_type": "opt",
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_size": 768,
        }
        policy = CachePolicy.for_config(config)
        for slab in policy.slabs:
            assert slab.dims["layers"] == 12
            assert slab.dims["heads"] == 12
            assert slab.dims["dim"] == 64


class TestCompileConfigDrivenPolicy:
    """compile.py model targets declare config-driven policy instead of literals."""

    def test_compile_llama_1b_declares_config_driven_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_compile_model(cfg: compile_mod.CompileConfig) -> None:
            captured["cfg"] = cfg

        monkeypatch.setattr(compile_mod, "_compile_model", fake_compile_model)
        compile_mod.compile_llama_1b("/tmp/compileforge-e1-test-llama-1b")
        cfg = captured["cfg"]
        assert isinstance(cfg, compile_mod.CompileConfig)
        assert cfg.cache_policy_from_config is True
        assert cfg.cache_policy is None
        assert "attention_mask" not in cfg.example_kwargs
        assert "attention_mask" not in cfg.dynamic_shapes
        assert cfg.patch_causal_mask_to_none is True

    def test_compile_llama_3b_declares_config_driven_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_compile_model(cfg: compile_mod.CompileConfig) -> None:
            captured["cfg"] = cfg

        monkeypatch.setattr(compile_mod, "_compile_model", fake_compile_model)
        compile_mod.compile_llama_3b("/tmp/compileforge-e1-test-llama-3b")
        cfg = captured["cfg"]
        assert isinstance(cfg, compile_mod.CompileConfig)
        assert cfg.cache_policy_from_config is True
        assert cfg.cache_policy is None
        assert "attention_mask" not in cfg.example_kwargs
        assert "attention_mask" not in cfg.dynamic_shapes
        assert cfg.patch_causal_mask_to_none is True

    def test_compile_qwen_declares_config_driven_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_compile_model(cfg: compile_mod.CompileConfig) -> None:
            captured["cfg"] = cfg

        monkeypatch.setattr(compile_mod, "_compile_model", fake_compile_model)
        compile_mod.compile_qwen("/tmp/compileforge-e1-test-qwen")
        cfg = captured["cfg"]
        assert isinstance(cfg, compile_mod.CompileConfig)
        assert cfg.cache_policy_from_config is True
        assert cfg.cache_policy is None


class TestResolveCachePolicyFromModel:
    """The compile-time resolver never guesses dimensions."""

    def test_llama_model_gets_correct_policy(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(**_llama_1b_config()))
        policy = compile_mod._cache_policy_from_model(model)
        assert isinstance(policy, CachePolicy)
        for slab in policy.slabs:
            assert slab.dims["layers"] == 16
            assert slab.dims["heads"] == 8
            assert slab.dims["dim"] == 64

    def test_mixed_attention_model_gets_empty_policy_until_uplink(self) -> None:
        text_config = SimpleNamespace(**_qwen_text_config())
        model = SimpleNamespace(config=SimpleNamespace(text_config=text_config))
        policy = compile_mod._cache_policy_from_model(model)
        assert policy.is_empty
