"""Tests for quantization calibration pipelines (Phase 2 Stages 2-4).

Covers SmoothQuant (W8A8), AWQ (W4A16), and FP8 KV Cache quantization.
All tests are CPU-feasible using small synthetic models and tensors.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests.helpers import assert_cosine_above

# ── Tiny helper model for calibration tests ──────────────


class TwoLayerMLP(nn.Module):
    def __init__(self, in_f: int = 32, hidden: int = 64, out_f: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ── SmoothQuant ──────────────────────────────────────────


@pytest.mark.unit
class TestSmoothQuantMath:
    def test_smoothing_factor_formula(self) -> None:
        from compiler.quantize.smoothquant import SmoothQuantCalibrator

        calib = SmoothQuantCalibrator(nn.Linear(1, 1), alpha=0.5)
        act_max = torch.tensor([4.0, 1.0, 0.25], dtype=torch.float32)
        w_max = torch.tensor([1.0, 4.0, 0.25], dtype=torch.float32)

        s = calib._compute_smoothing_factor(act_max, w_max)
        expected = torch.sqrt(act_max.float()) / torch.sqrt(w_max.float())
        assert torch.allclose(s, expected, atol=1e-6)

    def test_alpha_zero_migrates_nothing(self) -> None:
        from compiler.quantize.smoothquant import SmoothQuantCalibrator

        calib = SmoothQuantCalibrator(nn.Linear(1, 1), alpha=0.0)
        act_max = torch.tensor([4.0, 1.0], dtype=torch.float32)
        w_max = torch.tensor([2.0, 0.5], dtype=torch.float32)

        s = calib._compute_smoothing_factor(act_max, w_max)
        expected = 1.0 / w_max.float()
        assert torch.allclose(s, expected, atol=1e-6)

    def test_alpha_one_migrates_all(self) -> None:
        from compiler.quantize.smoothquant import SmoothQuantCalibrator

        calib = SmoothQuantCalibrator(nn.Linear(1, 1), alpha=1.0)
        act_max = torch.tensor([4.0, 1.0], dtype=torch.float32)
        w_max = torch.tensor([2.0, 0.5], dtype=torch.float32)

        s = calib._compute_smoothing_factor(act_max, w_max)
        expected = act_max.float()
        assert torch.allclose(s, expected, atol=1e-6)

    def test_output_invariant_after_smoothing(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(4, 32)
        weight = torch.randn(64, 32)

        ref = x @ weight.T

        act_max = x.abs().amax(dim=0)
        w_max = weight.abs().amax(dim=0)

        from compiler.quantize.smoothquant import SmoothQuantCalibrator

        calib = SmoothQuantCalibrator(nn.Linear(1, 1), alpha=0.5)
        s = calib._compute_smoothing_factor(act_max, w_max)

        smoothed_weight = weight.float() * s
        smoothed_act = x.float() / s

        result = smoothed_act @ smoothed_weight.T
        assert torch.allclose(result, ref.float(), atol=1e-4)


@pytest.mark.unit
class TestSmoothQuantCalibration:
    def test_calibrate_on_two_layer_mlp(self) -> None:
        from compiler.quantize.smoothquant import SmoothQuantCalibrator

        torch.manual_seed(7)
        model = TwoLayerMLP()
        model.eval()

        inputs = torch.randn(8, 32)
        dataloader = [(inputs,)]

        calib = SmoothQuantCalibrator(model, alpha=0.5)
        calib.calibrate(dataloader, num_samples=1)

        assert calib.num_layers_processed == 2
        assert "fc1" in calib.smoothing_factors
        assert "fc2" in calib.smoothing_factors
        assert calib.smoothing_factors["fc1"].shape == (32,)

    def test_output_cosine_after_calibration(self) -> None:
        from compiler.quantize.smoothquant import SmoothQuantCalibrator

        torch.manual_seed(13)
        model = TwoLayerMLP()
        model.eval()

        x = torch.randn(4, 32)
        with torch.no_grad():
            ref = model(x).clone()

        calib = SmoothQuantCalibrator(model, alpha=0.5)
        calib.calibrate([(x,)], num_samples=1)

        with torch.no_grad():
            result = model(x)

        assert_cosine_above(result, ref, threshold=0.94)

    def test_quantize_with_activation_scale(self) -> None:
        from compiler.quantize._utils import get_layer_by_name
        from compiler.quantize.smoothquant import SmoothQuantCalibrator

        torch.manual_seed(3)
        model = TwoLayerMLP(in_f=32, hidden=64, out_f=16)
        model.eval()

        x = torch.randn(8, 32)
        calib = SmoothQuantCalibrator(model, alpha=0.5)
        calib.calibrate([(x,)], num_samples=1)
        calib.quantize()

        layer = get_layer_by_name(model, "fc1")
        assert layer is not None
        assert hasattr(layer, "weight_quant")
        assert hasattr(layer, "weight_scale")
        assert hasattr(layer, "activation_scale")

        assert layer.weight_quant.dtype == torch.int8
        assert layer.weight_scale.dtype == torch.float32
        assert layer.activation_scale.shape == (32,)

    def test_calibrate_with_none_dataloader(self) -> None:
        from compiler.quantize.smoothquant import SmoothQuantCalibrator

        torch.manual_seed(1)
        model = TwoLayerMLP(in_f=32, hidden=64, out_f=16)
        calib = SmoothQuantCalibrator(model, alpha=0.5)
        calib.calibrate(dataloader=None, num_samples=1)

        assert calib.num_layers_processed >= 1

    def test_alpha_out_of_range_raises(self) -> None:
        from compiler.quantize.smoothquant import SmoothQuantCalibrator

        with pytest.raises(ValueError, match="alpha"):
            SmoothQuantCalibrator(nn.Linear(1, 1), alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            SmoothQuantCalibrator(nn.Linear(1, 1), alpha=-0.1)


# ── AWQ ──────────────────────────────────────────────────


@pytest.mark.unit
class TestAWQ:
    def test_salient_channel_identification(self) -> None:
        from compiler.quantize.awq import AWQQuantizer

        torch.manual_seed(42)
        model = TwoLayerMLP(in_f=32, hidden=64, out_f=16)
        model.eval()

        x = torch.randn(8, 32)
        aq = AWQQuantizer(model, group_size=128, salient_fraction=0.01)
        aq.identify_salient_channels([(x,)], num_samples=1)

        assert "fc1" in aq.salient_channels
        assert "fc2" in aq.salient_channels
        fc1_channels = aq.salient_channels["fc1"]
        assert len(fc1_channels) == 1
        assert fc1_channels.dtype == torch.int64

    def test_salient_fraction_floor_to_one(self) -> None:
        from compiler.quantize.awq import AWQQuantizer

        torch.manual_seed(1)
        model = TwoLayerMLP(in_f=8, hidden=16, out_f=8)
        model.eval()

        aq = AWQQuantizer(model, group_size=16, salient_fraction=0.01)
        aq.identify_salient_channels([(torch.randn(4, 8),)], num_samples=1)
        assert aq.num_layers_processed == 0

        for indices in aq.salient_channels.values():
            assert len(indices) == 1

    def test_grid_search_finds_scale(self) -> None:
        from compiler.quantize.awq import AWQQuantizer

        torch.manual_seed(7)
        model = TwoLayerMLP(in_f=32, hidden=64, out_f=16)
        model.eval()

        aq = AWQQuantizer(model, group_size=64, salient_fraction=0.01)
        aq.identify_salient_channels([(torch.randn(8, 32),)], num_samples=1)
        aq.find_optimal_scales([(torch.randn(8, 32),)], scale_range=(1.0, 1.3), n_grid=10)

        assert aq.num_layers_processed >= 1
        assert "fc1" in aq.optimal_scales
        assert aq.optimal_scales["fc1"] >= 1.0

    def test_quantize_stores_buffers(self) -> None:
        from compiler.quantize._utils import get_layer_by_name
        from compiler.quantize.awq import AWQQuantizer

        torch.manual_seed(13)
        model = TwoLayerMLP(in_f=32, hidden=64, out_f=16)
        model.eval()

        aq = AWQQuantizer(model, group_size=32, salient_fraction=0.01)
        aq.identify_salient_channels([(torch.randn(8, 32),)], num_samples=1)
        aq.find_optimal_scales([(torch.randn(8, 32),)], scale_range=(1.0, 1.2), n_grid=5)
        aq.quantize()

        layer = get_layer_by_name(model, "fc1")
        assert layer is not None
        assert hasattr(layer, "weight_quant")
        assert hasattr(layer, "weight_scale")
        assert hasattr(layer, "weight_zero")
        assert layer.weight_quant.dtype == torch.uint8

    def test_w4a16_output_after_awq(self) -> None:
        from compiler.quantize.awq import AWQQuantizer
        from kernels.quantize.w4a16_gemm import w4a16_gemm

        torch.manual_seed(42)
        model = TwoLayerMLP(in_f=32, hidden=64, out_f=16)
        model.eval()

        x = torch.randn(4, 32)
        with torch.no_grad():
            ref = model(x).clone()

        aq = AWQQuantizer(model, group_size=32, salient_fraction=0.05)
        aq.identify_salient_channels([(x,)], num_samples=1)
        aq.find_optimal_scales([(x,)], scale_range=(1.0, 1.2), n_grid=10)
        aq.quantize()

        fc1_out = w4a16_gemm(
            x,
            aq.weight_quant["fc1"],
            aq.weight_scales["fc1"],
            aq.weight_zeros["fc1"],
            group_size=32,
        )
        fc1_act = torch.relu(fc1_out)
        fc2_out = w4a16_gemm(
            fc1_act,
            aq.weight_quant["fc2"],
            aq.weight_scales["fc2"],
            aq.weight_zeros["fc2"],
            group_size=32,
        )

        assert_cosine_above(fc2_out, ref, threshold=0.92)

    def test_salient_fraction_validation(self) -> None:
        from compiler.quantize.awq import AWQQuantizer

        with pytest.raises(ValueError, match="salient_fraction"):
            AWQQuantizer(nn.Linear(1, 1), salient_fraction=0.0)
        with pytest.raises(ValueError, match="salient_fraction"):
            AWQQuantizer(nn.Linear(1, 1), salient_fraction=1.5)


# ── FP8 KV Cache ─────────────────────────────────────────


@pytest.mark.unit
class TestFP8KVCache:
    def test_quantize_dequantize_roundtrip(self) -> None:
        from compiler.quantize.fp8_kv_cache import FP8KVCacheQuantizer

        torch.manual_seed(42)
        k = torch.randn(16, 8, 128)  # [block_size, num_kv_heads, head_dim]
        v = torch.randn(16, 8, 128)

        q = FP8KVCacheQuantizer(block_size=16)
        k_q, k_s, v_q, v_s = q.quantize_kv(k, v)
        k_fp, v_fp = q.dequantize_kv(k_q, k_s, v_q, v_s)

        assert_cosine_above(k, k_fp, threshold=0.99)
        assert_cosine_above(v, v_fp, threshold=0.99)

    def test_zero_block_handling(self) -> None:
        from compiler.quantize.fp8_kv_cache import FP8KVCacheQuantizer

        k = torch.zeros(16, 4, 64)
        q = FP8KVCacheQuantizer(block_size=16)
        k_q, k_s = q.quantize_block(k)
        k_fp = q.dequantize_block(k_q, k_s)

        assert torch.allclose(k_fp, torch.zeros(16, 4, 64), atol=1e-6)

    def test_hardware_detection(self) -> None:
        from compiler.quantize.fp8_kv_cache import FP8KVCacheQuantizer

        q = FP8KVCacheQuantizer()
        assert q.check_hardware_support("NVIDIA H100")
        assert q.check_hardware_support("AMD MI300")
        assert not q.check_hardware_support("NVIDIA A100")
        assert not q.check_hardware_support("CPU")

    def test_small_scale_clamped(self) -> None:
        from compiler.quantize.fp8_kv_cache import FP8KVCacheQuantizer

        k = torch.randn(4, 2, 32) * 0.02
        q = FP8KVCacheQuantizer(block_size=4)
        k_q, k_s = q.quantize_block(k)
        k_fp = q.dequantize_block(k_q, k_s)

        assert_cosine_above(k, k_fp, threshold=0.95)

    def test_invalid_block_size_raises(self) -> None:
        from compiler.quantize.fp8_kv_cache import FP8KVCacheQuantizer

        with pytest.raises(ValueError, match="block_size"):
            FP8KVCacheQuantizer(block_size=0)


# ── Mixed Precision Config ───────────────────────────────


@pytest.mark.unit
class TestMixedPrecision:
    def test_default_strategy_returns_expected(self) -> None:
        from compiler.quantize.mixed_precision import MixedPrecisionConfig

        cfg = MixedPrecisionConfig()
        assert cfg.get_precision("model.layers.0.self_attn.q_proj") == "w8a8"
        assert cfg.get_precision("model.layers.0.self_attn.o_proj") == "w4a16"
        assert cfg.get_precision("model.layers.0.mlp.gate_proj") == "w8a8"
        assert cfg.get_precision("model.layers.0.mlp.down_proj") == "w4a16"
        assert cfg.get_precision("model.embed_tokens") == "fp16"
        assert cfg.get_precision("lm_head") == "fp16"

    def test_unknown_layer_falls_back_to_fp16(self) -> None:
        from compiler.quantize.mixed_precision import MixedPrecisionConfig

        cfg = MixedPrecisionConfig()
        assert cfg.get_precision("some_unknown_layer") == "fp16"

    def test_override_from_dict(self) -> None:
        from compiler.quantize.mixed_precision import MixedPrecisionConfig

        cfg = MixedPrecisionConfig.from_dict({"q_proj": "w4a16", "o_proj": "w8a8"})
        assert cfg.get_precision("model.layers.0.self_attn.q_proj") == "w4a16"
        assert cfg.get_precision("model.layers.0.self_attn.o_proj") == "w8a8"

    def test_validate_all_precisions_valid(self) -> None:
        from compiler.quantize.mixed_precision import MixedPrecisionConfig

        cfg = MixedPrecisionConfig()
        assert cfg.validate() is True

    def test_validate_rejects_invalid_precision(self) -> None:
        from compiler.quantize.mixed_precision import MixedPrecisionConfig

        cfg = MixedPrecisionConfig.from_dict({"q_proj": "int2"})
        assert cfg.validate() is False

    def test_to_dict_roundtrip(self) -> None:
        from compiler.quantize.mixed_precision import MixedPrecisionConfig

        cfg = MixedPrecisionConfig()
        d = cfg.to_dict()
        cfg2 = MixedPrecisionConfig.from_dict(d)
        assert cfg2.strategy == cfg.strategy
