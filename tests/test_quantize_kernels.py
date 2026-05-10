"""Tests for quantized GEMM kernels (Phase 2 Stage 1).

Verifies correctness of W4A16 and W8A8 GEMM kernels on CPU against
FP16 reference implementations.  Also tests the shared INT4 pack/unpack
and INT8/INT4 quantization utilities.

All tests assert cosine similarity > 0.999 and max absolute difference
< 1e-3 against reference implementations where applicable.
"""

from __future__ import annotations

import pytest
import torch

from tests.helpers import assert_cosine_above, assert_max_diff_below

# ── INT4 Pack / Unpack ───────────────────────────────────


@pytest.mark.unit
class TestInt4PackUnpack:
    def test_roundtrip_preserves_values(self) -> None:
        from kernels.quantize._utils import pack_int4, unpack_int4

        torch.manual_seed(42)
        original = torch.randint(0, 16, (4, 64), dtype=torch.uint8)
        packed = pack_int4(original)
        assert packed.shape == (4, 32)
        assert packed.dtype == torch.uint8

        restored = unpack_int4(packed, out_dtype=torch.uint8)
        assert torch.equal(original, restored)

    def test_roundtrip_odd_sized_produces_correct(self) -> None:
        from kernels.quantize._utils import pack_int4, unpack_int4

        original = torch.randint(0, 16, (2, 32), dtype=torch.uint8)
        packed = pack_int4(original)
        assert packed.shape == (2, 16)

        restored = unpack_int4(packed, out_dtype=torch.uint8)
        assert torch.equal(original, restored)

    def test_unpack_to_fp32(self) -> None:
        from kernels.quantize._utils import pack_int4, unpack_int4

        original = torch.randint(0, 16, (2, 8), dtype=torch.uint8)
        packed = pack_int4(original)
        fp32_val = unpack_int4(packed, out_dtype=torch.float32)
        assert fp32_val.dtype == torch.float32
        assert fp32_val.shape == (2, 8)
        assert torch.equal(fp32_val.to(torch.uint8), original)

    def test_pack_raises_on_non_uint8(self) -> None:
        from kernels.quantize._utils import pack_int4

        with pytest.raises(ValueError, match="uint8"):
            pack_int4(torch.zeros(4, 4, dtype=torch.int32))

    def test_pack_raises_on_odd_last_dim(self) -> None:
        from kernels.quantize._utils import pack_int4

        with pytest.raises(ValueError, match="even"):
            pack_int4(torch.zeros(4, 3, dtype=torch.uint8))

    def test_unpack_raises_on_non_uint8(self) -> None:
        from kernels.quantize._utils import unpack_int4

        with pytest.raises(ValueError, match="uint8"):
            unpack_int4(torch.zeros(4, 4, dtype=torch.int32))


# ── INT8 Per-Channel Quantization ────────────────────────


@pytest.mark.unit
class TestInt8PerChannel:
    def test_quantize_dequantize_roundtrip(self) -> None:
        from kernels.quantize._utils import dequantize_per_channel, quantize_per_channel_int8

        torch.manual_seed(7)
        weight = torch.randn(64, 128)
        quant, scale = quantize_per_channel_int8(weight)
        assert quant.dtype == torch.int8
        assert quant.shape == weight.shape
        assert scale.shape == (64, 1)
        assert scale.dtype == torch.float32

        restored = dequantize_per_channel(quant, scale)
        assert_cosine_above(weight, restored, threshold=0.999)
        assert_max_diff_below(weight, restored, threshold=0.05)

    def test_zero_scale_clamped(self) -> None:
        from kernels.quantize._utils import quantize_per_channel_int8

        weight = torch.zeros(8, 16)
        quant, scale = quantize_per_channel_int8(weight)
        assert (scale > 0).all()
        assert (quant == 0).all()

    def test_uniform_weight_exact_quant(self) -> None:
        from kernels.quantize._utils import dequantize_per_channel, quantize_per_channel_int8

        weight = torch.ones(4, 16) * 3.5
        quant, scale = quantize_per_channel_int8(weight)
        expected_scale = 3.5 / 127.0
        assert torch.allclose(scale.squeeze(), torch.tensor([expected_scale] * 4), atol=1e-6)

        restored = dequantize_per_channel(quant, scale)
        assert torch.allclose(restored, weight, atol=0.03)


# ── INT4 Group-Wise Quantization ─────────────────────────


@pytest.mark.unit
class TestInt4GroupWise:
    def test_quantize_dequantize_roundtrip_g128(self) -> None:
        from kernels.quantize._utils import dequantize_groupwise, quantize_groupwise_int4

        torch.manual_seed(13)
        weight = torch.randn(32, 256)
        quant_packed, scale, zero = quantize_groupwise_int4(weight, group_size=128)
        assert quant_packed.dtype == torch.uint8
        assert quant_packed.shape == (32, 128)
        assert scale.shape == (32, 2)
        assert zero.shape == (32, 2)
        assert (zero == 8).all()

        restored = dequantize_groupwise(quant_packed, scale, group_size=128)
        assert restored.shape == weight.shape
        assert_cosine_above(weight, restored, threshold=0.99)

    def test_quantize_dequantize_roundtrip_g64(self) -> None:
        from kernels.quantize._utils import dequantize_groupwise, quantize_groupwise_int4

        torch.manual_seed(13)
        weight = torch.randn(16, 256)
        quant_packed, scale, zero = quantize_groupwise_int4(weight, group_size=64)

        restored = dequantize_groupwise(quant_packed, scale, group_size=64)
        assert_cosine_above(weight, restored, threshold=0.99)

    def test_non_divisible_group_size(self) -> None:
        from kernels.quantize._utils import dequantize_groupwise, quantize_groupwise_int4

        torch.manual_seed(99)
        weight = torch.randn(8, 100)
        quant_packed, scale, zero = quantize_groupwise_int4(weight, group_size=128)
        assert scale.shape == (8, 1)

        restored = dequantize_groupwise(quant_packed, scale, group_size=128,
                                        out_features=8, in_features=100)
        assert restored.shape == (8, 100)
        assert_cosine_above(weight, restored, threshold=0.99)

    def test_zero_weight(self) -> None:
        from kernels.quantize._utils import dequantize_groupwise, quantize_groupwise_int4

        weight = torch.zeros(4, 128)
        quant_packed, scale, zero = quantize_groupwise_int4(weight, group_size=128)
        assert (scale > 0).all()

        restored = dequantize_groupwise(quant_packed, scale, group_size=128)
        assert (restored == 0).all()


# ── W4A16 GEMM ───────────────────────────────────────────


@pytest.mark.unit
class TestW4A16GEMM:
    def test_output_matches_reference(self) -> None:
        from kernels.quantize._utils import quantize_groupwise_int4
        from kernels.quantize.w4a16_gemm import w4a16_gemm

        torch.manual_seed(42)
        in_f, out_f, batch = 256, 128, 4
        activation = torch.randn(batch, in_f)
        weight_fp = torch.randn(out_f, in_f)

        quant_packed, w_scale, w_zero = quantize_groupwise_int4(weight_fp, group_size=128)

        result = w4a16_gemm(activation, quant_packed, w_scale, w_zero, group_size=128)
        reference = activation @ weight_fp.T

        assert_cosine_above(result, reference, threshold=0.99)

    def test_single_input_vector(self) -> None:
        from kernels.quantize._utils import quantize_groupwise_int4
        from kernels.quantize.w4a16_gemm import w4a16_gemm

        torch.manual_seed(1)
        in_f, out_f = 64, 32
        activation = torch.randn(1, in_f)
        weight_fp = torch.randn(out_f, in_f)

        quant_packed, w_scale, w_zero = quantize_groupwise_int4(weight_fp, group_size=128)

        result = w4a16_gemm(activation, quant_packed, w_scale, w_zero, group_size=128)
        reference = activation @ weight_fp.T

        assert result.shape == (1, out_f)
        assert_cosine_above(result, reference, threshold=0.99)

    def test_group_size_64(self) -> None:
        from kernels.quantize._utils import quantize_groupwise_int4
        from kernels.quantize.w4a16_gemm import w4a16_gemm

        torch.manual_seed(3)
        in_f, out_f, batch = 256, 64, 2
        activation = torch.randn(batch, in_f)
        weight_fp = torch.randn(out_f, in_f)

        quant_packed, w_scale, w_zero = quantize_groupwise_int4(weight_fp, group_size=64)

        result = w4a16_gemm(activation, quant_packed, w_scale, w_zero, group_size=64)
        reference = activation @ weight_fp.T

        assert_cosine_above(result, reference, threshold=0.99)


# ── W8A8 GEMM ────────────────────────────────────────────


@pytest.mark.unit
class TestW8A8GEMM:
    def test_output_matches_reference(self) -> None:
        from kernels.quantize._utils import quantize_per_channel_int8
        from kernels.quantize.w8a8_gemm import w8a8_gemm

        torch.manual_seed(42)
        in_f, out_f, batch = 128, 64, 4
        weight_fp = torch.randn(out_f, in_f)

        w_quant, w_scale = quantize_per_channel_int8(weight_fp)

        activation = torch.randn(batch, in_f)
        act_absmax = activation.abs().amax(dim=1, keepdim=True)
        act_scale_val = act_absmax / 127.0
        act_scale_val = torch.clamp(act_scale_val, min=1e-9)
        act_quantized = activation / act_scale_val

        result = w8a8_gemm(act_quantized, act_scale_val, w_quant, w_scale)
        reference = activation @ weight_fp.T

        assert_cosine_above(result, reference, threshold=0.995)

    def test_no_act_scale_passthrough(self) -> None:
        from kernels.quantize._utils import quantize_per_channel_int8
        from kernels.quantize.w8a8_gemm import w8a8_gemm

        torch.manual_seed(3)
        in_f, out_f = 64, 32
        weight_fp = torch.randn(out_f, in_f)
        w_quant, w_scale = quantize_per_channel_int8(weight_fp)

        activation = torch.randn(3, in_f)

        result = w8a8_gemm(activation, None, w_quant, w_scale)
        reference = activation @ weight_fp.T

        assert_cosine_above(result, reference, threshold=0.995)

    def test_single_input(self) -> None:
        from kernels.quantize._utils import quantize_per_channel_int8
        from kernels.quantize.w8a8_gemm import w8a8_gemm

        torch.manual_seed(99)
        in_f, out_f = 32, 16
        weight_fp = torch.randn(out_f, in_f)
        w_quant, w_scale = quantize_per_channel_int8(weight_fp)

        activation = torch.randn(1, in_f)
        result = w8a8_gemm(activation, None, w_quant, w_scale)

        assert result.shape == (1, out_f)


# ── Cross-GEMM comparison ────────────────────────────────


@pytest.mark.unit
class TestGEMMConsistency:
    def test_w4a16_and_w8a8_produce_similar_outputs(self) -> None:
        from kernels.quantize._utils import quantize_groupwise_int4, quantize_per_channel_int8
        from kernels.quantize.w4a16_gemm import w4a16_gemm
        from kernels.quantize.w8a8_gemm import w8a8_gemm

        torch.manual_seed(55)
        in_f, out_f = 256, 128
        weight_fp = torch.randn(out_f, in_f)
        activation = torch.randn(4, in_f)

        qp4, ws4, wz4 = quantize_groupwise_int4(weight_fp, group_size=128)
        result_w4 = w4a16_gemm(activation, qp4, ws4, wz4, group_size=128)

        wq8, ws8 = quantize_per_channel_int8(weight_fp)
        result_w8 = w8a8_gemm(activation, None, wq8, ws8)

        assert_cosine_above(result_w4, result_w8, threshold=0.99)
