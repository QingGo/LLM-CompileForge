"""Phase 2 cross-module integration tests.

Validates that Modules A (Quantization), B (Speculative Decoding),
and D (Tensor Parallelism) work correctly with real compiled model
artifacts.

Requires compiled models in outputs/compiled/.  Tests are marked 'integration'
and gracefully skip when models are absent.

Reference: design-phase2.md §3, §5.2
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from tests.helpers import assert_cosine_above  # noqa: F401

# ── SmoothQuant on compiled model weights ────────────────


@pytest.mark.integration
class TestSmoothQuantCompiled:
    def test_calibrate_and_quantize_tiny_llama_weights(self, module_tiny_llama: Any) -> None:
        from compiler.quantize.smoothquant import SmoothQuantCalibrator

        weights = module_tiny_llama.main.weights
        linear_weights: dict[str, torch.Tensor] = {k: v for k, v in weights.items() if v.dim() == 2}
        if not linear_weights:
            pytest.skip("No 2D weight tensors")

        w_name, w = list(linear_weights.items())[0]

        class SingleLayer(nn.Module):
            def __init__(self, weight: torch.Tensor) -> None:
                super().__init__()
                self.linear = nn.Linear(weight.size(1), weight.size(0))
                self.linear.weight.data = weight.clone()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = SingleLayer(w)
        model.eval()

        calib = SmoothQuantCalibrator(model, alpha=0.5)
        calib.calibrate([(torch.randn(1, w.size(1)),)], num_samples=1)

        assert calib.num_layers_processed == 1

        calib.quantize()
        assert hasattr(model.linear, "weight_quant")
        assert model.linear.weight_quant.dtype == torch.int8

    def test_w8a8_output_cosine_tiny_llama_layer(self, module_tiny_llama: Any) -> None:
        from kernels.quantize._utils import quantize_per_channel_int8
        from kernels.quantize.w8a8_gemm import w8a8_gemm

        weights = module_tiny_llama.main.weights
        linear_weights = {k: v for k, v in weights.items() if v.dim() == 2}
        if not linear_weights:
            pytest.skip("No 2D weight tensors in model")

        w_name, w = list(linear_weights.items())[0]
        in_f = w.size(1)
        x = torch.randn(2, in_f)

        w_int8, w_scale = quantize_per_channel_int8(w.float())
        result = w8a8_gemm(x, None, w_int8, w_scale)
        reference = x @ w.float().T

        assert_cosine_above(result, reference, threshold=0.99)


# ── AWQ on compiled model weights ────────────────────────


@pytest.mark.integration
class TestAWQCompiled:
    def test_identify_and_quantize_tiny_llama(self, module_tiny_llama: Any) -> None:
        from compiler.quantize.awq import AWQQuantizer

        weights = module_tiny_llama.main.weights
        linear_weights = {k: v for k, v in weights.items() if v.dim() == 2}
        if not linear_weights:
            pytest.skip("No 2D weights")

        w_name, w = list(linear_weights.items())[0]

        class SingleLayer(nn.Module):
            def __init__(self, weight: torch.Tensor) -> None:
                super().__init__()
                self.linear = nn.Linear(weight.size(1), weight.size(0))
                self.linear.weight.data = weight.clone()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        model = SingleLayer(w)
        model.eval()

        aq = AWQQuantizer(model, group_size=16, salient_fraction=0.05)
        aq.identify_salient_channels([(torch.randn(1, w.size(1)),)], num_samples=1)

        assert len(aq.salient_channels) >= 1

        aq.find_optimal_scales([(torch.randn(1, w.size(1)),)], scale_range=(1.0, 1.2), n_grid=5)
        aq.quantize()

        assert aq.num_layers_processed >= 1

    def test_w4a16_output_cosine_opt125m_dynamic(self, module_opt_125m_dynamic: Any) -> None:
        from kernels.quantize._utils import quantize_groupwise_int4
        from kernels.quantize.w4a16_gemm import w4a16_gemm

        weights = module_opt_125m_dynamic.main.weights
        linear_weights = {k: v for k, v in weights.items() if v.dim() == 2}
        if not linear_weights:
            pytest.skip("No 2D weight tensors in model")

        w_name, w = list(linear_weights.items())[0]
        in_f = w.size(1)
        x = torch.randn(4, in_f)

        qp, qs, qz = quantize_groupwise_int4(w.float(), group_size=128)
        result = w4a16_gemm(x, qp, qs, qz, group_size=128)
        reference = x @ w.float().T

        assert_cosine_above(result, reference, threshold=0.99)


# ── SpecDec with compiled model ──────────────────────────


@pytest.mark.integration
class TestSpecDecCompiled:
    def test_verifier_with_model_weights(self, module_tiny_llama: Any) -> None:
        from python_runtime.engine.speculative.verifier import SpeculativeVerifier

        weights = module_tiny_llama.main.weights
        vocab_size = weights["model_embed_tokens_weight"].size(0)

        draft = torch.tensor([[5, 10]])
        logits = torch.randn(1, 3, vocab_size)

        verifier = SpeculativeVerifier()
        accepted, all_ok = verifier.verify_greedy(draft, logits)

        assert len(accepted) >= 1
        for tok in accepted:
            assert tok.shape == (1,)

    def test_mtp_proposer_with_compiled_hidden_size(self, module_tiny_llama: Any) -> None:
        from python_runtime.engine.speculative.mtp_proposer import MTPProposer

        weights = module_tiny_llama.main.weights
        hidden_size = weights["model_layers_0_self_attn_q_proj_weight"].size(0)
        vocab_size = weights["model_embed_tokens_weight"].size(0)

        proposer = MTPProposer(hidden_size, vocab_size, num_mtp_layers=2)
        h = torch.randn(1, hidden_size)
        drafts = proposer.propose(h, torch.tensor([[0]]), num_tokens=2)

        assert drafts.shape == (1, 2)
        assert drafts.dtype == torch.int64


# ── TP on compiled model layers ──────────────────────────


@pytest.mark.integration
class TestTPCompiled:
    def test_column_parallel_on_compiled_weight(self, module_tiny_llama: Any) -> None:
        from compiler.tp.linear import ColumnParallelLinear

        weights = module_tiny_llama.main.weights
        linear_weights = {k: v for k, v in weights.items() if v.dim() == 2}
        if not linear_weights:
            pytest.skip("No 2D weight tensors")

        w_name, w = list(linear_weights.items())[0]
        in_f, out_f = w.size(1), w.size(0)

        class _MockComm:
            rank = 0
            world_size = 1

            def all_gather(self, t: torch.Tensor) -> torch.Tensor:
                return t

            def all_reduce(self, t: torch.Tensor, op: str = "sum") -> torch.Tensor:
                return t

            def broadcast(self, t: torch.Tensor, src: int = 0) -> torch.Tensor:
                return t

        comm = _MockComm()
        tp_layer = ColumnParallelLinear(in_f, out_f, comm, bias=False)  # type: ignore[arg-type]
        ref_layer = nn.Linear(in_f, out_f, bias=False)

        with torch.no_grad():
            tp_layer.weight.copy_(w)
            ref_layer.weight.copy_(w)

            x = torch.randn(4, in_f)
            result = tp_layer(x)
            expected = ref_layer(x)

        assert_cosine_above(result, expected)


# ── Cross-module: Quantize + TP ──────────────────────────


@pytest.mark.integration
class TestQuantizeTPIntegration:
    def test_w8a8_quantized_weight_in_column_parallel(self, module_tiny_llama: Any) -> None:
        from compiler.tp.linear import ColumnParallelLinear
        from kernels.quantize._utils import dequantize_per_channel, quantize_per_channel_int8

        weights = module_tiny_llama.main.weights
        linear_weights = {k: v for k, v in weights.items() if v.dim() == 2}
        if not linear_weights:
            pytest.skip("No 2D weight tensors")

        w_name, w_orig = list(linear_weights.items())[0]
        in_f, out_f = w_orig.size(1), w_orig.size(0)

        w_int8, w_scale = quantize_per_channel_int8(w_orig.float())
        w_deq = dequantize_per_channel(w_int8, w_scale)

        class _MockComm:
            rank = 0
            world_size = 1

            def all_gather(self, t: torch.Tensor) -> torch.Tensor:
                return t

            def all_reduce(self, t: torch.Tensor, op: str = "sum") -> torch.Tensor:
                return t

            def broadcast(self, t: torch.Tensor, src: int = 0) -> torch.Tensor:
                return t

        comm = _MockComm()
        tp_layer = ColumnParallelLinear(in_f, out_f, comm, bias=False)  # type: ignore[arg-type]
        ref_layer = nn.Linear(in_f, out_f, bias=False)

        with torch.no_grad():
            tp_layer.weight.copy_(w_deq)
            ref_layer.weight.copy_(w_orig)

            x = torch.randn(4, in_f)
            result = tp_layer(x)
            expected = ref_layer(x)

        assert_cosine_above(result, expected, threshold=0.99)
