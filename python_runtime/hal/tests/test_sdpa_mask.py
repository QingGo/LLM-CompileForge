# ruff: noqa: E501
"""Tests for SDPA mask conversion logic in _AttentionOps.

The _op_scaled_dot_product_attention handler converts positional masks
to causal additive masks. This test suite verifies:

1. is_causal=True → attn_mask set to None (PyTorch handles causal internally)
2. is_causal=False → positional mask [batch, 1, seq, 1] → additive [batch, 1, seq, seq]
   with -inf for masked-out (future) positions
3. No mask → attn_mask=None (passthrough)
4. Non-float mask → untouched (conversion only applies to float masks)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from python_runtime.hal.pytorch_backend._ops_attention import _AttentionOps


@pytest.mark.unit
class TestSDPAMaskConversion:

    # ── Helpers ────────────────────────────────────────────────────────

    def _make_qkv(self, batch: int = 2, seq: int = 4, head: int = 4, dim: int = 8) -> list[torch.Tensor]:
        """Create standard Q/K/V tensors for testing."""
        return [torch.randn(batch, head, seq, dim) for _ in range(3)]

    def _make_positional_mask(self, batch: int = 2, seq: int = 4) -> torch.Tensor:
        """Create a positional mask [batch, 1, seq, 1] with values 0..seq-1."""
        pos = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1)
        return pos.expand(batch, -1, -1, -1)

    # ── Tests ──────────────────────────────────────────────────────────

    def test_is_causal_true_clears_mask(self):
        """With is_causal=True, attn_mask should be set to None."""
        ops = _AttentionOps()
        qkv = self._make_qkv()
        pos_mask = self._make_positional_mask()

        mock_return = torch.randn_like(qkv[0])
        with patch(
            "hal.pytorch_backend._ops_attention.F.scaled_dot_product_attention",
            return_value=mock_return,
        ) as mock_sdpa:
            ops._op_scaled_dot_product_attention(
                [*qkv, pos_mask], is_causal=True, dropout_p=0.0,
            )
            _call_args, call_kwargs = mock_sdpa.call_args
            assert call_kwargs["attn_mask"] is None, (
                "is_causal=True should clear attn_mask to None"
            )

    def test_is_causal_false_converts_positional_to_additive(self):
        """With is_causal=False, positional [0,1,2,3] converts to causal additive mask."""
        ops = _AttentionOps()
        qkv = self._make_qkv(batch=1, seq=4)
        # Positional mask [1, 1, 4, 1] with values 0, 1, 2, 3
        pos_mask = torch.tensor([[[[0.0], [1.0], [2.0], [3.0]]]])

        mock_return = torch.randn_like(qkv[0])
        with patch(
            "hal.pytorch_backend._ops_attention.F.scaled_dot_product_attention",
            return_value=mock_return,
        ) as mock_sdpa:
            ops._op_scaled_dot_product_attention(
                [*qkv, pos_mask], is_causal=False, dropout_p=0.0,
            )
            _call_args, call_kwargs = mock_sdpa.call_args
            result_mask = call_kwargs["attn_mask"]

            assert result_mask is not None, "attn_mask should not be None"
            assert result_mask.shape == (1, 1, 4, 4), (
                f"Expected (1, 1, 4, 4) got {result_mask.shape}"
            )
            for i in range(4):
                for j in range(4):
                    if j > i:
                        assert result_mask[0, 0, i, j] == float("-inf"), (
                            f"Position {i} should NOT attend to {j}"
                        )
                    else:
                        assert result_mask[0, 0, i, j] == 0.0, (
                            f"Position {i} SHOULD attend to {j}"
                        )

    def test_no_mask_passed_through(self):
        """When no attn_mask is provided, attn_mask should remain None."""
        ops = _AttentionOps()
        qkv = self._make_qkv()

        mock_return = torch.randn_like(qkv[0])
        with patch(
            "hal.pytorch_backend._ops_attention.F.scaled_dot_product_attention",
            return_value=mock_return,
        ) as mock_sdpa:
            ops._op_scaled_dot_product_attention(
                qkv, is_causal=True,
            )
            _call_args, call_kwargs = mock_sdpa.call_args
            assert call_kwargs["attn_mask"] is None, (
                "No mask provided should keep attn_mask=None"
            )

    def test_non_float_mask_not_touched(self):
        """Integer mask (non-float) should pass through without conversion."""
        ops = _AttentionOps()
        qkv = self._make_qkv(batch=1)
        int_mask = torch.zeros(1, 1, 4, 4, dtype=torch.long)

        mock_return = torch.randn_like(qkv[0])
        with patch(
            "hal.pytorch_backend._ops_attention.F.scaled_dot_product_attention",
            return_value=mock_return,
        ) as mock_sdpa:
            ops._op_scaled_dot_product_attention(
                [*qkv, int_mask], is_causal=False,
            )
            _call_args, call_kwargs = mock_sdpa.call_args
            result_mask = call_kwargs["attn_mask"]
            assert result_mask is int_mask, (
                "Integer mask should be passed through unchanged"
            )
