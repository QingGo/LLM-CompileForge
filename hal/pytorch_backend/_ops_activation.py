from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


class _ActivationOps:
    def _op_gelu(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return F.gelu(inputs[0])

    def _op_silu(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return F.silu(inputs[0])

    def _op_relu(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return F.relu(inputs[0])

    def _op_sigmoid(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.sigmoid(inputs[0])

    def _op_softplus(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return F.softplus(inputs[0])

    def _op_exp(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.exp(inputs[0])

    def _op_fused_silu_mul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        gate = inputs[0]
        up = inputs[1]
        return F.silu(gate) * up


from hal.pytorch_backend import _register_handler  # noqa: E402

_register_handler("gelu", "_op_gelu", 1, 1)
_register_handler("silu", "_op_silu", 1, 1)
_register_handler("relu", "_op_relu", 1, 1)
_register_handler("sigmoid", "_op_sigmoid", 1, 1)
_register_handler("softplus", "_op_softplus", 1, 1)
_register_handler("exp", "_op_exp", 1, 1)
_register_handler("fused_silu_mul", "_op_fused_silu_mul", 2, 2)
