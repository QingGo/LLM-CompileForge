from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

_log = logging.getLogger("hal.pytorch.math")


class _MathOps:
    def _op_matmul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        a, b = inputs[0], inputs[1]
        if a.dtype != b.dtype:
            b = b.to(a.dtype)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("matmul: a=%s b=%s", a.shape, b.shape)
        return torch.matmul(a, b)

    def _op_linear(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        w = inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        if x.dtype != w.dtype:
            w = w.to(x.dtype)
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
        return F.linear(x, w, bias)

    def _op_add(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.add(inputs[0], inputs[1])

    def _op_mul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.mul(inputs[0], inputs[1])

    def _op_sub(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if len(inputs) >= 2:
            return inputs[0] - inputs[1]
        return inputs[0]

    def _op_neg(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return -inputs[0]

    def _op_pow(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if len(inputs) >= 2:
            return torch.pow(inputs[0], inputs[1])
        exponent = kwargs.get("exponent", 2)
        return torch.pow(inputs[0], exponent)

    def _op_max(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if len(inputs) >= 2:
            return torch.max(inputs[0], inputs[1])
        return inputs[0]


from python_runtime.hal.pytorch_backend import _register_handler  # noqa: E402

_register_handler("matmul", "_op_matmul", 2, 2)
_register_handler("linear", "_op_linear", 2, 3)
_register_handler("add", "_op_add", 2, 2)
_register_handler("mul", "_op_mul", 2, 2)
_register_handler("sub", "_op_sub", 2, 2)
_register_handler("neg", "_op_neg", 1, 1)
_register_handler("pow", "_op_pow", 1, 2)
_register_handler("max", "_op_max", 2, 2)
