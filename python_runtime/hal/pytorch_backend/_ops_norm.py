from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

_log = logging.getLogger("hal.pytorch.norm")


class _NormOps:
    def _op_softmax(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", -1)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("softmax: %s dim=%s", inputs[0].shape, dim)
        return F.softmax(inputs[0], dim=dim)

    def _op_layer_norm(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0].float()
        normalized_shape = kwargs.get("normalized_shape", x.shape[-1:])
        weight = inputs[1].float() if len(inputs) > 1 and inputs[1] is not None else None
        bias = inputs[2].float() if len(inputs) > 2 and inputs[2] is not None else None
        eps = kwargs.get("eps", 1e-5)
        result = F.layer_norm(x, normalized_shape, weight, bias, eps=eps)
        if len(inputs) > 1 and inputs[1] is not None:
            result = result.to(inputs[1].dtype)
        return result

    def _op_rms_norm(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        weight = inputs[1] if len(inputs) > 1 else None
        eps = kwargs.get("eps", 1e-5)
        orig_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x_f32 = x.float()
        else:
            x_f32 = x
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        out = x_f32 * torch.rsqrt(variance + eps)
        if weight is not None:
            if weight.dtype != out.dtype:
                weight = weight.to(out.dtype)
            out = out * weight
        return out.to(orig_dtype)

    def _op_fused_rms_norm_matmul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        rms_weight = inputs[1]
        mat_weight = inputs[-1]
        eps = kwargs.get("eps", 1e-5)
        orig_dtype = x.dtype
        if x.dtype in (torch.float16, torch.bfloat16):
            x_f32 = x.float()
        else:
            x_f32 = x
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + eps)
        if rms_weight.dtype != x_norm.dtype:
            rms_weight = rms_weight.to(x_norm.dtype)
        x_norm = x_norm * rms_weight
        if mat_weight.dtype != x_norm.dtype:
            mat_weight = mat_weight.to(x_norm.dtype)
        result = torch.matmul(x_norm, mat_weight)
        return result.to(orig_dtype)


from python_runtime.hal.pytorch_backend import _register_handler  # noqa: E402

_register_handler("softmax", "_op_softmax", 1, 2)
_register_handler("layer_norm", "_op_layer_norm", 1, 3)
_register_handler("rms_norm", "_op_rms_norm", 1, 2)
_register_handler("fused_rms_norm_matmul", "_op_fused_rms_norm_matmul", 2, 3)
