from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from hal.interface import Buffer, Device, OpExecutor


class PyTorchDevice(Device):
    """PyTorch 设备实现。

    封装 torch.cuda.synchronize() 或 CPU 空操作，
    作为 HAL Device 的 PyTorch 后端起效。
    """

    def __init__(self, device_type: str = "cpu") -> None:
        self._device_type = device_type

    @property
    def device_type(self) -> str:
        return self._device_type

    def synchronize(self) -> None:
        if self._device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()


class PyTorchBuffer(Buffer):
    """PyTorch 缓冲区实现。

    包装一个 torch.Tensor 作为底层内存，提供数据传输和视图创建。
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    @property
    def data_ptr(self) -> int:
        return self._tensor.data_ptr()

    def copy_from(self, src: torch.Tensor) -> None:
        self._tensor.copy_(src)

    def copy_to(self, dst: torch.Tensor) -> None:
        dst.copy_(self._tensor)

    def create_tensor(self, shape: Any, dtype: Any, device: Any) -> torch.Tensor:
        if dtype != self._tensor.dtype:
            raise ValueError(f"Buffer dtype mismatch: {self._tensor.dtype} != {dtype}")
        if str(device) != str(self._tensor.device):
            raise ValueError(f"Buffer device mismatch: {self._tensor.device} != {device}")
        return self._tensor.view(shape)


class PyTorchBackend(OpExecutor):
    """PyTorch 算子执行器。

    将 op_name 映射到 torch 原生操作。
    支持的算子覆盖设计文档算子映射表中定义的全部 14 个 HAL 操作。
    """

    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)

    # ── 算子映射表 ──────────────────────────────────────────

    def _op_matmul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.matmul(inputs[0], inputs[1])

    def _op_add(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.add(inputs[0], inputs[1])

    def _op_mul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.mul(inputs[0], inputs[1])

    def _op_gelu(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return F.gelu(inputs[0])

    def _op_silu(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return F.silu(inputs[0])

    def _op_softmax(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", -1)
        return F.softmax(inputs[0], dim=dim)

    def _op_layer_norm(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        normalized_shape = kwargs.get("normalized_shape", x.shape[-1:])
        weight = inputs[1] if len(inputs) > 1 else None
        bias = inputs[2] if len(inputs) > 2 else None
        eps = kwargs.get("eps", 1e-5)
        return F.layer_norm(x, normalized_shape, weight, bias, eps=eps)

    def _op_rms_norm(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        weight = inputs[1] if len(inputs) > 1 else None
        eps = kwargs.get("eps", 1e-5)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + eps)
        if weight is not None:
            out = out * weight
        return out

    def _op_permute(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dims = kwargs["dims"]
        return inputs[0].permute(*dims)

    def _op_transpose(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim0 = kwargs.get("dim0", 0)
        dim1 = kwargs.get("dim1", 1)
        return inputs[0].transpose(dim0, dim1)

    def _op_scaled_dot_product_attention(
        self, inputs: list[torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        query, key, value = inputs[0], inputs[1], inputs[2]
        attn_mask = kwargs.get("attn_mask", None)
        dropout_p = kwargs.get("dropout_p", 0.0)
        is_causal = kwargs.get("is_causal", False)
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

    def _op_cat(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", 0)
        return torch.cat(inputs, dim=dim)

    def _op_slice(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        dim = kwargs.get("dim", 0)
        start = kwargs.get("start", 0)
        end = kwargs.get("end", x.shape[dim])
        step = kwargs.get("step", 1)
        slicing: list[Any] = [slice(None)] * x.ndim
        slicing[dim] = slice(start, end, step)
        return x[tuple(slicing)]

    def _op_view(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        shape = kwargs["shape"]
        resolved = list(shape)
        # Resolve -1 dimensions
        if -1 in resolved:
            neg_idx = resolved.index(-1)
            total_elements = inputs[0].numel()
            product_other = 1
            for i, s in enumerate(resolved):
                if i != neg_idx:
                    product_other *= s
            if product_other == 0 or total_elements % product_other != 0:
                resolved[neg_idx] = max(1, total_elements // max(1, product_other))
            else:
                resolved[neg_idx] = total_elements // product_other
        else:
            # No -1 in shape — if sizes don't match, compute a new leading dim
            total_elements = inputs[0].numel()
            product = 1
            for s in resolved:
                product *= s
            if product != total_elements:
                # Compute a new first dimension to match actual size
                remaining = 1
                for s in resolved[1:]:
                    remaining *= s
                if remaining > 0 and total_elements % remaining == 0:
                    resolved[0] = total_elements // remaining
        return inputs[0].reshape(*resolved)

    def _op_identity(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return inputs[0]

    def _op_relu(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return F.relu(inputs[0])

    def _op_embedding(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        # FX graph order: (weight, indices). F.embedding expects (indices, weight).
        indices = inputs[1].to(torch.long)
        return F.embedding(indices, inputs[0])

    def _op_unsqueeze(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", 0)
        return inputs[0].unsqueeze(dim)

    def _op_sub(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        # Handle both sub(a, b) and rsub(a, b) → a - b
        if len(inputs) >= 2:
            return inputs[0] - inputs[1]
        return inputs[0]

    def _op_max(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if len(inputs) >= 2:
            return torch.max(inputs[0], inputs[1])
        return inputs[0]

    def _op_ones_like(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if inputs:
            return torch.ones_like(inputs[0])
        shape = kwargs.get("shape", (1,))
        return torch.ones(shape, dtype=torch.float32)

    def _op_full_like(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if inputs:
            value = kwargs.get("fill_value", 0)
            return torch.full_like(inputs[0], value)
        shape = kwargs.get("shape", (1,))
        return torch.full(shape, 0, dtype=torch.float32)

    def _op_arange(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if inputs:
            end = int(inputs[0].item()) if inputs[0].numel() == 1 else 1
        else:
            end = kwargs.get("end", 1)
        start = kwargs.get("start", 0)
        return torch.arange(start, end, dtype=inputs[0].dtype if inputs else torch.float32)

    # ── 融合算子 ────────────────────────────────────────────

    def _op_fused_rms_norm_matmul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        rms_weight = inputs[1]
        mat_weight = inputs[-1]
        eps = kwargs.get("eps", 1e-5)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)
        x_norm = x_norm * rms_weight
        return torch.matmul(x_norm, mat_weight)

    def _op_fused_silu_mul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        gate = inputs[0]
        up = inputs[1]
        return F.silu(gate) * up

    # ── 公开接口 ────────────────────────────────────────────

    def execute(self, op_name: str, inputs: list[Any], **kwargs: Any) -> torch.Tensor:
        dispatch = {
            "matmul": self._op_matmul,
            "add": self._op_add,
            "mul": self._op_mul,
            "gelu": self._op_gelu,
            "silu": self._op_silu,
            "softmax": self._op_softmax,
            "layer_norm": self._op_layer_norm,
            "rms_norm": self._op_rms_norm,
            "permute": self._op_permute,
            "transpose": self._op_transpose,
            "scaled_dot_product_attention": self._op_scaled_dot_product_attention,
            "cat": self._op_cat,
            "slice": self._op_slice,
            "view": self._op_view,
            "identity": self._op_identity,
            "relu": self._op_relu,
            "embedding": self._op_embedding,
            "unsqueeze": self._op_unsqueeze,
            "sub": self._op_sub,
            "max": self._op_max,
            "ones_like": self._op_ones_like,
            "full_like": self._op_full_like,
            "arange": self._op_arange,
            "fused_rms_norm_matmul": self._op_fused_rms_norm_matmul,
            "fused_silu_mul": self._op_fused_silu_mul,
        }
        if op_name not in dispatch:
            raise ValueError(f"Unknown op: {op_name}. Available: {sorted(dispatch.keys())}")
        return dispatch[op_name](inputs, **kwargs)
