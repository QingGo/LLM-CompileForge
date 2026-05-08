from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from hal.interface import Buffer, Device, OpExecutor

# ── Op registry (validates input counts before dispatch) ────


@dataclass
class _OpSpec:
    name: str
    min_inputs: int = 0
    max_inputs: int | None = None


_op_registry: dict[str, _OpSpec] = {}


def _register_op(name: str, min_inputs: int = 0, max_inputs: int | None = None) -> None:
    """Register an op with its input-count constraints."""
    _op_registry[name] = _OpSpec(name=name, min_inputs=min_inputs, max_inputs=max_inputs)


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
        _register_op("add", min_inputs=2, max_inputs=2)
        _register_op("mul", min_inputs=2, max_inputs=2)
        _register_op("sub", min_inputs=2, max_inputs=2)
        _register_op("matmul", min_inputs=2, max_inputs=2)
        _register_op("neg", min_inputs=1, max_inputs=1)
        _register_op("cos", min_inputs=1, max_inputs=1)
        _register_op("sin", min_inputs=1, max_inputs=1)
        _register_op("zeros", min_inputs=0, max_inputs=10)
        _register_op("copy_", min_inputs=2, max_inputs=2)

    # ── 算子映射表 ──────────────────────────────────────────

    def _op_matmul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        a, b = inputs[0], inputs[1]
        if a.dtype != b.dtype:
            b = b.to(a.dtype)
        return torch.matmul(a, b)

    def _op_linear(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        w = inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        if x.dtype != w.dtype:
            w = w.to(x.dtype)
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
        return torch.nn.functional.linear(x, w, bias)

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
        orig_dtype = x.dtype
        # Upcast to float32 for numerical stability (matching HF implementation)
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

    def _op_type_as(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return inputs[0].to(inputs[1].dtype)

    def _op_copy_(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if len(inputs) >= 2:
            dst, src = inputs[0], inputs[1]
            return dst.copy_(src)
        return inputs[0]

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
        if attn_mask is None and len(inputs) > 3:
            attn_mask = inputs[3]
        # Normalize attn_mask shape for dynamic batch.
        # In dynamic batch mode, the mask may be 5D: [batch, ?, 1, seq_q, seq_k]
        # where dim 1 spuriously got batch size propagated instead of 1.
        if attn_mask is not None and attn_mask.dim() > 4:
            attn_mask = attn_mask.squeeze(2)
        if attn_mask is not None and attn_mask.dim() == 4:
            bsize = query.shape[0]
            # Fix spurious batch propagation into heads dim: [B, B, S, S] → [B, 1, S, S]
            if attn_mask.shape[1] > 1 and attn_mask.shape[1] == bsize:
                attn_mask = attn_mask[:, :1, :, :]
        dropout_p = kwargs.get("dropout_p", 0.0)
        is_causal = kwargs.get("is_causal", False)
        # Honor the IR's scale attribute.  When the model pre-scales Q
        # (e.g. dynamic-shape exports), scale=1.0 in the IR indicates
        # "no additional scaling".  When absent, PyTorch's default
        # (1/√head_dim) is used.
        scale = kwargs.get("scale", None)
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale,
        )

    def _op_cat(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", 0)
        resolved = []
        for t in inputs:
            if t.ndim == 0:
                t = t.unsqueeze(0)
            resolved.append(t)
        return torch.cat(resolved, dim=dim)

    def _op_slice(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        dim = kwargs.get("dim", 0)
        start = kwargs.get("start", 0)
        end = kwargs.get("end", None)
        step = kwargs.get("step", 1)
        extra = list(inputs[1:])
        if end is None and extra:
            end = int(extra.pop(0).item())
        if not isinstance(start, int) and extra:
            start = int(extra.pop(0).item())
        if not isinstance(dim, int) and extra:
            dim = int(extra.pop(0).item())
        if end is None:
            end = x.shape[dim]
        slicing: list[Any] = [slice(None)] * x.ndim
        slicing[dim] = slice(int(start), int(end), int(step))
        return x[tuple(slicing)]

    def _op_view(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        raw_shape = kwargs["shape"]
        x = inputs[0]
        # Separate shape dim inputs (SSA name references) from the main tensor
        dim_inputs = list(inputs[1:])
        resolved = []
        for s in raw_shape:
            if isinstance(s, str) and dim_inputs:
                resolved.append(int(dim_inputs.pop(0).item()))
            elif isinstance(s, int):
                resolved.append(s)
            else:
                resolved.append(1)
        # Resolve -1 dimensions
        if -1 in resolved:
            neg_idx = resolved.index(-1)
            total_elements = x.numel()
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
            total_elements = x.numel()
            product = 1
            for s in resolved:
                product *= s
            if product != total_elements:
                remaining = 1
                for s in resolved[1:]:
                    remaining *= s
                if remaining > 0 and total_elements % remaining == 0:
                    resolved[0] = total_elements // remaining
        return x.reshape(*resolved)

    def _op_identity(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        dtype_str = kwargs.get("dtype", None)
        if dtype_str is not None:
            dtype_map = {
                "torch.bfloat16": torch.bfloat16,
                "torch.float16": torch.float16,
                "torch.float32": torch.float32,
                "torch.float64": torch.float64,
                "torch.int32": torch.int32,
                "torch.int64": torch.int64,
                "torch.bool": torch.bool,
            }
            target_dtype = dtype_map.get(dtype_str)
            if target_dtype is not None and x.dtype != target_dtype:
                x = x.to(target_dtype)
        if len(inputs) > 1 and "type_as" in str(kwargs.get("original", "")):
            x = x.to(inputs[1].dtype)
        return x

    def _op_relu(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return F.relu(inputs[0])

    def _op_embedding(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        # FX graph order: (weight, indices). F.embedding expects (indices, weight).
        indices = inputs[1].to(torch.long)
        return F.embedding(indices, inputs[0])

    def _op_unsqueeze(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", 0)
        ndim = inputs[0].ndim
        if dim > ndim:
            dim = ndim
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
        shape_spec = kwargs.get("shape", (1,))
        if isinstance(shape_spec, (list, tuple)):
            resolved: list[int] = []
            dim_inputs = list(inputs)
            for s in shape_spec:
                if isinstance(s, str) and dim_inputs:
                    resolved.append(int(dim_inputs.pop(0).item()))
                elif isinstance(s, int):
                    resolved.append(s)
                else:
                    resolved.append(1)
            if resolved:
                return torch.ones(resolved, dtype=torch.float32)
        if inputs:
            return torch.ones_like(inputs[0])
        shape = tuple(shape_spec) if isinstance(shape_spec, (list, tuple)) else (1,)
        return torch.ones(shape, dtype=torch.float32)

    def _op_full_like(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        shape_spec = kwargs.get("shape", (1,))
        value = kwargs.get("fill_value", 0)
        # Shape may contain SSA name strings (dynamic dims) resolved via inputs
        if isinstance(shape_spec, (list, tuple)):
            resolved: list[int] = []
            dim_inputs = list(inputs)
            for s in shape_spec:
                if isinstance(s, str) and dim_inputs:
                    resolved.append(int(dim_inputs.pop(0).item()))
                elif isinstance(s, int):
                    resolved.append(s)
                else:
                    resolved.append(1)
            if resolved:
                return torch.full(resolved, value, dtype=torch.float32)
        # Fallback: use inputs as template (original full_like behavior)
        is_simple = not any(
            isinstance(s, str) for s in (
                shape_spec if isinstance(shape_spec, (list, tuple)) else ()
            )
        )
        if inputs and is_simple:
            return torch.full_like(inputs[0], value)
        shape = tuple(shape_spec) if isinstance(shape_spec, (list, tuple)) else (1,)
        return torch.full(shape, value, dtype=torch.float32)

    def _op_arange(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        start = kwargs.get("start", 0)
        if len(inputs) >= 2:
            start = int(inputs[0].item()) if inputs[0].numel() == 1 else start
            end = int(inputs[1].item()) if inputs[1].numel() == 1 else 1
        elif inputs:
            end = int(inputs[0].item()) if inputs[0].numel() == 1 else 1
        else:
            end = kwargs.get("end", 1)
        return torch.arange(start, end, dtype=inputs[0].dtype if inputs else torch.float32)

    def _op_neg(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return -inputs[0]

    def _op_cos(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.cos(inputs[0])

    def _op_sin(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.sin(inputs[0])

    def _op_rsqrt(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.rsqrt(inputs[0])

    def _op_mean(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", -1)
        keepdim = kwargs.get("keepdim", True)
        return torch.mean(inputs[0], dim=dim, keepdim=keepdim)

    def _op_pow(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if len(inputs) >= 2:
            return torch.pow(inputs[0], inputs[1])
        exponent = kwargs.get("exponent", 2)
        return torch.pow(inputs[0], exponent)

    def _op_triu(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        diagonal = kwargs.get("diagonal", 0)
        while x.ndim < 2:
            x = x.unsqueeze(0)
        return torch.triu(x, diagonal=diagonal)

    def _op_sym_size(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", 0)
        size_val = inputs[0].shape[dim]
        return torch.tensor(size_val, dtype=torch.int64)

    # ── 比较运算 ────────────────────────────────────────────

    def _op_gt(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.gt(inputs[0], inputs[1])

    def _op_lt(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.lt(inputs[0], inputs[1])

    # ── 掩码和归约 ──────────────────────────────────────────

    def _op_masked_fill(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        tensor, mask = inputs[0], inputs[1]
        value = inputs[2]
        return tensor.masked_fill(mask.to(torch.bool), value)

    def _op_cumsum(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", -1)
        return torch.cumsum(inputs[0], dim=dim)

    def _op_expand(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        sizes: list[int] = []
        for t in inputs[1:]:
            val = int(t.item()) if t.numel() == 1 else 1
            sizes.append(val if val >= 0 else x.shape[len(sizes)])
        # Pad with -1 on the left if fewer sizes than input dims
        while len(sizes) < x.dim():
            sizes.insert(0, -1)
        return x.expand(*sizes)

    # ── 融合算子 ────────────────────────────────────────────

    def _op_fused_rms_norm_matmul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        rms_weight = inputs[1]
        mat_weight = inputs[-1]
        eps = kwargs.get("eps", 1e-5)
        orig_dtype = x.dtype
        # Upcast to float32 for numerical stability (matching HF implementation)
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

    def _op_fused_silu_mul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        gate = inputs[0]
        up = inputs[1]
        return F.silu(gate) * up

    # ── Qwen3.5 / extended ops ───────────────────────────────

    def _op_logical_and(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.logical_and(inputs[0], inputs[1])

    def _op_eq(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.eq(inputs[0], inputs[1])

    def _op_le(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.le(inputs[0], inputs[1])

    def _op_ne(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.ne(inputs[0], inputs[1])

    def _op_sigmoid(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.sigmoid(inputs[0])

    def _op_softplus(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return F.softplus(inputs[0])

    def _op_exp(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.exp(inputs[0])

    def _op_sum(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", None)
        keepdim = kwargs.get("keepdim", False)
        if dim is not None:
            return torch.sum(inputs[0], dim=dim, keepdim=keepdim)
        return torch.sum(inputs[0])

    def _op_tril(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        diagonal = kwargs.get("diagonal", 0)
        return torch.tril(inputs[0], diagonal=diagonal)

    def _op_chunk(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        chunks = kwargs.get("chunks", 2)
        dim = kwargs.get("dim", 0)
        # Return stacked chunks as a single tensor
        return torch.stack(torch.chunk(x, chunks, dim=dim), dim=dim)

    def _op_split(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        split_sizes = kwargs.get("split_sizes", None)
        dim = kwargs.get("dim", 0)
        if split_sizes is not None:
            # Adjust split_sizes proportionally if input dim doesn't match
            total_expected = sum(split_sizes)
            actual_dim = x.shape[dim]
            if actual_dim != total_expected:
                ratio = actual_dim / total_expected
                adjusted = [max(1, int(s * ratio)) for s in split_sizes]
                # Ensure split sizes sum to actual_dim
                diff = actual_dim - sum(adjusted)
                if diff != 0:
                    adjusted[-1] += diff
                return torch.cat(torch.split(x, adjusted, dim=dim), dim=dim)
            return torch.cat(torch.split(x, list(split_sizes), dim=dim), dim=dim)
        return x

    def _op_conv1d(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x, weight = inputs[0], inputs[1]
        bias = inputs[2] if len(inputs) > 2 else None
        if x.dtype != weight.dtype:
            weight = weight.to(x.dtype)
        if bias is not None:
            bias = bias.to(x.dtype)
        stride = kwargs.get("stride", 1)
        padding = kwargs.get("padding", 0)
        dilation = kwargs.get("dilation", 1)
        groups = kwargs.get("groups", 1)
        return F.conv1d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    def _op_diff(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", -1)
        import warnings
        n = kwargs.get("n", 1)
        dim = kwargs.get("dim", -1)
        prepend = kwargs.get("prepend", None)
        append = kwargs.get("append", None)
        if prepend is None and len(inputs) > 1:
            prepend = inputs[1].to(inputs[0].dtype)
        if append is None and len(inputs) > 2:
            append = inputs[2].to(inputs[0].dtype)
        try:
            return torch.diff(inputs[0], n=n, dim=dim, prepend=prepend, append=append)
        except TypeError:
            warnings.warn(
                f"torch.diff(prepend=, append=) not supported — "
                f"falling back to torch.diff(dim={dim})",
                stacklevel=2,
            )
            return torch.diff(inputs[0], dim=dim)

    def _op_pad(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        pad = kwargs.get("pad", [0, 0])
        mode = kwargs.get("mode", "constant")
        value = kwargs.get("value", 0.0)
        return F.pad(x, list(pad), mode=mode, value=value)

    def _op_index(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        indices = [inp.to(torch.int64) for inp in inputs[1:]]
        if len(indices) == 1:
            return inputs[0][indices[0]]
        return inputs[0][tuple(indices)]

    def _op_eye(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        n = int(inputs[0].item()) if inputs else kwargs.get("n", 1)
        m = kwargs.get("m", n)
        dtype = inputs[0].dtype if inputs else torch.float32
        return torch.eye(n, m, dtype=dtype)

    def _op_zeros(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if inputs:
            shape = [int(t.item()) for t in inputs]
        else:
            shape_raw = kwargs.get("shape", ())
            shape = [int(s) for s in shape_raw if s is not None]
        dtype = kwargs.get("dtype", torch.float32)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.replace("torch.", ""))
        return torch.zeros(shape, dtype=dtype)

    def _op_zeros_like(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.zeros_like(inputs[0])

    def _op_new_ones(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        shape = [int(t.item()) for t in inputs[1:]]
        return inputs[0].new_ones(shape)

    def _op_select(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        dim = kwargs.get("dim", 0)
        index = kwargs.get("index", 0)
        return x.select(dim, index)

    # ── 公开接口 ────────────────────────────────────────────

    def execute(self, op_name: str, inputs: list[Any], **kwargs: Any) -> torch.Tensor:
        spec = _op_registry.get(op_name)
        if spec is not None:
            n_in = len(inputs)
            if n_in < spec.min_inputs:
                raise ValueError(
                    f"Op '{op_name}' expects at least {spec.min_inputs} input(s), "
                    f"got {n_in}."
                )
            if spec.max_inputs is not None and n_in > spec.max_inputs:
                raise ValueError(
                    f"Op '{op_name}' expects at most {spec.max_inputs} input(s), "
                    f"got {n_in}."
                )

        dispatch = {
            "matmul": self._op_matmul,
            "linear": self._op_linear,
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
            "neg": self._op_neg,
            "cos": self._op_cos,
            "sin": self._op_sin,
            "rsqrt": self._op_rsqrt,
            "mean": self._op_mean,
            "pow": self._op_pow,
            "triu": self._op_triu,
            "sym_size": self._op_sym_size,
            "gt": self._op_gt,
            "lt": self._op_lt,
            "masked_fill": self._op_masked_fill,
            "cumsum": self._op_cumsum,
            "expand": self._op_expand,
            "fused_rms_norm_matmul": self._op_fused_rms_norm_matmul,
            "fused_silu_mul": self._op_fused_silu_mul,
            # Qwen3.5 extended ops
            "logical_and": self._op_logical_and,
            "eq": self._op_eq,
            "le": self._op_le,
            "ne": self._op_ne,
            "sigmoid": self._op_sigmoid,
            "softplus": self._op_softplus,
            "exp": self._op_exp,
            "sum": self._op_sum,
            "tril": self._op_tril,
            "chunk": self._op_chunk,
            "split": self._op_split,
            "conv1d": self._op_conv1d,
            "diff": self._op_diff,
            "pad": self._op_pad,
            "index": self._op_index,
            "eye": self._op_eye,
            "zeros": self._op_zeros,
            "zeros_like": self._op_zeros_like,
            "new_ones": self._op_new_ones,
            "select": self._op_select,
            "type_as": self._op_type_as,
            "copy_": self._op_copy_,
        }
        if op_name not in dispatch:
            raise ValueError(f"Unknown op: {op_name}. Available: {sorted(dispatch.keys())}")
        return dispatch[op_name](inputs, **kwargs)
