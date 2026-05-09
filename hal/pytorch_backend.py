from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from hal.interface import Buffer, Device, OpExecutor

# ── Unified op registry (single source of truth) ──────────────


@dataclass
class _OpSpec:
    name: str
    min_inputs: int = 0
    max_inputs: int | None = None


_OP_DISPATCH: dict[str, tuple[str, _OpSpec]] = {}


def _register_handler(
    name: str,
    handler: str,
    min_inputs: int = 0,
    max_inputs: int | None = None,
) -> None:
    _OP_DISPATCH[name] = (handler, _OpSpec(name=name, min_inputs=min_inputs, max_inputs=max_inputs))


def _build_dispatch_table(cls: type) -> dict[str, tuple[Callable[..., Any], _OpSpec]]:
    """Build dispatch table from _OP_DISPATCH, resolving method names on *cls*."""
    table: dict[str, tuple[Callable[..., Any], _OpSpec]] = {}
    for name, (method_name, spec) in _OP_DISPATCH.items():
        handler = getattr(cls, method_name)
        table[name] = (handler, spec)
    return table


class PyTorchDevice(Device):
    def __init__(self, device_type: str = "cpu") -> None:
        self._device_type = device_type

    @property
    def device_type(self) -> str:
        return self._device_type

    def synchronize(self) -> None:
        if self._device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()


class PyTorchBuffer(Buffer):
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
    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self._dispatch: dict[str, tuple[Callable[..., Any], _OpSpec]] = _build_dispatch_table(type(self))

    # ── Op implementations ────────────────────────────────────

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

    def _op_scaled_dot_product_attention(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        query, key, value = inputs[0], inputs[1], inputs[2]
        attn_mask = kwargs.get("attn_mask", None)
        if attn_mask is None and len(inputs) > 3:
            attn_mask = inputs[3]
        if attn_mask is not None and attn_mask.dim() > 4:
            attn_mask = attn_mask.squeeze(2)
        if attn_mask is not None and attn_mask.dim() == 4:
            bsize = query.shape[0]
            if attn_mask.shape[1] > 1 and attn_mask.shape[1] == bsize:
                attn_mask = attn_mask[:, :1, :, :]
        dropout_p = kwargs.get("dropout_p", 0.0)
        is_causal = kwargs.get("is_causal", False)
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
        dim_inputs = list(inputs[1:])
        resolved = []
        for s in raw_shape:
            if isinstance(s, str) and dim_inputs:
                resolved.append(int(dim_inputs.pop(0).item()))
            elif isinstance(s, int):
                resolved.append(s)
            else:
                resolved.append(1)
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
        indices = inputs[1].to(torch.long)
        return F.embedding(indices, inputs[0])

    def _op_unsqueeze(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", 0)
        ndim = inputs[0].ndim
        if dim > ndim:
            dim = ndim
        return inputs[0].unsqueeze(dim)

    def _op_sub(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
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

    def _op_gt(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.gt(inputs[0], inputs[1])

    def _op_lt(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.lt(inputs[0], inputs[1])

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
        while len(sizes) < x.dim():
            sizes.insert(0, -1)
        return x.expand(*sizes)

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

    def _op_fused_silu_mul(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        gate = inputs[0]
        up = inputs[1]
        return F.silu(gate) * up

    def _op_fused_attention_output(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        q_t, k_t, v_t = inputs[0], inputs[1], inputs[2]
        remaining = inputs[3:]
        mask: torch.Tensor | None = None
        o_weight: torch.Tensor | None = None
        o_bias: torch.Tensor | None = None
        for t in remaining:
            if t.dim() == 4:
                mask = t
            elif t.dim() == 2:
                o_weight = t
            elif t.dim() == 1:
                o_bias = t
            elif o_weight is None and t.dim() <= 2:
                o_weight = t
            else:
                o_bias = t
        if o_weight is None:
            raise ValueError("fused_attention_output: missing output projection weight")
        sdpa_kwargs: dict[str, Any] = {}
        for key in ("scale", "is_causal", "dropout_p"):
            if key in kwargs:
                sdpa_kwargs[key] = kwargs[key]
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask, **sdpa_kwargs)
        t_attrs = kwargs.get("fuse_transpose_attrs", {})
        if isinstance(t_attrs, str):
            import ast
            try:
                t_attrs = ast.literal_eval(t_attrs)
            except (ValueError, SyntaxError):
                t_attrs = {}
        dim0 = t_attrs.get("dim0", 1)
        dim1 = t_attrs.get("dim1", 2)
        if isinstance(dim0, int) and isinstance(dim1, int):
            attn_out = attn_out.transpose(dim0, dim1)
        b, s = attn_out.shape[0], attn_out.shape[1]
        hidden = attn_out.shape[2] * attn_out.shape[3]
        attn_out = attn_out.reshape(b, s, hidden)
        return F.linear(attn_out, o_weight, o_bias)

    def _op_fused_attention_block(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        tensors_1d: list[torch.Tensor] = []
        tensors_2d: list[torch.Tensor] = []
        tensors_3d: list[torch.Tensor] = []
        tensors_4d: list[torch.Tensor] = []
        for t in inputs:
            d = t.dim()
            if d == 1:
                tensors_1d.append(t)
            elif d == 2:
                tensors_2d.append(t)
            elif d == 4:
                tensors_4d.append(t)
            else:
                tensors_3d.append(t)
        rms_input = tensors_3d[0] if tensors_3d else inputs[0]
        rms_weight = tensors_1d[0] if tensors_1d else inputs[1]
        qkv_weight = tensors_2d[0] if len(tensors_2d) >= 1 else inputs[2]
        o_weight = tensors_2d[1] if len(tensors_2d) >= 2 else tensors_2d[0]
        o_bias = tensors_1d[1] if len(tensors_1d) >= 2 else None
        rms_normed = rms_input * torch.rsqrt(rms_input.pow(2).mean(-1, keepdim=True) + 1e-6)
        rms_normed = rms_normed * rms_weight
        orig_dtype = rms_normed.dtype
        qkv_out = torch.nn.functional.linear(rms_normed.to(qkv_weight.dtype), qkv_weight)
        qkv_out = qkv_out.to(orig_dtype)
        hidden = qkv_out.shape[-1] // 3
        q = qkv_out[..., :hidden]
        k = qkv_out[..., hidden:2 * hidden]
        v = qkv_out[..., 2 * hidden:]
        if qkv_weight.dim() == 2:
            total_hidden = qkv_weight.shape[0]
            n_heads = 4
            head_dim = total_hidden // (3 * n_heads)
        else:
            n_heads = 4
            head_dim = hidden // n_heads
        bsz, seq = q.shape[0], q.shape[1]
        q = q.reshape(bsz, seq, n_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(bsz, seq, n_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(bsz, seq, n_heads, head_dim).permute(0, 2, 1, 3)
        mask_4d = tensors_4d[0] if tensors_4d else None
        sdpa_kwargs: dict[str, Any] = {}
        for key in ("scale", "is_causal", "dropout_p"):
            if key in kwargs:
                sdpa_kwargs[key] = kwargs[key]
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d, **sdpa_kwargs)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(bsz, seq, hidden * n_heads)
        return torch.nn.functional.linear(attn_out, o_weight, o_bias)

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
        return torch.stack(torch.chunk(x, chunks, dim=dim), dim=dim)

    def _op_split(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        split_sizes = kwargs.get("split_sizes", None)
        dim = kwargs.get("dim", 0)
        if split_sizes is not None:
            total_expected = sum(split_sizes)
            actual_dim = x.shape[dim]
            if actual_dim != total_expected:
                ratio = actual_dim / total_expected
                adjusted = [max(1, int(s * ratio)) for s in split_sizes]
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

    # ── Public interface ──────────────────────────────────────

    def execute(self, op_name: str, inputs: list[Any], **kwargs: Any) -> torch.Tensor:
        entry = self._dispatch.get(op_name)
        if entry is None:
            raise ValueError(f"Unknown op: {op_name}. Available: {sorted(self._dispatch.keys())}")
        handler, spec = entry
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
        return handler(self, inputs, **kwargs)  # type: ignore[no-any-return]


# ── Module-level op registration ──────────────────────────────

_register_handler("matmul", "_op_matmul", 2, 2)
_register_handler("linear", "_op_linear", 2, 3)
_register_handler("add", "_op_add", 2, 2)
_register_handler("mul", "_op_mul", 2, 2)
_register_handler("gelu", "_op_gelu", 1, 1)
_register_handler("silu", "_op_silu", 1, 1)
_register_handler("softmax", "_op_softmax", 1, 2)
_register_handler("layer_norm", "_op_layer_norm", 1, 3)
_register_handler("rms_norm", "_op_rms_norm", 1, 2)
_register_handler("permute", "_op_permute", 1, 1)
_register_handler("transpose", "_op_transpose", 1, 1)
_register_handler("scaled_dot_product_attention", "_op_scaled_dot_product_attention", 3, 4)
_register_handler("cat", "_op_cat", 0, None)
_register_handler("slice", "_op_slice", 1, 4)
_register_handler("view", "_op_view", 1, None)
_register_handler("identity", "_op_identity", 1, 2)
_register_handler("relu", "_op_relu", 1, 1)
_register_handler("embedding", "_op_embedding", 2, 2)
_register_handler("unsqueeze", "_op_unsqueeze", 1, 1)
_register_handler("sub", "_op_sub", 2, 2)
_register_handler("max", "_op_max", 2, 2)
_register_handler("ones_like", "_op_ones_like", 0, None)
_register_handler("full_like", "_op_full_like", 0, None)
_register_handler("arange", "_op_arange", 0, 2)
_register_handler("neg", "_op_neg", 1, 1)
_register_handler("cos", "_op_cos", 1, 1)
_register_handler("sin", "_op_sin", 1, 1)
_register_handler("rsqrt", "_op_rsqrt", 1, 1)
_register_handler("mean", "_op_mean", 1, 1)
_register_handler("pow", "_op_pow", 1, 2)
_register_handler("triu", "_op_triu", 1, 1)
_register_handler("sym_size", "_op_sym_size", 1, 1)
_register_handler("gt", "_op_gt", 2, 2)
_register_handler("lt", "_op_lt", 2, 2)
_register_handler("masked_fill", "_op_masked_fill", 3, 3)
_register_handler("cumsum", "_op_cumsum", 1, 1)
_register_handler("expand", "_op_expand", 1, None)
_register_handler("fused_rms_norm_matmul", "_op_fused_rms_norm_matmul", 2, 3)
_register_handler("fused_silu_mul", "_op_fused_silu_mul", 2, 2)
_register_handler("logical_and", "_op_logical_and", 2, 2)
_register_handler("eq", "_op_eq", 2, 2)
_register_handler("le", "_op_le", 2, 2)
_register_handler("ne", "_op_ne", 2, 2)
_register_handler("sigmoid", "_op_sigmoid", 1, 1)
_register_handler("softplus", "_op_softplus", 1, 1)
_register_handler("exp", "_op_exp", 1, 1)
_register_handler("sum", "_op_sum", 1, 1)
_register_handler("tril", "_op_tril", 1, 1)
_register_handler("chunk", "_op_chunk", 1, 1)
_register_handler("split", "_op_split", 1, 1)
_register_handler("conv1d", "_op_conv1d", 2, 3)
_register_handler("diff", "_op_diff", 1, 3)
_register_handler("pad", "_op_pad", 1, 1)
_register_handler("index", "_op_index", 1, None)
_register_handler("eye", "_op_eye", 0, 1)
_register_handler("zeros", "_op_zeros", 0, 10)
_register_handler("zeros_like", "_op_zeros_like", 1, 1)
_register_handler("new_ones", "_op_new_ones", 1, None)
_register_handler("select", "_op_select", 1, 1)
_register_handler("type_as", "_op_type_as", 2, 2)
_register_handler("copy_", "_op_copy_", 2, 2)
_register_handler("fused_attention_output", "_op_fused_attention_output", 3, None)
_register_handler("fused_attention_block", "_op_fused_attention_block", 3, None)
