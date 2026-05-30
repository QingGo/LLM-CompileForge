from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

_log = logging.getLogger("hal.pytorch.shape")


class _ShapeOps:
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
        # Use dim_inputs for dynamic-dim sentinels (values < -1).
        # During FX→MLIR lowering, each dynamic dim (sym_size node) is
        # replaced by sentinel -(dyn_pos+2) to distinguish it from static
        # -1 entries.  Static -1 entries remain for inference.
        for i in range(len(resolved)):
            if resolved[i] < -1 and dim_inputs:
                resolved[i] = int(dim_inputs.pop(0).item())
        # All -1 with no dim_inputs → flatten (equivalent to reshape(-1))
        if all(s == -1 for s in resolved) and not dim_inputs:
            resolved = [-1]
        elif -1 in resolved:
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
        result = x.reshape(*resolved)
        return result

    def _op_permute(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dims = kwargs["dims"]
        return inputs[0].permute(*dims)

    def _op_transpose(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim0 = kwargs.get("dim0", 0)
        dim1 = kwargs.get("dim1", 1)
        return inputs[0].transpose(dim0, dim1)

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

    def _op_unsqueeze(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", 0)
        ndim = inputs[0].ndim
        if dim > ndim:
            dim = ndim
        return inputs[0].unsqueeze(dim)

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

    def _op_chunk(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        chunks = kwargs.get("chunks", 2)
        dim = kwargs.get("dim", 0)
        return torch.stack(torch.chunk(x, chunks, dim=dim), dim=dim)

    def _op_expand(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        sizes: list[int] = []
        for t in inputs[1:]:
            val = int(t.item()) if t.numel() == 1 else 1
            sizes.append(val if val >= 0 else x.shape[len(sizes)])
        while len(sizes) < x.dim():
            sizes.insert(0, -1)
        return x.expand(*sizes)

    def _op_select(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        dim = kwargs.get("dim", 0)
        index = kwargs.get("index", 0)
        return x.select(dim, index)

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

    def _op_pad(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        x = inputs[0]
        pad = kwargs.get("pad", [0, 0])
        mode = kwargs.get("mode", "constant")
        value = kwargs.get("value", 0.0)
        return F.pad(x, list(pad), mode=mode, value=value)


from hal.pytorch_backend import _register_handler  # noqa: E402

_register_handler("permute", "_op_permute", 1, 1)
_register_handler("transpose", "_op_transpose", 1, 1)
_register_handler("cat", "_op_cat", 0, None)
_register_handler("slice", "_op_slice", 1, 4)
_register_handler("view", "_op_view", 1, None)
_register_handler("identity", "_op_identity", 1, 2)
_register_handler("unsqueeze", "_op_unsqueeze", 1, 1)
_register_handler("expand", "_op_expand", 1, None)
_register_handler("chunk", "_op_chunk", 1, 1)
_register_handler("split", "_op_split", 1, 1)
_register_handler("conv1d", "_op_conv1d", 2, 3)
_register_handler("pad", "_op_pad", 1, 1)
_register_handler("select", "_op_select", 1, 1)
