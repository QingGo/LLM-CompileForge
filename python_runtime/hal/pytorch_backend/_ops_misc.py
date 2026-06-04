from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

_log = logging.getLogger("hal.pytorch.misc")


class _MiscOps:
    def _op_embedding(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        indices = inputs[1].to(torch.long)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug("embedding: weight=%s indices=%s", inputs[0].shape, indices.shape)
        return F.embedding(indices, inputs[0])

    def _op_type_as(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return inputs[0].to(inputs[1].dtype)

    def _op_copy_(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        if len(inputs) >= 2:
            dst, src = inputs[0], inputs[1]
            return dst.copy_(src)
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
            for i in range(len(resolved) - 1, -1, -1):
                if resolved[i] == -1 and dim_inputs:
                    resolved[i] = int(dim_inputs.pop().item())
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
            for i in range(len(resolved) - 1, -1, -1):
                if resolved[i] == -1 and dim_inputs:
                    resolved[i] = int(dim_inputs.pop().item())
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
        return torch.arange(start, end, dtype=torch.int64)

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

    def _op_logical_and(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.logical_and(inputs[0], inputs[1])

    def _op_eq(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.eq(inputs[0], inputs[1])

    def _op_le(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.le(inputs[0], inputs[1])

    def _op_ne(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.ne(inputs[0], inputs[1])

    def _op_sum(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", None)
        keepdim = kwargs.get("keepdim", False)
        if dim is not None:
            return torch.sum(inputs[0], dim=dim, keepdim=keepdim)
        return torch.sum(inputs[0])

    def _op_tril(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        diagonal = kwargs.get("diagonal", 0)
        return torch.tril(inputs[0], diagonal=diagonal)

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

    def _op_div(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return inputs[0] / inputs[1]

    def _op_tanh(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.tanh(inputs[0])

    def _op_sqrt(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return torch.sqrt(inputs[0])

    def _op_clamp_min(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        min_val = kwargs.get("min", 0.0)
        if len(inputs) > 1:
            t = inputs[1]
            if isinstance(t, torch.Tensor) and t.numel() == 1:
                min_val = float(t.item())
            else:
                min_val = float(t)
        return torch.clamp(inputs[0], min=float(min_val))

    def _op_einsum(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        equation = kwargs.get("equation", "")
        if not equation:
            return inputs[0]
        return torch.einsum(equation, *inputs)

    def _op_stack(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", 0)
        if isinstance(dim, torch.Tensor):
            dim = int(dim.item()) if dim.numel() == 1 else 0
        return torch.stack(inputs, dim=dim)

    def _op_linalg_norm(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        result: torch.Tensor = torch.linalg.vector_norm(inputs[0], ord=2, dim=-1, keepdim=True)
        return result

    def _op_var(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        dim = kwargs.get("dim", -1)
        keepdim = kwargs.get("keepdim", True)
        return inputs[0].float().var(dim=dim, keepdim=keepdim, unbiased=False)

    def _op_view_as(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return inputs[0].view_as(inputs[1])

    def _op_expand_as(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        return inputs[0].expand_as(inputs[1])


from python_runtime.hal.pytorch_backend import _register_handler  # noqa: E402

_register_handler("embedding", "_op_embedding", 2, 2)
_register_handler("type_as", "_op_type_as", 2, 2)
_register_handler("copy_", "_op_copy_", 2, 2)
_register_handler("ones_like", "_op_ones_like", 0, None)
_register_handler("full_like", "_op_full_like", 0, None)
_register_handler("arange", "_op_arange", 0, 2)
_register_handler("cos", "_op_cos", 1, 1)
_register_handler("sin", "_op_sin", 1, 1)
_register_handler("rsqrt", "_op_rsqrt", 1, 1)
_register_handler("mean", "_op_mean", 1, 1)
_register_handler("triu", "_op_triu", 1, 1)
_register_handler("sym_size", "_op_sym_size", 1, 1)
_register_handler("gt", "_op_gt", 2, 2)
_register_handler("lt", "_op_lt", 2, 2)
_register_handler("masked_fill", "_op_masked_fill", 3, 3)
_register_handler("cumsum", "_op_cumsum", 1, 1)
_register_handler("logical_and", "_op_logical_and", 2, 2)
_register_handler("eq", "_op_eq", 2, 2)
_register_handler("le", "_op_le", 2, 2)
_register_handler("ne", "_op_ne", 2, 2)
_register_handler("sum", "_op_sum", 1, 1)
_register_handler("tril", "_op_tril", 1, 1)
_register_handler("diff", "_op_diff", 1, 3)
_register_handler("index", "_op_index", 1, None)
_register_handler("eye", "_op_eye", 0, 1)
_register_handler("zeros", "_op_zeros", 0, 10)
_register_handler("zeros_like", "_op_zeros_like", 1, 1)
_register_handler("new_ones", "_op_new_ones", 1, None)
_register_handler("div", "_op_div", 2, 2)
_register_handler("tanh", "_op_tanh", 1, 1)
_register_handler("sqrt", "_op_sqrt", 1, 1)
_register_handler("clamp_min", "_op_clamp_min", 1, 2)
_register_handler("einsum", "_op_einsum", 1, None)
_register_handler("stack", "_op_stack", 1, None)
_register_handler("linalg_norm", "_op_linalg_norm", 1, 3)
_register_handler("var", "_op_var", 1, 1)
_register_handler("view_as", "_op_view_as", 2, 2)
_register_handler("expand_as", "_op_expand_as", 2, 2)
