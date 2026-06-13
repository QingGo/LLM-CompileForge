"""VirtualDevice — simulated HAL backend for hardware-independent verification.

Hardware-spec-based latency simulation instead of actual computation.
Each operation returns a correctly-shaped random tensor and sleeps for
the predicted execution time.

Multiple VirtualDevice instances with different hardware specs can be combined
to simulate heterogeneous (CPU+GPU+NPU) deployments.

Reference: design-phase3.md §1.5 (Hardware Verification Framework)
"""

from __future__ import annotations

import time
from typing import Any

import torch

from python_runtime.hal.hardware_spec import HardwareSpec


class VirtualDevice:
    """Simulated hardware device.

    Args:
        name: Human-readable device name (e.g. "A100", "NPU-Sim").
        spec: HardwareSpec that defines peak FLOPS, bandwidth, and
              per-operation latency tables for this device.
    """

    def __init__(self, name: str, spec: HardwareSpec) -> None:
        self._name = name
        self._spec = spec
        self._total_time_ns: float = 0.0
        self._op_counts: dict[str, int] = {}
        self._device = torch.device("cpu")
        self._rng = torch.Generator()
        self._rng.manual_seed(42)

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> HardwareSpec:
        return self._spec

    @property
    def total_time_ns(self) -> float:
        return self._total_time_ns

    @property
    def total_time_ms(self) -> float:
        return self._total_time_ns / 1e6

    def execute(self, op_name: str, inputs: list[Any], **kwargs: Any) -> torch.Tensor:
        latency_ns = self._spec.predict_latency(op_name, inputs, **kwargs)
        dtype = self._resolve_dtype(inputs)
        shape = self._infer_output_shape(op_name, inputs, **kwargs)

        if latency_ns > 0:
            time.sleep(latency_ns / 1e9)

        self._total_time_ns += latency_ns
        self._op_counts[op_name] = self._op_counts.get(op_name, 0) + 1

        return torch.randn(shape, generator=self._rng, dtype=dtype)

    def reset_stats(self) -> None:
        self._total_time_ns = 0.0
        self._op_counts.clear()

    def _resolve_dtype(self, inputs: list[Any]) -> torch.dtype:
        for v in inputs:
            if isinstance(v, torch.Tensor):
                return v.dtype if v.dtype.is_floating_point else torch.float32
        return torch.float32

    def _infer_output_shape(self, op_name: str, inputs: list[Any], **kwargs: Any) -> tuple[int, ...]:
        if op_name == "weight":
            wt = kwargs.get("_weight_tensor")
            if isinstance(wt, torch.Tensor):
                return tuple(wt.shape)
            return (1,)
        if op_name == "constant":
            wt = kwargs.get("_const_tensor")
            if isinstance(wt, torch.Tensor):
                return tuple(wt.shape)
            return (1,)
        if op_name == "identity":
            if inputs and isinstance(inputs[0], torch.Tensor):
                return tuple(inputs[0].shape)
            return (1,)
        if op_name in ("embedding", "gather"):
            if len(inputs) >= 2 and isinstance(inputs[1], torch.Tensor):
                w = inputs[0] if isinstance(inputs[0], torch.Tensor) else inputs[1]
                idx = inputs[1] if isinstance(inputs[0], torch.Tensor) else inputs[0]
                embed_dim = w.shape[-1] if w.dim() >= 2 else w.shape[0]
                return tuple(idx.shape) + (embed_dim,)
            return (1, 256)
        if op_name == "scaled_dot_product_attention":
            for v in inputs:
                if isinstance(v, torch.Tensor) and v.dim() >= 3:
                    return tuple(v.shape)
            return (1, 8, 16, 64)
        if op_name == "fused_attention_output":
            for v in inputs:
                if isinstance(v, torch.Tensor) and v.dim() >= 3:
                    return tuple(v.shape[:2]) + (v.shape[-1],)
            return (1, 8, 64)
        if op_name == "fused_rms_norm_matmul":
            for v in inputs:
                if isinstance(v, torch.Tensor) and v.dim() == 2:
                    return (v.shape[0], 256)
            return (1, 256)
        if op_name == "fused_silu_mul":
            if inputs and isinstance(inputs[0], torch.Tensor):
                return tuple(inputs[0].shape)
            return (1,)

        for v in inputs:
            if isinstance(v, torch.Tensor):
                vs = tuple(v.shape)
                if op_name in ("matmul", "linear"):
                    if v.dim() >= 2 and len(inputs) >= 2:
                        b = inputs[1]
                        if isinstance(b, torch.Tensor) and b.dim() >= 2:
                            return vs[:-1] + (b.shape[-1],)
                    return vs
                if op_name in (
                    "add",
                    "sub",
                    "mul",
                    "div",
                    "max",
                    "min",
                    "gt",
                    "lt",
                    "relu",
                    "gelu",
                    "silu",
                    "sigmoid",
                    "softplus",
                    "exp",
                    "neg",
                    "tanh",
                    "sqrt",
                    "rsqrt",
                    "cos",
                    "sin",
                    "clamp_min",
                    "triu",
                    "tril",
                    "type_as",
                    "copy_",
                    "layer_norm",
                    "rms_norm",
                    "softmax",
                    "zeros_like",
                    "masked_fill",
                    "logical_and",
                    "eq",
                    "le",
                    "ne",
                    "ones_like",
                    "full_like",
                    "linalg_norm",
                    "var",
                ):
                    return vs
                if op_name == "permute":
                    dims = kwargs.get("dims", ())
                    if dims:
                        return tuple(vs[d] for d in dims)
                    return tuple(reversed(vs))
                if op_name == "view":
                    s = kwargs.get("shape")
                    if isinstance(s, (list, tuple)):
                        resolved = tuple(int(x) if isinstance(x, int) else -1 for x in s)
                        return _resolve_view_shape(vs, resolved)
                    return vs
                if op_name == "transpose":
                    d0 = kwargs.get("dim0", 0)
                    d1 = kwargs.get("dim1", 1)
                    s = list(vs)
                    s[d0], s[d1] = s[d1], s[d0]
                    return tuple(s)
                if op_name == "slice":
                    return _slice_shape(vs, kwargs)
                if op_name == "unsqueeze":
                    dim = kwargs.get("dim", -1)
                    s = list(vs)
                    if dim < 0:
                        dim = len(s) + 1 + dim
                    s.insert(dim, 1)
                    return tuple(s)
                if op_name == "expand":
                    target = kwargs.get("shape")
                    if isinstance(target, (list, tuple)):
                        resolved_t = tuple(int(x) if isinstance(x, int) else -1 for x in target)
                        return _resolve_view_shape(vs, resolved_t)
                    return vs
                if op_name == "select":
                    s = list(vs)
                    dim = kwargs.get("dim", 0)
                    if 0 <= dim < len(s):
                        s.pop(dim)
                    return tuple(s) if s else (1,)
                if op_name == "split":
                    sizes = kwargs.get("split_sizes", ())
                    if sizes:
                        dim = kwargs.get("dim", 0)
                        s = list(vs)
                        if 0 <= dim < len(s):
                            s[dim] = sizes[0] if isinstance(sizes[0], int) else 1
                        return tuple(s)
                    return vs
                if op_name == "chunk":
                    chunks = kwargs.get("chunks", 1)
                    if chunks and isinstance(chunks, int) and chunks > 0:
                        dim = kwargs.get("dim", 0)
                        s = list(vs)
                        if 0 <= dim < len(s):
                            s[dim] = (s[dim] + chunks - 1) // chunks
                        return tuple(s)
                    return vs
                if op_name in ("mean", "sum"):
                    dim = kwargs.get("dim", -1)
                    keepdim = kwargs.get("keepdim", False)
                    s = list(vs)
                    if isinstance(dim, (list, tuple)):
                        for d in sorted(dim, reverse=True):
                            if 0 <= d < len(s):
                                if keepdim:
                                    s[d] = 1
                                else:
                                    s.pop(d)
                    elif 0 <= dim < len(s):
                        if keepdim:
                            s[dim] = 1
                        else:
                            s.pop(dim)
                    return tuple(s) if s else (1,)
                if op_name == "cat":
                    dim = kwargs.get("dim", 0)
                    items = [inp for inp in inputs if isinstance(inp, torch.Tensor)]
                    if items:
                        total = sum(t.shape[dim] for t in items if dim < t.dim())
                        s = list(items[0].shape)
                        if dim < len(s):
                            s[dim] = total
                        return tuple(s)
                    return (1,)
                if op_name == "pad":
                    pad_vals = kwargs.get("pad", ())
                    s = list(vs)
                    if len(pad_vals) >= 2:
                        s[-1] = s[-1] + pad_vals[-2] + pad_vals[-1]
                    return tuple(s)
                if op_name == "einsum":
                    equation = kwargs.get("equation", "")
                    if isinstance(equation, str) and "->" in equation:
                        out_part = equation.split("->")[1]
                        return tuple(1 for _ in out_part) if out_part else (1,)
                    return (1,)
                if op_name == "stack":
                    dim = kwargs.get("dim", 0)
                    s = list(vs)
                    s.insert(dim, len([inp for inp in inputs if isinstance(inp, torch.Tensor)]))
                    return tuple(s)
                if op_name == "index":
                    return vs
                if op_name == "diff":
                    n = kwargs.get("n", 1)
                    s = list(vs)
                    dim = kwargs.get("dim", -1)
                    if 0 <= dim < len(s):
                        s[dim] -= n
                    return tuple(s)
                if op_name == "conv1d":
                    return vs
                if op_name in ("eye", "arange", "zeros"):
                    shape = kwargs.get("shape")
                    if isinstance(shape, (list, tuple)):
                        return tuple(int(s) for s in shape if s is not None)
                    return (1,)
                return vs
        return (1,)


def _resolve_view_shape(original: tuple[int, ...], target: tuple[int, ...]) -> tuple[int, ...]:
    resolved: list[int] = []
    total = 1
    for d in original:
        total *= d
    known_prod = 1
    for s in target:
        if isinstance(s, int) and s != -1:
            known_prod *= s
    for s in target:
        if isinstance(s, int) and s == -1:
            resolved.append(total // max(known_prod, 1))
        elif isinstance(s, int):
            resolved.append(s)
        else:
            resolved.append(1)
    return tuple(resolved)


def _slice_shape(original: tuple[int, ...], kwargs: dict[str, Any]) -> tuple[int, ...]:
    d = kwargs.get("dim", 0)
    start = kwargs.get("start", 0)
    end_val = kwargs.get("end", 1)
    step = kwargs.get("step", 1)
    s = list(original)
    if 0 <= d < len(s):
        size = max((end_val - start + step - 1) // step if step > 0 else 0, 0)
        s[d] = size if isinstance(size, int) else 1
    return tuple(s)
