"""PyTorch backend — Hardware Abstraction Layer for PyTorch tensors.

Provides the default HAL backend using PyTorch operations on
CPU and CUDA devices.  All 60+ op handlers are split across
sub-modules by category for maintainability.

Sub-modules:
  _ops_math.py        — add, mul, sub, neg, pow, max, matmul, linear
  _ops_activation.py  — gelu, silu, relu, sigmoid, softplus, exp
  _ops_norm.py        — layer_norm, rms_norm, softmax
  _ops_shape.py       — view, permute, transpose, cat, split, slice
  _ops_attention.py   — scaled_dot_product_attention, fused_attention_*
  _ops_misc.py        — cos, sin, mean, triu, ones_like, zeros, etc.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from python_runtime.hal.interface import Buffer, Device, OpExecutor

# ── Op Registry ────────────────────────────────────────


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
    table: dict[str, tuple[Callable[..., Any], _OpSpec]] = {}
    for name, (method_name, spec) in _OP_DISPATCH.items():
        handler = getattr(cls, method_name)
        table[name] = (handler, spec)
    return table


# ── Device / Buffer ─────────────────────────────────────


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


# ── Mixin imports (register handlers into _OP_DISPATCH) ──

from python_runtime.hal.pytorch_backend._ops_activation import _ActivationOps  # noqa: E402
from python_runtime.hal.pytorch_backend._ops_attention import _AttentionOps  # noqa: E402
from python_runtime.hal.pytorch_backend._ops_math import _MathOps  # noqa: E402
from python_runtime.hal.pytorch_backend._ops_misc import _MiscOps  # noqa: E402
from python_runtime.hal.pytorch_backend._ops_norm import _NormOps  # noqa: E402
from python_runtime.hal.pytorch_backend._ops_shape import _ShapeOps  # noqa: E402

# ── PyTorchBackend ──────────────────────────────────────


class PyTorchBackend(_MathOps, _ActivationOps, _NormOps, _ShapeOps, _AttentionOps, _MiscOps, OpExecutor):
    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self._dispatch = _build_dispatch_table(type(self))

    def execute(self, op_name: str, inputs: list[Any], **kwargs: Any) -> torch.Tensor:
        if op_name not in self._dispatch:
            raise ValueError(f"Unknown op: {op_name}")
        handler, spec = self._dispatch[op_name]
        n = len(inputs)
        if n < spec.min_inputs:
            raise ValueError(f"Op '{op_name}' requires at least {spec.min_inputs} inputs, got {n}")
        if spec.max_inputs is not None and n > spec.max_inputs:
            raise ValueError(f"Op '{op_name}' accepts at most {spec.max_inputs} inputs, got {n}")
        return handler(self, inputs, **kwargs)  # type: ignore[no-any-return]

    def _resolve(self, value: Any, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            t = value.to(device=self._device, dtype=dtype)
        elif isinstance(value, (int, float)):
            t = torch.tensor(value, device=self._device, dtype=dtype)
        else:
            t = torch.tensor(value, device=self._device, dtype=dtype)
        return t
