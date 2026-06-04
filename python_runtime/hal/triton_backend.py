"""Triton backend for the Hardware Abstraction Layer.

Provides a TritonKernelRegistry that maps HAL op names to custom Triton
kernels, and a TritonBackend that dispatches ops through the registry
with a fallback to PyTorch for unsupported ops.

Reference: design-phase1.md §3.1; design-phase2.md §2.3
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from python_runtime.hal.interface import OpExecutor


class TritonKernelRegistry:
    """Global registry mapping HAL op names → Triton kernel functions.

    Usage:
        @TritonKernelRegistry.register("scaled_dot_product_attention")
        def my_sdpa_kernel(q, k, v, **kwargs):
            ...
    """

    _kernels: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, op_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            cls._kernels[op_name] = func
            return func
        return decorator

    @classmethod
    def get(cls, op_name: str) -> Callable[..., Any] | None:
        return cls._kernels.get(op_name)

    @classmethod
    def list_registered(cls) -> list[str]:
        return sorted(cls._kernels.keys())


class TritonBackend(OpExecutor):
    """HAL backend that dispatches ops through Triton kernels.

    Ops registered in TritonKernelRegistry are handled by the custom
    Triton kernel.  All other ops fall through to the PyTorch backend.

    Args:
        fallback: The PyTorch backend used for ops without Triton kernels.
    """

    def __init__(self, fallback: OpExecutor) -> None:
        self._fallback = fallback

    def execute(self, op_name: str, inputs: list[Any], **kwargs: Any) -> torch.Tensor:
        kernel = TritonKernelRegistry.get(op_name)
        if kernel is not None:
            return kernel(*inputs, **kwargs)  # type: ignore[no-any-return]
        return self._fallback.execute(op_name, inputs, **kwargs)
