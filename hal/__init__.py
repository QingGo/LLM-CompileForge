"""Hardware Abstraction Layer — Phase 1 MVP."""

from typing import Any

from hal.interface import Buffer, Device, OpExecutor
from hal.registry import create, get, list_backends, register

__all__ = [
    "Buffer",
    "Device",
    "OpExecutor",
    "PyTorchBackend",
    "PyTorchBuffer",
    "PyTorchDevice",
    "create",
    "get",
    "list_backends",
    "register",
]

_LAZY_BACKEND_ATTRS = frozenset({"PyTorchBackend", "PyTorchBuffer", "PyTorchDevice"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_BACKEND_ATTRS:
        import hal.pytorch_backend as _backend

        value = getattr(_backend, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'hal' has no attribute '{name}'")
