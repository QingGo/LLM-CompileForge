"""HAL backend registration and discovery."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hal.interface import OpExecutor

_backends: dict[str, Callable[..., OpExecutor]] = {}


def register(name: str, factory: Callable[..., OpExecutor]) -> None:
    """Register a backend factory under a unique name."""
    if name in _backends:
        raise ValueError(f"Backend '{name}' is already registered")
    _backends[name] = factory


def get(name: str) -> Callable[..., OpExecutor]:
    """Retrieve a registered backend factory by name."""
    if name not in _backends:
        available = sorted(_backends.keys())
        raise KeyError(f"Backend '{name}' not found. Available: {available}")
    return _backends[name]


def list_backends() -> list[str]:
    """Return the names of all registered backends."""
    return sorted(_backends.keys())


def create(name: str, **kwargs: Any) -> OpExecutor:
    """Instantiate a backend by name with the given constructor arguments."""
    factory = get(name)
    return factory(**kwargs)
