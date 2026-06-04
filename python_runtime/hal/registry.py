"""Backend registry — maps backend names to factory functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from python_runtime.hal.interface import OpExecutor

_registry: dict[str, Callable[..., OpExecutor]] = {}


def register(name: str, factory: Callable[..., OpExecutor]) -> None:
    """Register a backend factory under the given name.

    Raises:
        ValueError: If the name is already registered.
    """
    if name in _registry:
        raise ValueError(f"Backend '{name}' already registered")
    _registry[name] = factory


def create(name: str, **kwargs: Any) -> OpExecutor:
    """Instantiate a new backend by calling its registered factory.

    Raises:
        KeyError: If no backend is registered under the given name.
    """
    factory = _registry.get(name)
    if factory is None:
        raise KeyError(f"Backend '{name}' not found")
    return factory(**kwargs)


def get(name: str) -> Callable[..., OpExecutor]:
    """Return the registered factory for the given backend name.

    Raises:
        KeyError: If no backend is registered under the given name.
    """
    if name not in _registry:
        raise KeyError(f"Backend '{name}' not found")
    return _registry[name]


def list_backends() -> list[str]:
    """Return sorted list of registered backend names."""
    return sorted(_registry.keys())
