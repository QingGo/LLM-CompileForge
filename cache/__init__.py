"""Prefix Cache module — Radix Tree KV cache sharing."""

from typing import Any

__all__ = ["RadixCache"]

_LAZY_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import cache.radix_cache as _rc
        value = getattr(_rc, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'cache' has no attribute '{name}'")
