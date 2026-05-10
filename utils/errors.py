"""Structured error types for LLM-ServeForge.

Provides a type hierarchy so callers can programmatically distinguish
error conditions (OOM vs scheduling error vs compilation failure)
without parsing exception message strings.

Usage:
    from utils.errors import OutOfMemoryError

    try:
        blocks = block_manager.allocate(rid, n_tokens)
    except OutOfMemoryError as e:
        cache.evict(e.needed)
"""

from __future__ import annotations


class ServeForgeError(Exception):
    """Base exception for all LLM-ServeForge errors."""


class OutOfMemoryError(ServeForgeError):
    """Block pool exhausted.

    Attributes:
        needed: Blocks required for the allocation.
        free:   Blocks currently free in the pool.
        total:  Total blocks in the pool.
    """

    def __init__(self, needed: int, free: int, total: int) -> None:
        self.needed = needed
        self.free = free
        self.total = total
        super().__init__(
            f"OutOfMemory: need {needed} blocks but only {free} free "
            f"(total pool: {total})"
        )
