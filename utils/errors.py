"""Structured error types for LLM-ServeForge.

Provides a type hierarchy so callers can programmatically distinguish
error conditions (OOM vs scheduling error vs compilation failure)
without parsing exception message strings.

Usage:
    from utils.errors import OutOfMemoryError, ScheduleError

    try:
        blocks = block_manager.allocate(rid, n_tokens)
    except OutOfMemoryError as e:
        # e.needed, e.free, e.total are available
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


class ScheduleError(ServeForgeError):
    """Scheduler invariant violation.

    Raised when the scheduler detects an internal inconsistency, such as
    a request with no blocks allocated or a state machine transition error.
    """


class CompilationError(ServeForgeError):
    """MLIR compilation failure.

    Carries the pass name and the wrapped exception for debugging.
    """

    def __init__(self, pass_name: str, detail: str = "") -> None:
        self.pass_name = pass_name
        super().__init__(
            f"CompilationError in '{pass_name}': {detail}" if detail
            else f"CompilationError in '{pass_name}'"
        )


class KVWriteError(ServeForgeError):
    """KV cache position mismatch.

    Raised when a position index cannot be mapped to any request's
    block table, indicating a block allocation or scheduling bug.
    """

    def __init__(self, position: int, block_tables: dict[str, list[int]]) -> None:
        self.position = position
        super().__init__(
            f"KVWriteError: position {position} not found in any block table "
            f"({len(block_tables)} tables)"
        )


class ConfigurationError(ServeForgeError):
    """Invalid engine or compiler configuration.

    Raised at startup time for unrecoverable configuration mistakes
    (e.g., num_layers=0 while KV cache is enabled).
    """
