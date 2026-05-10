"""Protocol definitions for type-safe component boundaries.

These protocols replace `Any` annotations at module interfaces,
enabling static type checking without creating circular imports.
"""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    """Minimal tokenizer interface used by LLMEngine."""

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class BlockManagerLike(Protocol):
    """Minimal block manager interface consumed by RadixCache.

    Both the Python BlockManager and Rust PyBlockManager satisfy this
    protocol via duck typing, so RadixCache works with either backend.
    """

    block_size: int

    def increment_ref_count(self, block_id: int) -> None: ...
    def free_block(self, block_id: int) -> None: ...
