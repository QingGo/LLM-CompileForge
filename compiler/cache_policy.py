"""Cache policy — declarative KV cache strategy per model.

Defines what gets cached, how it's stored, and how it's indexed.
The policy is serialized into metadata.json at compile time and read
by the executor/engine at runtime.

Design principle: Slab (storage format) × Indexer (addressing method)
are orthogonal axes. Any storage type can pair with any index type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _SlabSpec:
    """Describes one contiguous cache region.

    Attributes:
        slab_id: Logical name, e.g. "k", "v", "latent", "k_rope", "state".
        storage: Storage scheme — "paged" | "flat" | "circular" | "fixed".
        dims: Dimension hints for allocation, e.g. {"layers": 16, "heads": 8, "dim": 64}.
        layout: Tensor dim order string, e.g. "BNLD", "NLD", "WLD", "RLD".
        dtype: Element type string, e.g. "float32", "bfloat16".
    """

    slab_id: str
    storage: str
    dims: dict[str, int]
    layout: str
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slab_id": self.slab_id,
            "storage": self.storage,
            "dims": self.dims,
            "layout": self.layout,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _SlabSpec:
        return cls(
            slab_id=d["slab_id"],
            storage=d["storage"],
            dims={k: int(v) for k, v in d["dims"].items()},
            layout=d["layout"],
            dtype=d.get("dtype", "float32"),
        )


@dataclass
class _InterceptSpec:
    """Describes when and how the executor intercepts an op for cache I/O.

    Attributes:
        slab_id: Which slab to read from / write to.
        op_name: Short op name (no dialect prefix), e.g. "scaled_dot_product_attention".
        direction: "read", "write", or "read_write".
        source: How to extract data from the op.
            - "operand[N]" → use op.operands[N] as the SSA key.
            - "output" → use op.results[0] as the SSA key.
        layer: Layer indexing strategy:
            - "sequential" → increment counter per intercept (auto).
            - A static index string → used directly.
    """

    slab_id: str
    op_name: str
    direction: str
    source: str
    layer: str = "sequential"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slab_id": self.slab_id,
            "op_name": self.op_name,
            "direction": self.direction,
            "source": self.source,
            "layer": self.layer,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _InterceptSpec:
        return cls(
            slab_id=d["slab_id"],
            op_name=d["op_name"],
            direction=d["direction"],
            source=d["source"],
            layer=d.get("layer", "sequential"),
        )


@dataclass
class CachePolicy:
    """Declarative model cache strategy.

    Written into metadata.json at compile time so the runtime reads
    exactly what the compiler intended — no heuristic guessing.

    Attributes:
        slabs: Storage regions to pre-allocate.
        intercepts: Per-op cache I/O instructions.
        block_size: Block size for paged slabs (default 16).
        max_requests: Pool capacity for fixed-size slabs (default 256).
    """

    slabs: list[_SlabSpec] = field(default_factory=list)
    intercepts: list[_InterceptSpec] = field(default_factory=list)
    block_size: int = 16
    max_requests: int = 256

    @property
    def is_empty(self) -> bool:
        return len(self.slabs) == 0

    @classmethod
    def none(cls) -> CachePolicy:
        """A no-cache policy — executor does full recompute every step."""
        return cls()

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> CachePolicy:
        if d is None:
            return cls.none()
        return cls(
            slabs=[_SlabSpec.from_dict(s) for s in d.get("slabs", [])],
            intercepts=[_InterceptSpec.from_dict(i) for i in d.get("intercepts", [])],
            block_size=int(d.get("block_size", 16)),
            max_requests=int(d.get("max_requests", 256)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slabs": [s.to_dict() for s in self.slabs],
            "intercepts": [i.to_dict() for i in self.intercepts],
            "block_size": self.block_size,
            "max_requests": self.max_requests,
        }

    # ── Factory: Llama / standard transformer ──────────────────

    @classmethod
    def for_llama(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
    ) -> CachePolicy:
        return cls(
            slabs=[
                _SlabSpec(
                    slab_id="k",
                    storage="paged",
                    dims={"layers": num_layers, "heads": num_kv_heads, "dim": head_dim},
                    layout="BNLD",
                    dtype="float32",
                ),
                _SlabSpec(
                    slab_id="v",
                    storage="paged",
                    dims={"layers": num_layers, "heads": num_kv_heads, "dim": head_dim},
                    layout="BNLD",
                    dtype="float32",
                ),
            ],
            intercepts=[
                _InterceptSpec(
                    slab_id="k",
                    op_name="scaled_dot_product_attention",
                    direction="read_write",
                    source="operand[1]",
                    layer="sequential",
                ),
                _InterceptSpec(
                    slab_id="v",
                    op_name="scaled_dot_product_attention",
                    direction="read_write",
                    source="operand[2]",
                    layer="sequential",
                ),
            ],
            block_size=block_size,
        )

    # ── Factory: RWKV (fixed state) ───────────────────────────

    @classmethod
    def for_rwkv(
        cls,
        num_layers: int,
        state_dim: int,
        max_requests: int = 256,
    ) -> CachePolicy:
        return cls(
            slabs=[
                _SlabSpec(
                    slab_id="state",
                    storage="fixed",
                    dims={"layers": num_layers, "dim": state_dim},
                    layout="RLD",
                    dtype="float32",
                ),
            ],
            intercepts=[
                _InterceptSpec(
                    slab_id="state",
                    op_name="state_evolve",
                    direction="read_write",
                    source="output",
                    layer="sequential",
                ),
            ],
            max_requests=max_requests,
        )
