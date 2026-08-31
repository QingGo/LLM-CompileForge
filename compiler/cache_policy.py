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
    func_index: int | None = None
    output_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "slab_id": self.slab_id,
            "op_name": self.op_name,
            "direction": self.direction,
            "source": self.source,
            "layer": self.layer,
        }
        if self.func_index is not None:
            d["func_index"] = self.func_index
        if self.output_index is not None:
            d["output_index"] = self.output_index
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _InterceptSpec:
        return cls(
            slab_id=d["slab_id"],
            op_name=d["op_name"],
            direction=d["direction"],
            source=d["source"],
            layer=d.get("layer", "sequential"),
            func_index=d.get("func_index"),
            output_index=d.get("output_index"),
        )


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read a config field from either a mapping or an attribute object."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


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
    def for_config(cls, config: Any, block_size: int = 16) -> CachePolicy:
        """Derive a standard SDPA K/V policy from a model config.

        This is the only supported config->policy path for decoder models.
        Per-model layer/kv-head/head-dim triples must not be hard-coded in
        callers.  Multimodal configs (Qwen3.5 image-text-to-text) are
        resolved through their ``text_config``.

        Raises:
            ValueError: Missing or inconsistent attention dimensions.
            NotImplementedError: Mixed linear/full attention (E11).  Such
                models need a recurrent-state slab in addition to SDPA K/V;
                ``for_llama`` would silently describe the wrong contract.
        """
        cfg = config
        if isinstance(cfg, dict):
            nested = cfg.get("text_config")
        else:
            nested = getattr(cfg, "text_config", None)
        if nested is not None:
            cfg = nested

        num_layers = _config_value(cfg, "num_hidden_layers")
        num_attention_heads = _config_value(cfg, "num_attention_heads")
        num_kv_heads = _config_value(cfg, "num_key_value_heads") or num_attention_heads
        head_dim = _config_value(cfg, "head_dim")
        hidden_size = _config_value(cfg, "hidden_size")

        missing = [
            name
            for name, value in (
                ("num_hidden_layers", num_layers),
                ("num_attention_heads", num_attention_heads),
                ("num_key_value_heads", num_kv_heads),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"Config is missing attention fields required for CachePolicy: {missing}"
            )

        if head_dim is None:
            if hidden_size is None:
                raise ValueError("Config is missing both head_dim and hidden_size")
            if int(hidden_size) % int(num_attention_heads) != 0:
                raise ValueError(
                    f"hidden_size={hidden_size} is not divisible by "
                    f"num_attention_heads={num_attention_heads}"
                )
            head_dim = int(hidden_size) // int(num_attention_heads)

        layer_types = _config_value(cfg, "layer_types")
        if layer_types is not None:
            if isinstance(layer_types, str):
                layer_types = [layer_types]
            non_full = [str(t) for t in layer_types if str(t) != "full_attention"]
            if non_full:
                return cls.for_mixed_linear(
                    num_layers=int(num_layers),
                    num_full_layers=sum(1 for t in layer_types if str(t) == "full_attention"),
                    num_linear_layers=sum(1 for t in layer_types if str(t) != "full_attention"),
                    num_kv_heads=int(num_kv_heads),
                    head_dim=int(head_dim),
                    linear_num_key_heads=int(_config_value(cfg, "linear_num_key_heads", 0) or 0),
                    linear_num_value_heads=int(_config_value(cfg, "linear_num_value_heads", 0) or 0),
                    linear_key_head_dim=int(_config_value(cfg, "linear_key_head_dim", 0) or 0),
                    linear_value_head_dim=int(_config_value(cfg, "linear_value_head_dim", 0) or 0),
                    linear_conv_kernel_dim=int(_config_value(cfg, "linear_conv_kernel_dim", 4) or 4),
                    block_size=block_size,
                )
            if len(layer_types) != int(num_layers):
                raise ValueError(
                    f"layer_types length {len(layer_types)} != num_hidden_layers {num_layers}"
                )

        return cls.for_llama(
            num_layers=int(num_layers),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            block_size=block_size,
        )

    @classmethod
    def for_mixed_linear(
        cls,
        num_layers: int,
        num_full_layers: int,
        num_linear_layers: int,
        num_kv_heads: int,
        head_dim: int,
        linear_num_key_heads: int,
        linear_num_value_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int = 4,
        block_size: int = 16,
    ) -> CachePolicy:
        """Cache policy for mixed full/linear attention models (Qwen3.5).

        The full-attention layers use the normal paged K/V slabs.  The
        GatedDeltaNet layers additionally need two fixed-size state slabs:
        - ``recurrent_state``: [batch, heads, key_dim, value_dim] delta state
        - ``conv_state``: [batch, conv_channels, kernel] short-conv state
        """
        conv_channels = (
            linear_num_key_heads * linear_key_head_dim * 2
            + linear_num_value_heads * linear_value_head_dim
        )
        policy = cls(
            slabs=[
                _SlabSpec(
                    slab_id="k",
                    storage="paged",
                    dims={"layers": num_full_layers, "heads": num_kv_heads, "dim": head_dim},
                    layout="BNLD",
                    dtype="float32",
                ),
                _SlabSpec(
                    slab_id="v",
                    storage="paged",
                    dims={"layers": num_full_layers, "heads": num_kv_heads, "dim": head_dim},
                    layout="BNLD",
                    dtype="float32",
                ),
                _SlabSpec(
                    slab_id="recurrent_state",
                    storage="fixed",
                    dims={
                        "layers": num_linear_layers,
                        "heads": linear_num_value_heads,
                        "key_dim": linear_key_head_dim,
                        "value_dim": linear_value_head_dim,
                    },
                    layout="BHKV",
                    dtype="float32",
                ),
                _SlabSpec(
                    slab_id="conv_state",
                    storage="fixed",
                    dims={
                        "layers": num_linear_layers,
                        "channels": conv_channels,
                        "kernel": linear_conv_kernel_dim,
                    },
                    layout="BCK",
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
                _InterceptSpec(
                    slab_id="recurrent_state",
                    op_name="linear_attn",
                    direction="read_write",
                    source="recurrent_state",
                    layer="sequential",
                ),
                _InterceptSpec(
                    slab_id="conv_state",
                    op_name="linear_attn",
                    direction="read_write",
                    source="conv_state",
                    layer="sequential",
                ),
            ],
            block_size=block_size,
        )
        return policy

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


def bind_cache_policy(
    policy: CachePolicy,
    bindings: list[tuple[int, int, str]],
) -> CachePolicy:
    """Expand SDPA intercept specs into per-``(func_index, output_index)`` entries.

    ``bindings`` comes from ``compiler.fx.split._make_multi_functions``:
    one ``(func_index, output_index, slab_id)`` tuple per cache-consumed
    K/V output.  Each template ``scaled_dot_product_attention`` intercept
    (one per slab) is replaced by concrete per-binding entries carrying
    ``func_index``/``output_index`` — the exact keys the Rust runtime
    matches against (``param_indices`` in the proto encoding).

    Intercepts unrelated to SDPA (e.g. RWKV state) are preserved as-is.
    When *bindings* is empty, the policy is returned unchanged.
    """
    if not bindings:
        return policy
    sdpa_templates = {
        i.slab_id: i for i in policy.intercepts if i.op_name == "scaled_dot_product_attention"
    }
    expanded: list[_InterceptSpec] = [
        i for i in policy.intercepts if i.op_name != "scaled_dot_product_attention"
    ]
    for fi, oi, slab_id in bindings:
        template = sdpa_templates.get(slab_id)
        expanded.append(
            _InterceptSpec(
                slab_id=slab_id,
                op_name="scaled_dot_product_attention",
                direction=template.direction if template else "read_write",
                source=template.source if template else "operand[1]",
                layer=template.layer if template else "sequential",
                func_index=fi,
                output_index=oi,
            )
        )
    return CachePolicy(
        slabs=list(policy.slabs),
        intercepts=expanded,
        block_size=policy.block_size,
        max_requests=policy.max_requests,
    )


def serialize_cache_policy(policy: CachePolicy) -> bytes:
    """Serialize a CachePolicy into the ``SfaCachePolicy`` protobuf binary.

    The resulting blob is embedded in the compiled dylib as the
    ``sfa_cache_policy`` / ``sfa_cache_policy_size`` symbol pair (see
    ``compiler/backend/dylib.py::_compile_blob_to_o``).
    """
    from gen.proto.python import sfa_abi_pb2

    proto = sfa_abi_pb2.SfaCachePolicy()  # type: ignore[attr-defined]
    for slab in policy.slabs:
        sp = proto.slabs.add()
        sp.name = slab.slab_id
        sp.slab_type = slab.storage
        sp.layout = slab.layout
        sp.dtype = slab.dtype
        sp.num_blocks = int(slab.dims.get("blocks", 0))
        sp.block_size = policy.block_size
        sp.num_layers = int(slab.dims.get("layers", 0))
        sp.num_heads = int(slab.dims.get("heads", 0))
        sp.head_dim = int(slab.dims.get("dim", 0))
    for intercept in policy.intercepts:
        ip = proto.intercepts.add()
        ip.slab_id = intercept.slab_id
        ip.op_name_pattern = intercept.op_name
        ip.intercept_type = intercept.direction
        ip.source = intercept.source
        ip.layer = intercept.layer
        if intercept.func_index is not None:
            ip.param_indices.append(int(intercept.func_index))
        if intercept.output_index is not None:
            ip.param_indices.append(int(intercept.output_index))
    proto.block_size = policy.block_size
    proto.max_requests = policy.max_requests
    return bytes(proto.SerializeToString())
