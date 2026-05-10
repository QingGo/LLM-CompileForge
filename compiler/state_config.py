"""Model state configuration — declares internal states per layer.

Defines what state tensors each model architecture needs for
incremental decode, enabling the monkey-patch export to expose
them as graph I/O.

Design: StateSpec (per-state metadata) + ModelStateConfig (per-layer
listing of states), with factory presets for known architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StateSpec:
    """Describes one internal model state tensor.

    Attributes:
        name: Logical name, e.g. "conv_state", "recurrent_state".
        shape_hint: Per-token shape tuple for allocation, e.g. (6144, 3).
        storage: "fixed" (fixed-size, overwrite per step) or "paged" (grows).
        dtype: Element type string.
    """

    name: str
    shape_hint: tuple[int, ...]
    storage: str = "fixed"
    dtype: str = "float32"


@dataclass
class ModelStateConfig:
    """Per-layer state declaration for a model architecture.

    Attributes:
        per_layer: layer_idx → list of StateSpecs for that layer.
        input_name: Name prefix for the state input in the graph.
    """

    per_layer: dict[int, list[StateSpec]] = field(default_factory=dict)

    @property
    def state_names(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for specs in self.per_layer.values():
            for s in specs:
                key = f"{s.name}_layer"
                if key not in seen:
                    seen.add(key)
                    result.append(s.name)
        return result

    def has_layer_state(self, layer_idx: int) -> bool:
        return layer_idx in self.per_layer

    def states_for(self, layer_idx: int) -> list[StateSpec]:
        return self.per_layer.get(layer_idx, [])

    def get_paged_layer_indices(self) -> list[int]:
        return [
            idx for idx, specs in self.per_layer.items()
            if any(s.storage == "paged" for s in specs)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_layer": {
                str(k): [
                    {"name": s.name, "shape": list(s.shape_hint),
                     "storage": s.storage, "dtype": s.dtype}
                    for s in v
                ]
                for k, v in self.per_layer.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelStateConfig:
        return cls(
            per_layer={
                int(k): [
                    StateSpec(
                        name=s["name"],
                        shape_hint=tuple(s["shape"]),
                        storage=s.get("storage", "fixed"),
                        dtype=s.get("dtype", "float32"),
                    )
                    for s in v
                ]
                for k, v in d["per_layer"].items()
            }
        )

    # ── Factory presets ─────────────────────────────────

    @classmethod
    def for_qwen3_5(cls, config: Any) -> ModelStateConfig:
        per_layer: dict[int, list[StateSpec]] = {}

        layer_types = getattr(config, "layer_types", [])
        num_layers = len(layer_types)

        for i in range(num_layers):
            lt = layer_types[i]
            if lt == "linear_attention":
                conv_dim = (
                    getattr(config, "linear_key_head_dim", 128) *
                    getattr(config, "linear_num_key_heads", 16) * 2 +
                    getattr(config, "linear_value_head_dim", 128) *
                    getattr(config, "linear_num_value_heads", 16)
                )
                kernel = getattr(config, "linear_conv_kernel_dim", 4)
                per_layer[i] = [
                    StateSpec("conv_state", (conv_dim, kernel - 1), "fixed", "bfloat16"),
                    StateSpec(
                        "recurrent_state",
                        (
                            getattr(config, "linear_num_value_heads", 16),
                            getattr(config, "linear_key_head_dim", 128),
                            getattr(config, "linear_value_head_dim", 128),
                        ),
                        "fixed",
                        "bfloat16",
                    ),
                ]
            elif lt == "full_attention":
                num_heads = getattr(config, "num_attention_heads", 8)
                getattr(config, "num_key_value_heads", 2)
                head_dim = getattr(config, "head_dim", 256)
                per_layer[i] = [
                    StateSpec("k_cache", (num_heads, head_dim), "paged", "bfloat16"),
                    StateSpec("v_cache", (num_heads, head_dim), "paged", "bfloat16"),
                ]

        return cls(per_layer=per_layer)
