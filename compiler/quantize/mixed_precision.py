"""Mixed-precision per-layer strategy configuration.

Defines the default precision recommendations for different layer types
in a Transformer model.  Users can override these via a config dict or
a configuration file, and the MLIR quantization pass reads this config
to insert Q/DQ nodes accordingly.

Reference: design-phase2.md §2.1.4, §2.1.5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

VALID_PRECISIONS: frozenset[str] = frozenset(
    {
        "fp16",
        "fp32",
        "w8a8",
        "w4a16",
        "int8",
        "fp8",
    }
)


@dataclass
class MixedPrecisionConfig:
    """Per-layer precision strategy for mixed-precision quantization.

    Provides default recommendations that balance accuracy and compression:
      - Attention Q/K/V projections: W8A8 (activation-sensitive).
      - Attention Output projection: W4A16 (output-tolerant).
      - FFN gate/up (first layer): W8A8 (SwiGLU architecture, moderate activation).
      - FFN down (second layer): W4A16 (larger activation, quantization error tolerant).
      - Embedding / LM Head: FP16 (output precision-critical).
      - KV Cache: FP8 (halves memory with negligible quality loss).

    Attributes:
        strategy: dict mapping layer name patterns to precision strings.
    """

    DEFAULT_STRATEGY: ClassVar[dict[str, str]] = {
        "q_proj": "w8a8",
        "k_proj": "w8a8",
        "v_proj": "w8a8",
        "o_proj": "w4a16",
        "gate_proj": "w8a8",
        "up_proj": "w8a8",
        "down_proj": "w4a16",
        "embed_tokens": "fp16",
        "lm_head": "fp16",
        "kv_cache": "fp8",
    }

    strategy: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategy:
            self.strategy = dict(self.DEFAULT_STRATEGY)

    @classmethod
    def from_dict(cls, overrides: dict[str, str] | None = None) -> MixedPrecisionConfig:
        """Create a config from a dict, merging with defaults.

        Args:
            overrides: Key-value precision overrides on top of the defaults.

        Returns:
            A new MixedPrecisionConfig instance.
        """
        strategy = dict(cls.DEFAULT_STRATEGY)
        if overrides:
            strategy.update(overrides)
        return cls(strategy=strategy)

    def get_precision(self, layer_name: str) -> str:
        """Get the recommended precision for a given layer name.

        Performs suffix matching: if the layer name ends with any key
        in the strategy dict, return the corresponding precision.
        Falls back to 'fp16' if no match found.

        Args:
            layer_name: Layer name, e.g. "model.layers.0.self_attn.q_proj".

        Returns:
            Precision string: 'fp16', 'w8a8', 'w4a16', or 'fp8'.
        """
        for pattern, precision in self.strategy.items():
            if layer_name.endswith(pattern):
                return precision
        return "fp16"

    def validate(self) -> bool:
        """Check that all precision values in the strategy are valid.

        Returns:
            True if all values are recognized precision strings.
        """
        for precision in self.strategy.values():
            if precision not in VALID_PRECISIONS:
                return False
        return True

    def to_dict(self) -> dict[str, str]:
        """Export the strategy as a plain dict."""
        return dict(self.strategy)
