"""Per-op Rust function templates for the EmitRust backend.

Each constant is a raw Rust source string for one HAL CPU op implementation.
The ``OP_IMPLS`` dict maps op-name → template string for dispatch by the
``emit_rust()`` orchestrator.

Templates are organized by category in ``rust_templates/`` submodules.
"""

from __future__ import annotations

from compiler.hal.rust_templates.binary_ops import (
    OP_COMPARE,
    OP_ELEMENT_WISE,
    OP_MATMUL,
)
from compiler.hal.rust_templates.memory_ops import (
    OP_FILL,
    OP_GATHER,
    OP_RESHAPE,
    OP_SLICE,
    OP_TRANSPOSE,
    OP_UNSQUEEZE,
)
from compiler.hal.rust_templates.reduce_ops import (
    OP_REDUCE,
    OP_SHAPE_OF,
    OP_SOFTMAX,
)

# ── Op dispatch: name -> template ─────────────────────────────────────────

OP_IMPLS: dict[str, str] = {
    "matmul": OP_MATMUL,
    "element_wise": OP_ELEMENT_WISE,
    "softmax": OP_SOFTMAX,
    "reshape": OP_RESHAPE,
    "transpose": OP_TRANSPOSE,
    "reduce": OP_REDUCE,
    "gather": OP_GATHER,
    "fill": OP_FILL,
    "shape_of": OP_SHAPE_OF,
    "slice": OP_SLICE,
    "unsqueeze": OP_UNSQUEEZE,
    "compare": OP_COMPARE,
    "cache_read": "",
    "cache_write": "",
    # Future: "rms_norm", "layer_norm", "concat", "scan"
}
