"""Rust template constants — organized by op category.

Exports all individual template constants plus the combined OP_IMPLS dict.
"""

from __future__ import annotations

from compiler.mlir_dialect.hal_ir.rust_templates.binary_ops import (
    OP_COMPARE,
    OP_ELEMENT_WISE,
    OP_MATMUL,
)
from compiler.mlir_dialect.hal_ir.rust_templates.boilerplate import (
    BLAS_EXTERN,
    HEADER,
    OP_SHAPE_META,
    STUB_CACHE,
)
from compiler.mlir_dialect.hal_ir.rust_templates.memory_ops import (
    OP_FILL,
    OP_GATHER,
    OP_RESHAPE,
    OP_SLICE,
    OP_TRANSPOSE,
    OP_UNSQUEEZE,
)
from compiler.mlir_dialect.hal_ir.rust_templates.reduce_ops import (
    OP_REDUCE,
    OP_SHAPE_OF,
    OP_SOFTMAX,
)

__all__ = [
    "BLAS_EXTERN",
    "HEADER",
    "OP_COMPARE",
    "OP_ELEMENT_WISE",
    "OP_FILL",
    "OP_GATHER",
    "OP_MATMUL",
    "OP_REDUCE",
    "OP_RESHAPE",
    "OP_SHAPE_META",
    "OP_SHAPE_OF",
    "OP_SLICE",
    "OP_SOFTMAX",
    "STUB_CACHE",
    "OP_TRANSPOSE",
    "OP_UNSQUEEZE",
]
