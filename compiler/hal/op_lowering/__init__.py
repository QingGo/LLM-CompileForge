"""Op lowering — dispatch ``sf.*`` ops to HAL IR entries."""

from compiler.hal.op_lowering.core import (
    infer_dtype_from_type,
    lower_op,
    parse_sf_op_name,
    shape_from_type,
    strip_mlir_quotes,
)

__all__ = [
    "infer_dtype_from_type",
    "lower_op",
    "parse_sf_op_name",
    "shape_from_type",
    "strip_mlir_quotes",
]
