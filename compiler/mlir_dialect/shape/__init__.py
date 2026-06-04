# ruff: noqa: F401 — re-exports for backward compatibility
"""shape inference sub-package — type inference for sf dialect ops."""

from compiler.mlir_dialect.shape.shape_inference import (  # type: ignore[attr-defined]
    _INFERENCE_TABLE,
    infer_output_shape,
    infer_output_type,
)
from compiler.mlir_dialect.shape.shape_inference_pure import (
    _PURE_TABLE,
)
from compiler.mlir_dialect.shape.shape_inference_utils import (
    _broadcast_shapes,
    _broadcast_types,
    _elt_from_str,
    _elt_type_str,
    _get_elt_type_map,
    _infer_ir_via_pure,
    _make_ranked_type,
    _ranked_shape,
)
