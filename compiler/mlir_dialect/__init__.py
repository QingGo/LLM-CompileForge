"""MLIR sf dialect — formal dialect definition with shape inference.

Replaces the ad-hoc string-based sf operations with proper MLIR operations
that carry full tensor types.  This enables:
  - Shape-aware optimization (MLIR canonicalize, CSE)
  - Lowering to standard dialects (linalg, arith, math)
  - AOT compilation (bufferization, code generation)

Usage::

    from compiler.mlir_dialect.builder import SfModule
    from compiler.mlir_dialect.shape_inference import infer_output_type

    module = SfModule("main", input_types=[...])
    result = module.add_op("add", [module.inputs[0], module.inputs[1]])
    mlir_text = str(module)
"""

from compiler.mlir_dialect.builder import SfModule
from compiler.mlir_dialect.sf import get_op_class
from compiler.mlir_dialect.shape_inference import infer_output_shape, infer_output_type  # type: ignore[attr-defined]

__all__ = [
    "SfModule",
    "get_op_class",
    "infer_output_shape",
    "infer_output_type",
]
