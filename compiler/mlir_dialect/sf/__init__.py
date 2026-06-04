# ruff: noqa: F401 — re-exports for backward compatibility
"""sf dialect sub-package — op definitions, builder, types, catalog."""

from compiler.mlir_dialect.sf._op_defs import (
    _ATEN_TO_HAL,
    _LIST_ARG_ATTR,
    _OP_DEFS,
    _SCALAR_INT_POSITIONS,
    _SCALAR_KWARG_NAMES,
    _OpDef,
)
from compiler.mlir_dialect.sf.builder import SfModule
from compiler.mlir_dialect.sf.mlir_op_types import (
    MlirFunction,
    MlirModule,
    MlirOp,
    ssa,
)
from compiler.mlir_dialect.sf.op_catalog import build_op_catalog
from compiler.mlir_dialect.sf.sf import (
    _ALL_OPS,
    SfOp,
    get_op_class,
)
