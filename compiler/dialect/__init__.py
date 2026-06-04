# ruff: noqa: F401 — re-exports for backward compatibility
"""sf dialect sub-package — op definitions, builder, types, catalog."""

from compiler.dialect._op_defs import (
    _ATEN_TO_HAL,
    _LIST_ARG_ATTR,
    _OP_DEFS,
    _SCALAR_INT_POSITIONS,
    _SCALAR_KWARG_NAMES,
    _OpDef,
)
from compiler.dialect.builder import SfModule
from compiler.dialect.mlir_op_types import (
    MlirFunction,
    MlirModule,
    MlirOp,
    ssa,
)
from compiler.dialect.op_catalog import build_op_catalog
from compiler.dialect.sf import (
    _ALL_OPS,
    SfOp,
    get_op_class,
)
