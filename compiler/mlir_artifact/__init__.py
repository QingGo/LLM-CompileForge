"""MLIR artifact serialization — save/load compiled models in MLIR format.

Replaces the JSON-based model.ir with standard-compliant MLIR text.

This module re-exports all public API from the split sub-modules for backward
compatibility. All logic lives in:

  - ``compiler.mlir_artifact.serialize``  (Section A: serialization)
  - ``compiler.mlir_artifact.load``       (Section B: weight loading)
  - ``compiler.mlir_artifact.binary``     (Section C: binary generation)
  - ``compiler.mlir_artifact.ir``         (Section D: MLIR IR building)
  - ``compiler.mlir_artifact.parse``      (Section E: text parsing)

Output structure:
  compiled/<model>/
    model.mlir       — MLIR text (primary format)
    weights.pth      — PyTorch state dict
    metadata.json    — compilation metadata
"""

from __future__ import annotations

import logging

from compiler.mlir_artifact.binary import (  # noqa: F401
    _build_constants_binary,
    _build_name_mapping,
    _emit_compute_graph_section,
    _emit_string,
    _init_dtype_codes,
    _parse_type_shape,
)
from compiler.mlir_artifact.ir import (  # noqa: F401
    _build_mlir_attrs,
    _build_mlir_function,
    _build_return_op,
    _build_ssa_map,
    _emit_compute_op,
    _emit_weight_op,
    _infer_result_types,
    _map_op_results,
    _python_to_attr_ir,
    _resolve_operands,
    _resolve_output_values,
    _type_str_to_ir_type,
    _update_function_type,
    mlir_module_to_ir_module,
)
from compiler.mlir_artifact.load import (  # noqa: F401
    _candidate_names,
    _guess_func,
    _load_weights_legacy,
    _load_weights_via_bin,
    _load_weights_via_mmap,
    _load_weights_via_sharded,
    load_mlir_artifact,
)
from compiler.mlir_artifact.parse import (  # noqa: F401
    _parse_attr_value,
    _parse_attrs,
    _parse_mlir_op,
    _parse_mlir_text,
    _parse_type_list,
    _split_attrs,
    _split_comma,
    _split_qualified,
)
from compiler.mlir_artifact.serialize import (  # noqa: F401
    _format_attr,
    _unranked_tensor_type,
    mlir_module_to_text,
    save_mlir_module_artifact,
)
from compiler.mlir_dialect.sf.mlir_op_types import (  # noqa: F401
    MlirFunction,
    MlirModule,
    MlirOp,
)

_log = logging.getLogger(__name__)
