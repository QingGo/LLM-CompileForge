"""DEPRECATED — bridges to C++ sf-lower-to-linalg pass.

All lowering is now handled by the C++ sf-lower-to-linalg pass.
This module provides backward-compatible function signatures that
delegate to compiler.pipeline._apply_sf_to_linalg.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


def sf_to_linalg_pass(mlir_text: str) -> str:
    """Deprecated. Delegates to C++ pass via _apply_sf_to_linalg."""
    from compiler.pipeline import _apply_sf_to_linalg
    return _apply_sf_to_linalg(mlir_text)


def sf_to_linalg_pass_on_module(ir_mod: Any) -> str:
    """Deprecated. Delegates to C++ pass via _apply_sf_to_linalg."""
    import mlir.ir as _ir

    from compiler.pipeline import _apply_sf_to_linalg
    if isinstance(ir_mod, _ir.Module):
        mlir_text = str(ir_mod)
        return _apply_sf_to_linalg(mlir_text)

    from compiler.mlir_artifact import MlirModule
    if isinstance(ir_mod, MlirModule):
        return _apply_sf_to_linalg("", orig_mlir_mod=ir_mod)

    raise TypeError(f"Expected ir.Module or MlirModule, got {type(ir_mod)}")


_LOWER_TABLE: dict = {}
