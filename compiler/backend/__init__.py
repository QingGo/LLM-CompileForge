"""Compiler backend — IR verification and dylib compilation utilities."""

from compiler.backend.dylib import (
    _check_sf_dialect_freshness,
    _compile_blob_to_o,
    _sfa_relink_dylib,
)
from compiler.backend.verify import _save_failure_context, _verify_lowered_ir

__all__ = [
    "_verify_lowered_ir",
    "_save_failure_context",
    "_check_sf_dialect_freshness",
    "_compile_blob_to_o",
    "_sfa_relink_dylib",
]
