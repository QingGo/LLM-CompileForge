"""HAL IR layer — normalization, lowering, and hal operation definitions."""

from compiler.mlir_dialect.hal_ir.sf_normalize import normalize_sf_mlir

__all__ = [
    "normalize_sf_mlir",
]
