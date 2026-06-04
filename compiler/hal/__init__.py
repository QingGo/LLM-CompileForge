"""HAL IR layer — normalization, lowering, and hal operation definitions."""

from compiler.hal.lower_sf_to_hal import (
    lower_sf_to_hal,
    lower_sf_to_hal_file,
)
from compiler.hal.sf_normalize import normalize_sf_mlir

__all__ = [
    "normalize_sf_mlir",
    "lower_sf_to_hal",
    "lower_sf_to_hal_file",
]
