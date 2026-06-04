import mlir.ir as _ir

from .._mlir_libs._sfDialectsNanobind.sf import register_dialects as _register_dialects
from ._sf_ops_gen import *


def register_dialects(ctx: _ir.Context, load: bool = True):
    """Register and optionally load the sf dialect into the given MLIR context."""
    _register_dialects(ctx._CAPIPtr, load)
