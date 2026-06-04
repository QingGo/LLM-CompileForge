# Minimal init for mlir_sf._mlir_libs — delegates to 'mlir' (LLVM build).
# Must import mlir.ir FIRST so that libMLIRPythonCAPI symbols are in the
# flat namespace before the nanobind extension .so is loaded.
import os as _os

_this_dir = _os.path.dirname(__file__)

# Load MLIR CAPI dylib into process namespace before any extension .so is loaded.
# The _sfDialectsNanobind extension uses -undefined dynamic_lookup and requires
# libMLIRPythonCAPI symbols to be available at import time.
import mlir.ir  # noqa: F401


def get_lib_dirs():
    return [_this_dir]
def get_include_dirs():
    return []
try:
    import mlir._mlir_libs as _m
    get_dialect_registry = _m.get_dialect_registry
    get_load_on_create_dialects = _m.get_load_on_create_dialects
    append_load_on_create_dialect = _m.append_load_on_create_dialect
except ImportError:
    def _lazy(name):
        def f(*a, **kw):
            import mlir._mlir_libs as _m
            return getattr(_m, name)(*a, **kw)
        return f
    get_dialect_registry = _lazy("get_dialect_registry")
    get_load_on_create_dialects = _lazy("get_load_on_create_dialects")
    append_load_on_create_dialect = _lazy("append_load_on_create_dialect")
