"""MLIR Graph Compiler."""

from typing import Any

__all__ = [
    "compile_mlir",
    "load_artifact",
]

_LAZY_ATTRS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import compiler.pipeline as _pipeline
        import compiler.serialize as _serialize

        _globals = {
            "compile_mlir": _pipeline.compile_mlir,
            "load_artifact": _serialize.load_artifact,
        }
        if name in _globals:
            value = _globals[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module 'compiler' has no attribute '{name}'")
