"""MLIR Graph Compiler."""

from compiler._lazy_imports import lazy_imports

lazy_imports(__name__, globals(), {
    "compile_mlir": ("compiler.pipeline", "compile_mlir"),
    "load_artifact": ("compiler.serialize", "load_artifact"),
})
