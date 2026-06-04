"""FX Graph → MlirModule conversion pipeline.

Public API:
    fx_graph_to_mlir  — convert an ExportedProgram to MlirModule
"""

from compiler.fx.converter import fx_graph_to_mlir

__all__ = ["fx_graph_to_mlir"]
