"""MLIR Graph Compiler — Phase 1 MVP."""

from typing import Any

__all__ = [
    "CompilationPipeline",
    "IrFunction",
    "IrModule",
    "IrOp",
    "IrType",
    "compile_module",
    "default_pipeline",
    "ir_module_to_mlir",
]

_LAZY_ATTRS = frozenset(
    {
        "CompilationPipeline",
        "IrFunction",
        "IrModule",
        "IrOp",
        "IrType",
        "compile_module",
        "default_pipeline",
        "ir_module_to_mlir",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import compiler.ir as _ir
        import compiler.mlir_emitter as _mlir_emitter
        import compiler.pipeline as _pipeline

        _globals = {
            "IrType": _ir.IrType,
            "IrOp": _ir.IrOp,
            "IrFunction": _ir.IrFunction,
            "IrModule": _ir.IrModule,
            "CompilationPipeline": _pipeline.CompilationPipeline,
            "compile_module": _pipeline.compile_module,
            "default_pipeline": _pipeline.default_pipeline,
            "ir_module_to_mlir": _mlir_emitter.ir_module_to_mlir,
        }
        if name in _globals:
            value = _globals[name]
            globals()[name] = value
            return value
    raise AttributeError(f"module 'compiler' has no attribute '{name}'")
