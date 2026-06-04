"""MLIR-level utilities using official mlir Python bindings.

These functions require the official MLIR Python bindings to be
available (e.g. via mlir_binding/ or pip-installed mlir package).

Usage:
    from compiler.passes import mlir_count_ops, mlir_run_cse
"""

from __future__ import annotations

from typing import Any


def mlir_count_ops(mlir_module: Any, ctx: Any) -> dict[str, int]:
    import mlir.ir as ir

    stats: dict[str, int] = {}

    def _count(op: ir.Operation) -> None:
        name = str(op.name)
        dialect = name.split(".", 1)[0] if "." in name else name
        stats[dialect] = stats.get(dialect, 0) + 1
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    _count(child)

    with ctx:
        for region in mlir_module.operation.regions:
            for block in region.blocks:
                for op in block.operations:
                    _count(op)
    return stats


def mlir_run_cse(mlir_module: Any) -> Any:
    import mlir.passmanager as pm

    ctx = mlir_module.operation.context
    with ctx:
        p = pm.PassManager.parse("builtin.module(cse)", ctx)
        p.run(mlir_module.operation)
    return mlir_module


def mlir_run_canonicalize(mlir_module: Any) -> Any:
    import mlir.passmanager as pm

    ctx = mlir_module.operation.context
    with ctx:
        p = pm.PassManager.parse("builtin.module(canonicalize,cse)", ctx)
        p.run(mlir_module.operation)
    return mlir_module


def mlir_verify_structure(mlir_module: Any, ctx: Any) -> list[str]:
    issues: list[str] = []
    with ctx:
        for region in mlir_module.operation.regions:
            for block in region.blocks:
                func_count = 0
                for op in block.operations:
                    name = str(op.name)
                    # "func" in name catches func.func (LLVM 20.x) or ops with
                    # child regions (anonymous functions in LLVM 22.x).
                    is_func = "func" in name or bool(op.regions)
                    if is_func:
                        func_count += 1
                        if not op.regions:
                            issues.append(f"{name}: missing body region")
                if func_count == 0:
                    issues.append("module: no functions found")
    return issues


def mlir_count_ops_in_module(mlir_text: str) -> dict[str, int]:
    import mlir.ir as ir

    ctx = ir.Context()
    with ctx:
        module = ir.Module.parse(mlir_text, ctx)
        return mlir_count_ops(module, ctx)
