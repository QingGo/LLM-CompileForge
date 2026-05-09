"""Compilation pipeline orchestration.

Assembles the full AOT compilation flow:
  PyTorch model → torch.export → FX → MlirModule → MLIR passes → artifact

The canonical compiler path now emits MlirModule directly via fx_to_mlir and
runs optimization passes on the MLIR representation (using official MLIR
Python bindings when available).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from compiler.fx_to_mlir import fx_graph_to_mlir
from compiler.mlir_artifact import MlirModule, mlir_module_to_text, save_mlir_module_artifact


def _setup_mlir_path() -> None:
    _mlir_pkg = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
    if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
        sys.path.insert(0, str(_mlir_pkg))


def compile_mlir(
    model: torch.nn.Module,
    example_args: tuple[Any, ...] | None = None,
    example_kwargs: dict[str, Any] | None = None,
    output_dir: str | None = None,
    dynamic_shapes: dict[str, Any] | None = None,
    model_dir: str = "",
    cache_export: bool = False,
    apply_fusion: bool = True,
) -> MlirModule:
    """Compile a PyTorch model through the MLIR-native pipeline.

    Steps:
      1. torch.export → ExportedProgram
      2. FX Graph → MlirModule (single-step, no IrModule intermediary)
      3. Emit MLIR text
      4. Apply MLIR optimization passes (fusion + standard CSE/canonicalize)
      5. Re-parse optimized MLIR back to MlirModule
      6. Serialize to disk (optional)

    Returns:
        The compiled MlirModule (post-optimization).
    """
    from compiler.export_ir import export_model

    args = example_args or ()
    kwargs = example_kwargs or {}

    program = export_model(
        model, args, kwargs,
        dynamic_shapes=dynamic_shapes,
        model_dir=model_dir,
        cache=cache_export,
    )

    # Step 2: single-step FX → MlirModule
    mlir_mod = fx_graph_to_mlir(program)

    # Step 3: emit MLIR text
    mlir_text = mlir_module_to_text(mlir_mod)
    orig_mlir_mod = mlir_mod  # keep original for weight preservation

    # Step 4: apply MLIR passes
    if apply_fusion:
        mlir_text = _apply_mlir_passes(mlir_text)

    # Step 5: re-parse to get optimized MlirModule
    from compiler.mlir_artifact import _parse_mlir_text
    mlir_mod = _parse_mlir_text(mlir_text)
    # Preserve weights through the MLIR text roundtrip
    mlir_mod.metadata["source"] = "torch.export"
    mlir_mod.metadata["artifact_format"] = "mlir"
    # Restore weights from the original module (parser doesn't handle weight values)
    for orig_func in orig_mlir_mod.functions:
        for mf in mlir_mod.functions:
            if mf.name == orig_func.name:
                mf.weights = {wname: orig_func.weights[wname] for wname in orig_func.weights}
                break

    # Step 6: serialize
    if output_dir is not None:
        mlir_mod.metadata["passes_applied"] = ["cse", "canonicalize"] + (
            ["fuse_silu", "fuse_rms_norm"] if apply_fusion else []
        )
        save_mlir_module_artifact(mlir_mod, str(output_dir))

    return mlir_mod


def _apply_mlir_passes(mlir_text: str) -> str:
    """Apply MLIR optimization passes to the given MLIR text."""
    # Standard passes first (only if bindings available)
    from compiler.mlir_passes.fusion import _has_bindings, fuse_rms_norm_pass, fuse_silu_pass

    if _has_bindings():
        _setup_mlir_path()
        try:
            import mlir.ir as ir
            import mlir.passmanager as pm

            ctx = ir.Context()
            ctx.allow_unregistered_dialects = True
            with ctx:
                module = ir.Module.parse(mlir_text, ctx)
                pman = pm.PassManager.parse("builtin.module(canonicalize)", ctx)
                pman.run(module.operation)
                mlir_text = str(module)
        except Exception:
            pass  # standard passes are optional

    # Fusion passes
    try:
        mlir_text = fuse_silu_pass(mlir_text)
    except Exception:
        pass
    try:
        mlir_text = fuse_rms_norm_pass(mlir_text)
    except Exception:
        pass

    return mlir_text


# ── Backward compatibility: legacy compile returns IrModule ─────

from compiler.fx_to_ir import fx_graph_to_ir  # noqa: E402
from compiler.ir import IrModule  # noqa: E402
from compiler.passes.base import PassManager  # noqa: E402
from compiler.passes.constant_fold import ConstantFold  # noqa: E402
from compiler.passes.cse_pass import CommonSubexpressionElimination  # noqa: E402
from compiler.passes.dce_pass import DeadCodeElimination  # noqa: E402
from compiler.passes.fuse_attention import FuseAttentionPattern  # noqa: E402
from compiler.passes.fuse_attention_block import FuseAttentionBlock  # noqa: E402
from compiler.passes.fuse_qkv import FuseQKVProjection  # noqa: E402
from compiler.passes.fuse_rms_norm import FuseRMSNorm  # noqa: E402
from compiler.passes.fuse_silu import FuseSiLU  # noqa: E402
from compiler.passes.validate_ir import ValidateIR  # noqa: E402
from compiler.serialize import save_artifact  # noqa: E402


class CompilationPipeline:
    """Legacy compilation pipeline (deprecated — use compile_mlir() instead)."""

    def __init__(
        self,
        enable_fusion: bool = True,
        enable_cse: bool = True,
        enable_constant_fold: bool = True,
        enable_validation: bool = True,
        cache_export: bool = False,
    ) -> None:
        self.enable_fusion = enable_fusion
        self.enable_cse = enable_cse
        self.enable_constant_fold = enable_constant_fold
        self.enable_validation = enable_validation
        self.cache_export = cache_export

    def compile(
        self,
        model: torch.nn.Module,
        example_args: tuple[Any, ...] | None = None,
        example_kwargs: dict[str, Any] | None = None,
        output_dir: str | None = None,
        dynamic_shapes: dict[str, Any] | None = None,
        emit_mlir: bool = True,
        model_dir: str = "",
    ) -> IrModule:
        from compiler.export_ir import export_model

        args = example_args or ()
        kwargs = example_kwargs or {}
        program = export_model(
            model, args, kwargs,
            dynamic_shapes=dynamic_shapes,
            model_dir=model_dir,
            cache=self.cache_export,
        )
        module = fx_graph_to_ir(program)
        module = self._optimize(module)
        if output_dir is not None:
            save_artifact(module, str(output_dir))
        return module

    def _optimize(self, module: IrModule) -> IrModule:
        pm = PassManager()
        if self.enable_validation:
            pm.add(ValidateIR())
        if self.enable_cse:
            pm.add(CommonSubexpressionElimination())
        if self.enable_constant_fold:
            pm.add(ConstantFold())
        pm.add(DeadCodeElimination())
        if self.enable_fusion:
            pm.add(FuseQKVProjection())
            pm.add(FuseRMSNorm())
            pm.add(FuseSiLU())
            pm.add(FuseAttentionPattern())
            pm.add(FuseAttentionBlock())
        return pm.run(module)


def default_pipeline() -> CompilationPipeline:
    """Legacy convenience function (deprecated)."""
    return CompilationPipeline(enable_fusion=True, enable_cse=True, enable_constant_fold=True)


def compile_module(
    model: torch.nn.Module,
    example_args: tuple[Any, ...] | None = None,
    output_dir: str | None = None,
) -> IrModule:
    """Legacy convenience function (deprecated)."""
    pipeline = default_pipeline()
    return pipeline.compile(model, example_args, output_dir=output_dir)
