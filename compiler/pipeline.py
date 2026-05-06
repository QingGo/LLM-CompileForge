"""Compilation pipeline orchestration.

Assembles the full AOT compilation flow:
  PyTorch model → torch.export → FX → IR → optimization passes → artifact
"""

from __future__ import annotations

from typing import Any

import torch

from compiler.export_ir import export_model
from compiler.fx_to_ir import fx_graph_to_ir
from compiler.ir import IrModule
from compiler.passes.base import PassManager
from compiler.passes.constant_fold import ConstantFold
from compiler.passes.cse_pass import CommonSubexpressionElimination
from compiler.passes.fuse_rms_norm import FuseRMSNorm
from compiler.passes.fuse_silu import FuseSiLU
from compiler.serialize import save_artifact


class CompilationPipeline:
    """Orchestrates the full compilation pipeline.

    Usage:
        pipeline = CompilationPipeline()
        artifact_path = pipeline.compile(model, example_input)
    """

    def __init__(
        self,
        enable_fusion: bool = True,
        enable_cse: bool = True,
        enable_constant_fold: bool = True,
    ) -> None:
        self.enable_fusion = enable_fusion
        self.enable_cse = enable_cse
        self.enable_constant_fold = enable_constant_fold

    def compile(
        self,
        model: torch.nn.Module,
        example_args: tuple[Any, ...] | None = None,
        example_kwargs: dict[str, Any] | None = None,
        output_dir: str | None = None,
    ) -> IrModule:
        """Run the full compilation pipeline.

        Steps:
          1. torch.export → ExportedProgram
          2. FX Graph → IrModule
          3. Apply optimization passes
          4. Return compiled IrModule

        Args:
            model: The PyTorch nn.Module to compile.
            example_args: Dummy inputs for tracing.
            example_kwargs: Dummy kwargs for tracing.
            output_dir: If set, serialize the compiled artifact to this directory.

        Returns:
            The optimized IrModule.

        Raises:
            RuntimeError: If torch.export fails.
        """
        # Step 1: export
        program = export_model(model, example_args, example_kwargs)

        # Step 2: convert to IR
        module = fx_graph_to_ir(program)

        # Step 3: optimize
        module = self._optimize(module)

        # Step 4: serialize (optional)
        if output_dir is not None:
            save_artifact(module, str(output_dir))
            module.metadata["output_dir"] = output_dir

        return module

    def _optimize(self, module: IrModule) -> IrModule:
        """Apply the default pass pipeline."""
        pm = PassManager()

        if self.enable_cse:
            pm.add(CommonSubexpressionElimination())

        if self.enable_constant_fold:
            pm.add(ConstantFold())

        if self.enable_fusion:
            pm.add(FuseRMSNorm())
            pm.add(FuseSiLU())

        return pm.run(module)


def default_pipeline() -> CompilationPipeline:
    """Create a pipeline with all MVP passes enabled."""
    return CompilationPipeline(enable_fusion=True, enable_cse=True, enable_constant_fold=True)


def compile_module(
    model: torch.nn.Module,
    example_args: tuple[Any, ...] | None = None,
    output_dir: str | None = None,
) -> IrModule:
    """Convenience function: compile a model with default settings."""
    pipeline = default_pipeline()
    return pipeline.compile(model, example_args, output_dir=output_dir)
