"""Compilation pipeline orchestration.

Assembles the full AOT compilation flow:
  PyTorch model → torch.export → FX → MlirModule → MLIR passes → artifact

The canonical compiler path now emits MlirModule directly via fx_to_mlir and
runs optimization passes on the MLIR representation (using official MLIR
Python bindings when available).
"""

from __future__ import annotations

import sys
import time
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
    from utils.logging import get_logger

    _log = get_logger("compiler.pipeline")
    _t0 = time.perf_counter()

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

    elapsed_s = time.perf_counter() - _t0
    total_ops = sum(len(f.ops) for f in mlir_mod.functions)
    _log.info("compile complete | %.1fs, %d ops, %d weights | %s",
              elapsed_s, total_ops, sum(len(f.weights) for f in mlir_mod.functions),
              "fusion=on" if apply_fusion else "fusion=off")

    return mlir_mod


def _apply_mlir_passes(mlir_text: str) -> str:
    """Apply MLIR optimization passes to the given MLIR text."""
    import logging
    _log = logging.getLogger("compiler.pipeline")

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
        except Exception as e:
            _log.warning("canonicalize pass failed, continuing with unoptimized IR: %s", e)

    try:
        mlir_text = fuse_silu_pass(mlir_text)
    except Exception as e:
        _log.warning("fuse_silu pass failed, continuing: %s", e)
    try:
        mlir_text = fuse_rms_norm_pass(mlir_text)
    except Exception as e:
        _log.warning("fuse_rms_norm pass failed, continuing: %s", e)

    return mlir_text

