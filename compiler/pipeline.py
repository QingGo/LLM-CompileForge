"""Compilation pipeline orchestration.

Assembles the full AOT compilation flow:
  PyTorch model → torch.export → FX → MlirModule → MLIR passes → artifact

The canonical compiler path now emits MlirModule directly via fx_to_mlir and
runs optimization passes on the MLIR representation (using official MLIR
Python bindings when available).
"""

from __future__ import annotations

import os
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
    cache_policy: Any | None = None,
    apply_lowering: bool = False,
) -> MlirModule:
    """Compile a PyTorch model through the MLIR-native pipeline.

    Steps:
      1. torch.export → ExportedProgram
      2. FX Graph → MlirModule (single-step, no IrModule intermediary)
      3. Emit MLIR text
      4. Apply MLIR optimization passes (fusion + standard CSE/canonicalize)
      5. Optionally apply sf→linalg lowering (produces model.lowered.mlir)
      6. Re-parse optimized MLIR back to MlirModule
      7. Serialize to disk (optional)

    Args:
        cache_policy: Optional CachePolicy for KV cache strategy.
            Serialized into metadata.json at compile time.
        apply_lowering: If True, run sf→linalg lowering after fusion
            and save the lowered IR as model.lowered.mlir alongside
            the standard sf-dialect model.mlir artifact.

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
    lowered_text: str | None = None
    if apply_fusion:
        mlir_text, lowered_text = _apply_mlir_passes(
            mlir_text, orig_mlir_mod, apply_lowering=apply_lowering,
        )
    elif apply_lowering:
        _, lowered_text = _apply_mlir_passes(
            mlir_text, orig_mlir_mod, apply_lowering=True,
        )

    # Save lowered MLIR text for inspection (sf→linalg output)
    if lowered_text is not None and output_dir is not None:
        from pathlib import Path
        _lowered_path = Path(output_dir) / "model.lowered.mlir"
        _lowered_path.write_text(lowered_text)
        _log.info("lowered MLIR saved to %s (%d lines)",
                   _lowered_path, len(lowered_text.splitlines()))

    # Step 5: re-parse to get optimized MlirModule
    from compiler.mlir_artifact import _parse_mlir_text
    mlir_mod = _parse_mlir_text(mlir_text)
    # Preserve weights through the MLIR text roundtrip
    mlir_mod.metadata["source"] = "torch.export"
    mlir_mod.metadata["artifact_format"] = "mlir"
    # Restore metadata and weights from the original module
    if "hf_key_map" in orig_mlir_mod.metadata:
        mlir_mod.metadata["hf_key_map"] = orig_mlir_mod.metadata["hf_key_map"]
    for orig_func in orig_mlir_mod.functions:
        for mf in mlir_mod.functions:
            if mf.name == orig_func.name:
                mf.weights = {wname: orig_func.weights[wname] for wname in orig_func.weights}
                mf.param_weight_names = set(orig_func.param_weight_names)
                mf.const_weight_names = set(orig_func.const_weight_names)
                break

    # Step 6: serialize
    if output_dir is not None:
        mlir_mod.metadata["passes_applied"] = ["cse", "canonicalize"] + (
            ["fuse_silu", "fuse_rms_norm"] if apply_fusion else []
        ) + (
            ["sf_to_linalg"] if apply_lowering else []
        )
        if cache_policy is not None and hasattr(cache_policy, "to_dict"):
            mlir_mod.metadata["cache_policy"] = cache_policy.to_dict()
        if model_dir:
            safetensors_path = os.path.join(os.path.abspath(model_dir), "model.safetensors")
            bin_path = os.path.join(os.path.abspath(model_dir), "pytorch_model.bin")
            safetensors_index = os.path.join(os.path.abspath(model_dir), "model.safetensors.index.json")
            ws: dict[str, Any] = {}
            if os.path.isfile(safetensors_path):
                ws = {"path": safetensors_path, "format": "safetensors"}
            elif os.path.isfile(safetensors_index):
                ws = {"path": safetensors_index, "format": "safetensors_sharded"}
            elif os.path.isfile(bin_path):
                ws = {"path": bin_path, "format": "pytorch_bin"}
            if ws:
                # Detect tied weights (same tensor shared under different names)
                tied: dict[str, str] = {}
                for func in mlir_mod.functions:
                    wlist = list(func.weights.items())
                    for i, (n1, t1) in enumerate(wlist):
                        for n2, t2 in wlist[i + 1:]:
                            if t1.data_ptr() == t2.data_ptr():
                                tied[n2] = n1
                if tied:
                    ws["tied_weights"] = tied
                mlir_mod.metadata["weight_source"] = ws
        save_mlir_module_artifact(mlir_mod, str(output_dir))

    elapsed_s = time.perf_counter() - _t0
    total_ops = sum(len(f.ops) for f in mlir_mod.functions)
    _log.info("compile complete | %.1fs, %d ops, %d weights | %s%s",
              elapsed_s, total_ops, sum(len(f.weights) for f in mlir_mod.functions),
              "fusion=on" if apply_fusion else "fusion=off",
              " lowering=on" if apply_lowering else "")

    return mlir_mod


def _apply_mlir_passes(
    mlir_text: str, orig_mlir_mod: Any = None, apply_lowering: bool = False,
) -> tuple[str, str | None]:
    """Apply MLIR optimization passes.

    Phase 1: canonicalize (standard MLIR pass)
    Phase 2: fusion (fuse_silu, fuse_rms_norm)
    Phase 3: sf→linalg lowering (optional, after fusion, via API path)

    Returns:
        (optimized_sf_text, lowered_linalg_text_or_None).
        optimized_sf_text is always sf-dialect (suitable for MlirModule re-parse).
        lowered_linalg_text is the mixed-dialect output of sf_to_linalg_pass.
    """
    import logging
    _log = logging.getLogger("compiler.pipeline")

    from compiler.mlir_passes.fusion import _has_bindings, fuse_rms_norm_pass, fuse_silu_pass

    # Phase 1: canonicalize
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

    # Phase 2: fusion
    try:
        mlir_text = fuse_silu_pass(mlir_text)
    except Exception as e:
        _log.warning("fuse_silu pass failed, continuing: %s", e)
    try:
        mlir_text = fuse_rms_norm_pass(mlir_text)
    except Exception as e:
        _log.warning("fuse_rms_norm pass failed, continuing: %s", e)

    # Phase 3: sf→linalg lowering (optional, after fusion, via API path)
    lowered_text: str | None = None
    if apply_lowering:
        lowered_text = _apply_sf_to_linalg(mlir_text, orig_mlir_mod=orig_mlir_mod)

    return mlir_text, lowered_text


def _apply_sf_to_linalg(mlir_text: str, orig_mlir_mod: Any = None) -> str:
    """Apply sf→linalg lowering pass.

    Uses API-based ir.Module construction when an MlirModule is available
    (bypasses MLIR text round-trip parse issues). Falls back to text-based
    path otherwise.
    """
    import logging
    _log = logging.getLogger("compiler.pipeline")

    # Preferred path: API-based, uses MlirModule directly
    if orig_mlir_mod is not None:
        try:
            from compiler.mlir_artifact import mlir_module_to_ir_module
            from compiler.mlir_dialect.lowering import sf_to_linalg_pass_on_module
            ir_mod = mlir_module_to_ir_module(orig_mlir_mod)
            return _post_lowering_canonicalize(sf_to_linalg_pass_on_module(ir_mod))
        except Exception as e:
            _log.warning("API-based sf_to_linalg failed, falling back: %s", e)

    # Fallback: text-based path
    from compiler.mlir_dialect.lowering import sf_to_linalg_pass
    try:
        mlir_text = sf_to_linalg_pass(mlir_text)
    except Exception as e:
        _log.warning("sf_to_linalg pass failed, continuing: %s", e)
        return mlir_text

    return _post_lowering_canonicalize(mlir_text)


def _post_lowering_canonicalize(mlir_text: str) -> str:
    """Run canonicalize pass on lowered MLIR text."""
    import logging
    _log = logging.getLogger("compiler.pipeline")

    from compiler.mlir_passes.fusion import _has_bindings
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
            _log.warning("post-lowering canonicalize failed, continuing: %s", e)

    return mlir_text

