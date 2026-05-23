"""Compilation pipeline orchestration.

Assembles the full AOT compilation flow:
  PyTorch model → torch.export → FX → MlirModule → MLIR passes → artifact

The canonical compiler path now emits MlirModule directly via fx_to_mlir and
runs optimization passes on the MLIR representation (using official MLIR
Python bindings when available).
"""

from __future__ import annotations

import os
import time
import warnings
from typing import Any

import torch

from compiler.fx_to_mlir import fx_graph_to_mlir
from compiler.mlir_artifact import MlirModule, mlir_module_to_text, save_mlir_module_artifact
from compiler.mlir_dialect.compile_utils import _setup_mlir_path


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
    **kwargs,
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

    apply_lowering = kwargs.pop("apply_lowering", False)
    if apply_lowering:
        warnings.warn(
            "apply_lowering=True is deprecated. Use 'python scripts/compile_dylib.py <dir>"
            " --model-name <name>' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {kwargs}")

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
    # NOTE: _parse_mlir_text is a lightweight text parser that may not handle
    # output from the canonicalize pass (which uses a different text format).
    # Fall back to the original MlirModule if re-parse fails.
    from compiler.mlir_artifact import _parse_mlir_text
    try:
        mlir_mod = _parse_mlir_text(mlir_text)
    except (ValueError, IndexError, KeyError) as e:
        _log.warning("re-parse of optimized MLIR text failed, using original module: %s", e)
        mlir_mod = orig_mlir_mod
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
    mlir_text: str, orig_mlir_mod: Any = None, **kwargs,
) -> tuple[str, str | None]:
    """Apply MLIR optimization passes.

    Phase 1: canonicalize + CSE (standard MLIR pass, enabled by sf dialect traits)
    Phase 2: fusion (fuse_silu, fuse_rms_norm)
    Phase 3: sf→linalg lowering (via C++ DialectConversion pass)

    Args:
        apply_lowering (deprecated): If True, run sf→linalg lowering.

    Returns:
        (optimized_sf_text, lowered_linalg_text_or_None).
        optimized_sf_text is always sf-dialect (suitable for MlirModule re-parse).
        lowered_linalg_text is the mixed-dialect output of sf-to-linalg pass.
    """
    import logging

    _log = logging.getLogger("compiler.pipeline")

    apply_lowering = kwargs.pop("apply_lowering", False)
    if apply_lowering:
        warnings.warn(
            "apply_lowering=True is deprecated. Use compile_dylib.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments in _apply_mlir_passes: {kwargs}")

    from compiler.mlir_dialect.compile_utils import _has_bindings
    from compiler.mlir_passes.fusion import fuse_rms_norm_pass, fuse_silu_pass

    # Phase 0: register sf dialect (must happen before parsing MLIR text)
    if _has_bindings():
        _setup_mlir_path()
        try:
            import mlir.ir as ir
            import mlir.passmanager as pm
            from mlir_sf._mlir_libs._sfDialectsNanobind import sf

            ctx = ir.Context()
            sf.register_dialects(ctx._CAPIPtr, load=True)

            # Phase 1: canonicalize + CSE
            # NOTE: This is best-effort.  If anything fails (unusual attrs in
            # the custom text format, op verification, etc.), fall through and
            # continue with the original text — the fusion passes that follow
            # are also text-based and may still succeed.
            try:
                with ctx:
                    if orig_mlir_mod is not None:
                        from compiler.mlir_artifact import mlir_module_to_ir_module
                        module = mlir_module_to_ir_module(orig_mlir_mod, ctx=ctx)
                    else:
                        module = ir.Module.parse(mlir_text, ctx)
                    from compiler.mlir_dialect.fixups import _walk_and_fix_tensor_constants
                    _walk_and_fix_tensor_constants(module)
                    pman = pm.PassManager.parse("builtin.module(canonicalize,cse)", ctx)
                    pman.run(module.operation)
                    mlir_text = str(module)
            except Exception as e:
                raise RuntimeError(f"[pipeline] CRITICAL: canonicalize/cse failed: {e}") from e
        except ImportError as e:
            _log.warning("sf dialect Python bindings not available (canonicalize/cse skipped): %s", e)

    # Phase 2: fusion
    try:
        mlir_text = fuse_silu_pass(mlir_text)
    except Exception as e:
        raise RuntimeError(f"[pipeline] CRITICAL: fuse_silu pass failed: {e}") from e
    try:
        mlir_text = fuse_rms_norm_pass(mlir_text)
    except Exception as e:
        raise RuntimeError(f"[pipeline] CRITICAL: fuse_rms_norm pass failed: {e}") from e

    # Phase 3: sf→linalg lowering (optional, after fusion, via C++ DialectConversion)
    lowered_text: str | None = None
    if apply_lowering:
        lowered_text = _apply_sf_to_linalg(mlir_text, orig_mlir_mod=orig_mlir_mod)

    return mlir_text, lowered_text


def _apply_sf_to_linalg(mlir_text: str, orig_mlir_mod: Any = None) -> str:
    """Apply sf→linalg lowering pass using C++ passes only.

    Uses C++ passes: sf-promote-weights → canonicalize/cse → sf-lower-to-linalg.
    Raises RuntimeError with clear message if C++ bindings are unavailable.
    """
    _setup_mlir_path()
    import mlir.ir as ir
    import mlir.passmanager as pm
    try:
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf
    except ImportError as e:
        raise RuntimeError(
            "sf dialect Python bindings not available. "
            "The C++ lowering pipeline (sf-promote-weights, sf-lower-to-linalg) "
            "requires the sf dialect bindings. "
            "Build: cd sf-dialect && mkdir -p build && cd build && cmake .. && make"
        ) from e

    ctx = ir.Context()
    sf.register_dialects(ctx._CAPIPtr, load=True)
    with ctx:
        if orig_mlir_mod is not None:
            from compiler.mlir_artifact import mlir_module_to_ir_module
            ir_mod = mlir_module_to_ir_module(orig_mlir_mod, ctx=ctx)
        else:
            ir_mod = ir.Module.parse(mlir_text, ctx)
        pman = pm.PassManager.parse(
            "builtin.module(sf-promote-weights,canonicalize,cse,sf-lower-to-linalg)",
            ctx)
        pman.enable_verifier(True)
        pman.enable_timing()
        pman.run(ir_mod.operation)
        _annotate_debug_weight_names(ir_mod)
        mlir_text = ir_mod.operation.get_asm(print_generic_op_form=True)
    return _post_lowering_canonicalize(mlir_text)


def _annotate_debug_weight_names(ir_mod: Any) -> None:
    """Add debug_weight_names on func.func ops so weight tracking survives stripping.

    sf.weight_names (set by sf-promote-weights) is stripped before bufferize
    because unregistered dialect attributes block canonicalize.  debug_weight_names
    uses the debug_ prefix and is not stripped — it survives the full pipeline.
    """
    import mlir.ir as ir

    for op in ir_mod.operation.regions[0].blocks[0]:
        if str(op.operation.name) != "func.func":
            continue
        weight_names = op.operation.attributes.get("sf.weight_names")
        if weight_names is not None:
            op.operation.attributes["debug_weight_names"] = weight_names


def _post_lowering_canonicalize(mlir_text: str) -> str:
    """Run canonicalize pass on lowered MLIR text."""
    from compiler.mlir_dialect.compile_utils import _has_bindings, _setup_mlir_path

    if _has_bindings():
        _setup_mlir_path()
        try:
            import mlir.ir as ir
            import mlir.passmanager as pm

            ctx = ir.Context()
            with ctx:
                module = ir.Module.parse(mlir_text, ctx)
                pman = pm.PassManager.parse("builtin.module(canonicalize)", ctx)
                pman.run(module.operation)
                mlir_text = str(module)
        except Exception as e:
            raise RuntimeError(f"[pipeline] CRITICAL: post-lowering canonicalize failed: {e}") from e

    # Fix arith.constant ops with scalar value + tensor result type
    try:
        from compiler.mlir_dialect.fixups import _fixup_arith_tensor_constants_mlir
        mlir_text = _fixup_arith_tensor_constants_mlir(mlir_text)
    except Exception as e:
        raise RuntimeError(f"[pipeline] CRITICAL: arith.constant scalar→tensor fixup failed: {e}") from e

    return mlir_text

