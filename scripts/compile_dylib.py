"""Compile model.mlir → .dylib for Rust runtime.

Usage: python scripts/compile_dylib.py <compiled_dir> [--model-name <name>]
Example: python scripts/compile_dylib.py compiled/tiny_llama --model-name tiny_llama
"""

import faulthandler
import json
import logging
import os
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

faulthandler.enable()

DEBUG: bool = False


from compiler.mlir_dialect.compile_utils import _setup_mlir_path


def _verify_lowered_ir(lowered_text: str) -> None:
    """Verify lowered IR contains no illegal ops that would fail re-parse."""
    errors: list[str] = []

    # 1. No bare arith ops on tensors at module level (must be inside linalg.generic)
    bare_arith = re.findall(
        r'%\d+\s+=\s+"(arith\.\w+)"\(',
        lowered_text,
    )
    if bare_arith:
        tensor_arith = re.findall(
            r'"arith\.(mul|add|sub|div)f".*tensor<',
            lowered_text,
        )
        if tensor_arith:
            for op in tensor_arith:
                errors.append(
                    f"Bare arith.{op}f on tensor detected — should be inside linalg.generic"
                )

    # 2. No unresolved sf.* ops (except sf.weight/sf.constant which are handled later)
    sf_ops = set(re.findall(r'"sf\.(\w+)"', lowered_text))
    sf_ignored = {"weight", "constant"}
    unresolved = sf_ops - sf_ignored
    if unresolved:
        errors.append(f"Unresolved sf ops remaining: {sorted(unresolved)}")

    # 3. Must contain at least one linalg op (sanity check)
    if "linalg." not in lowered_text and "scf." not in lowered_text:
        errors.append("No linalg or scf ops found — lowering may have produced nothing")

    # 4. Warn about 0D tensors (tensor<f32>, tensor<i64>, etc. — no dimensions)
    zero_dim_tensors = re.findall(
        r'tensor<(f32|f64|i1|i8|i16|i32|i64)>',
        lowered_text,
    )
    if zero_dim_tensors:
        logging.warning(
            "Lowered IR contains %d zero-dimensional tensor(s) — "
            "types: %s",
            len(zero_dim_tensors),
            sorted(set(zero_dim_tensors)),
        )

    if errors:
        raise ValueError(
            "Lowered IR verification failed:\n  - " + "\n  - ".join(errors)
        )


def _save_failure_context(
    step_num: str,
    pass_name: str,
    compiled_path: Path,
    ir_text: str | None = None,
    copy_source: str | None = None,
) -> Path:
    """Save diagnostic context on pipeline failure and print diagnosis guide."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failure_dir = Path("logs/pipeline") / f"failure_{timestamp}"
    failure_dir.mkdir(parents=True, exist_ok=True)

    if ir_text is not None:
        ir_path = failure_dir / "model.snapshot.mlir"
        ir_path.write_text(ir_text)
        print(f"   Saved IR snapshot: {ir_path}")

    if copy_source is not None:
        src = Path(copy_source)
        if src.exists():
            shutil.copy2(str(src), str(failure_dir / "source.mlir"))
            print(f"   Saved source MLIR: {failure_dir / 'source.mlir'}")

    error_path = failure_dir / "error.txt"
    with open(error_path, "w") as f:
        f.write(f"Step: [{step_num}/5] ({pass_name})\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Compiled dir: {compiled_path}\n")
        f.write("-" * 60 + "\n")
        traceback.print_exc(file=f)
    print(f"   Error details saved to: {error_path}")

    print(f"\n❌ Pipeline failed at step [{step_num}/5] ({pass_name})")
    print(f"📍 Diagnostic context saved to: {failure_dir}")
    print("📋 Suggested next steps:")
    print("   1. Check saved IR files for unexpected ops")
    print("   2. Run: python scripts/bisect_pipeline_stages.py --auto")
    print("   3. Or use: python scripts/compile_dylib.py --debug for per-pass snapshots")

    return failure_dir


def main() -> None:
    from utils.logging import init_logging
    init_logging()
    global DEBUG
    if "--debug" in sys.argv:
        DEBUG = True
        sys.argv.remove("--debug")
        print("[debug] Debug mode enabled")

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    compiled_dir = sys.argv[1]
    model_name = "model"
    for i, arg in enumerate(sys.argv):
        if arg == "--model-name" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]

    compiled_path = Path(compiled_dir)
    snapshot_dir: str | None = None
    if DEBUG:
        snapshot_dir = str(compiled_path / "debug_snapshots")
        print(f"  [debug] Snapshot directory: {snapshot_dir}")

    mlir_path = compiled_path / "model.mlir"
    metadata_path = compiled_path / "metadata.json"

    if not mlir_path.exists():
        print(f"ERROR: model.mlir not found at {mlir_path}")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from compiler.mlir_artifact import (
        _build_constants_binary,
        _build_name_mapping,
        _parse_mlir_text,
        mlir_module_to_ir_module,
    )
    from compiler.mlir_dialect.llvm_backend import (
        compile_module_to_dylib,
        lower_linalg_to_llvm_ir,
    )
    # Step 1: Load model.mlir → MlirModule
    if DEBUG:
        print(f"  [debug] Step [1/5] starting: parse {mlir_path}")
    print(f"[1/5] Parsing {mlir_path} ...")
    mlir_text = mlir_path.read_text()
    module = _parse_mlir_text(mlir_text)
    print(f"   {len(module.functions)} functions, {sum(len(f.ops) for f in module.functions)} ops")

    # Load metadata first (needed for weight classification restoration)
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    # Restore weight classification from metadata.json
    if "weight_classification" not in module.metadata and metadata:
        module.metadata["weight_classification"] = metadata.get("weight_classification", {})
    wc = module.metadata.get("weight_classification", {})
    for func in module.functions:
        fwc = wc.get(func.name, {})
        func.param_weight_names = set(fwc.get("params", []))
        func.const_weight_names = set(fwc.get("constants", []))

    # Restore constant tensors from constants.pth
    const_pth = compiled_path / "constants.pth"
    if const_pth.exists():
        import torch
        const_state = torch.load(str(const_pth), weights_only=True)
        restored = 0
        for func in module.functions:
            for wname in list(func.const_weight_names):
                prefixed = f"{func.name}.{wname}"
                if prefixed in const_state:
                    func.weights[wname] = const_state[prefixed]
                    restored += 1
                elif wname in const_state:
                    func.weights[wname] = const_state[wname]
                    restored += 1
        print(f"   Restored {restored} constant tensors from constants.pth")
    else:
        print("   No constants.pth found")

    # Step 2: Reconstruct hf_key_map
    if DEBUG:
        print("  [debug] Step [2/5] starting: reconstruct hf_key_map")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    if "hf_key_map" not in module.metadata:
        # First try metadata.json (stored at compile time)
        hf_from_meta = metadata.get("hf_key_map", {})
        if hf_from_meta:
            print(f"[2/5] Loading hf_key_map from metadata.json ({len(hf_from_meta)} entries)")
            module.metadata["hf_key_map"] = hf_from_meta
        else:
            ws = metadata.get("weight_source", {})
            idx_path = ws.get("path", "")
            fmt = ws.get("format", "")

        if not hf_from_meta and idx_path and os.path.isfile(idx_path) and "safetensors" in fmt:
            print(f"[2/5] Reconstructing hf_key_map from {idx_path} ...")
            with open(idx_path) as f:
                index = json.load(f)

            hf_key_map: dict[str, str] = {}
            for hf_key in index.get("weight_map", {}):
                clean = hf_key.replace(".", "_")
                hf_key_map[clean] = hf_key

            weight_names: set[str] = set()
            for func in module.functions:
                for op in func.ops:
                    if op.op_name == "weight":
                        weight_names.add(op.attributes.get("name", ""))

            matched = 0
            final_map: dict[str, str] = {}
            for mlir_name in sorted(weight_names):
                if not mlir_name:
                    continue
                if mlir_name in hf_key_map:
                    final_map[mlir_name] = hf_key_map[mlir_name]
                    matched += 1
                    continue
                for clean, hf in hf_key_map.items():
                    if clean.endswith("_" + mlir_name) or clean.endswith(mlir_name):
                        final_map[mlir_name] = hf
                        matched += 1
                        break
                if mlir_name not in final_map:
                    for clean, hf in hf_key_map.items():
                        parts = clean.split("_")
                        for idx in range(1, min(4, len(parts))):
                            candidate = "_".join(parts[idx:])
                            if candidate == mlir_name or candidate.endswith(mlir_name):
                                final_map[mlir_name] = hf
                                matched += 1
                                break
                        if mlir_name in final_map:
                            break

            module.metadata["hf_key_map"] = final_map
            print(f"   Matched {matched}/{len(weight_names)} weight names")
        else:
            print("[2/5] No safetensors index found, skipping hf_key_map")

    # Step 3: Generate constants.bin
    if DEBUG:
        print("  [debug] Step [3/5] starting: generate constants.bin")
    # Inject tied_weights from metadata.json into module.metadata for name_mapping
    ws = metadata.get("weight_source", {})
    if "weight_source" not in module.metadata:
        module.metadata["weight_source"] = ws
    print("[3/5] Generating constants.bin ...")
    name_mapping = _build_name_mapping(module)
    if name_mapping:
        print(f"   Name mapping: {len(name_mapping)} entries")
    else:
        print("   WARNING: No name mapping built")
    const_bin = _build_constants_binary(module, name_mapping or {})
    bin_path = compiled_path / "constants.bin"
    bin_path.write_bytes(const_bin)
    print(f"   Written {len(const_bin)} bytes to {bin_path}")

    # Step 4: MlirModule → ir.Module → register sf dialect → C++ lowering
    if DEBUG:
        print("  [debug] Step [4/5] starting: convert to ir.Module and lower to linalg")
    print("[4/5] Converting MlirModule → ir.Module → sf→linalg lowering ...")
    _setup_mlir_path()
    import mlir.ir as ir
    import mlir.passmanager as pm

    # Register sf dialect FIRST, before creating ops. MLIR requires dialect
    # registration before op creation for typed op access in C++ passes.
    ctx_lower = ir.Context()
    try:
        from mlir_sf._mlir_libs._sfDialectsNanobind import sf
        sf.register_dialects(ctx_lower._CAPIPtr, load=True)
    except ImportError as e:
        print("   ERROR: sf dialect Python bindings not available.")
        print("   The C++ lowering pipeline (sf-promote-weights, sf-lower-to-linalg)")
        print("   requires the sf dialect bindings. Install/build the sf-dialect")
        print("   Python package first.")
        print(f"   ImportError: {e}")
        sys.exit(1)

    ir_mod = mlir_module_to_ir_module(module, ctx=ctx_lower)

    # Direct C++ pipeline (separate passes to allow fixup between them)
    print("   Running C++ lowering...")
    _pass_pipelines = [
        ("sf-promote-weights", "builtin.module(sf-promote-weights)"),
        ("canonicalize", "builtin.module(canonicalize)"),
        ("cse", "builtin.module(cse)"),
        ("sf-lower-to-linalg", "builtin.module(sf-lower-to-linalg)"),
    ]
    _no_verify = "--no-verify" in sys.argv
    if _no_verify:
        sys.argv.remove("--no-verify")

    for _pass_name, _pipeline_str in _pass_pipelines:
        if _pass_name in ("canonicalize", "cse"):
            # canonicalize/cse runs between promote-weights and lowering.
            # Serialize to text, run _fixup_arith_constant_scalar_tensor,
            # then re-parse before continuing. This ensures proper types.
            _pass_text = ir_mod.operation.get_asm(print_generic_op_form=True)
            from compiler.mlir_dialect.fixups import _fixup_arith_constant_scalar_tensor
            _fixed = _fixup_arith_constant_scalar_tensor(_pass_text)
            if _fixed != _pass_text:
                _pass_ctx = ir.Context()
                _pass_ctx.allow_unregistered_dialects = True
                # Register sf dialect so subsequent lowering passes can match typed ops
                try:
                    from mlir_sf._mlir_libs._sfDialectsNanobind import sf as _sf
                    _sf.register_dialects(_pass_ctx._CAPIPtr, load=True)
                except ImportError:
                    pass  # sf dialect may not be available in all environments
                with ir.Location.unknown(_pass_ctx):
                    ir_mod = ir.Module.parse(_fixed, _pass_ctx)
                ctx_lower = _pass_ctx
        try:
            _pman = pm.PassManager.parse(_pipeline_str, ctx_lower)
            if not _no_verify:
                _pman.enable_verifier(True)
            _pman.run(ir_mod.operation)
        except Exception:
            _save_failure_context("4", _pass_name, compiled_path,
                                  copy_source=str(mlir_path))
            raise
        if DEBUG:
            _snapshot_text = ir_mod.operation.get_asm(print_generic_op_form=True)
            _snapshot_path = Path(snapshot_dir) / f"snapshot_{_pass_name}.mlir"
            _snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            _snapshot_path.write_text(_snapshot_text)
            print(f"  [debug] Saved snapshot: {_snapshot_path}")
    # Serialize with generic op format (required for mlir-opt without sf dialect)
    lowered_text = ir_mod.operation.get_asm(print_generic_op_form=True)
    print(f"   C++ lowering succeeded (verifier {'disabled' if _no_verify else 'enabled'})")

    # Verify weight promotion ordering: check that sf.weight_names in lowered IR
    # matches weight op order from the Python MlirModule (dict insertion order
    # from fx_to_mlir.py Phase 4). Catches order mismatches between the compute
    # graph binary (built before C++ lowering) and the promoted function args.
    from scripts.verify_weight_consistency import verify_weight_promotion_order
    try:
        weight_errors = verify_weight_promotion_order(module, lowered_text)
    except Exception as e:
        print(f"   ⚠ Weight promotion check crashed: {e}")
        weight_errors = None
    if weight_errors:
        _err_msg = "Weight promotion order mismatch:\n  " + "\n  ".join(weight_errors)
        print(f"\n❌ {_err_msg}")
        raise RuntimeError(_err_msg)
    print("   ✔ Weight promotion order verified")

    lowered_path = compiled_path / "model.lowered.mlir"
    lowered_path.write_text(lowered_text)
    print(f"   Saved lowered MLIR to {lowered_path}")

    import re as _re
    _lines = lowered_text.split('\n')
    _const_vals = {}
    for _line in _lines:
        _cm = _re.match(
            r'\s*(%\w+)\s*=\s*"arith\.constant"\s*\(\)\s*<{value\s*=\s*(\d+)\s*:\s*index}>',
            _line,
        )
        if _cm:
            _const_vals[_cm.group(1)] = int(_cm.group(2))
    _dim_records = []
    for _li, _line in enumerate(_lines):
        _dm = _re.match(
            r'\s*(%\w+)\s*=\s*"tensor\.dim"\s*\(\s*(%\w+)\s*,\s*(%\w+|\d+)\s*\)',
            _line,
        )
        if _dm:
            _dim_ref = _dm.group(3)
            _dim_idx = int(_dim_ref) if _dim_ref.isdigit() else _const_vals.get(_dim_ref, -1)
            if _dim_idx >= 0:
                _dim_records.append((_li, _dim_idx, _dm.group(1)))
    _changes = 0
    for _li in range(len(_lines)):
        _me = _re.match(
            r'(\s*%\w+\s*=\s*"tensor\.empty"\s*\()\s*(\)\s*:\s*\(\)\s*->\s*tensor<(.+?)>\s*$)',
            _lines[_li],
        )
        if not _me:
            continue
        _type = _me.group(3)
        _shape_part = _type.rsplit('x', 1)[0]
        if '?' not in _shape_part:
            continue
        _dims = [d.strip() for d in _shape_part.split('x') if d.strip()]
        _dyn_pos = [i for i, d in enumerate(_dims) if '?' in d]
        _sizes = {}
        for _dl, _di, _dv in reversed(_dim_records):
            if _dl < _li and _di in _dyn_pos and _di not in _sizes:
                _sizes[_di] = _dv
                if len(_sizes) == len(_dyn_pos):
                    break
        if len(_sizes) == len(_dyn_pos):
            _prefix = _me.group(1)
            _sorted = [_sizes[p] for p in sorted(_sizes.keys())]
            _lines[_li] = (
                f'{_prefix}{", ".join(_sorted)})'
                f' : ({", ".join(["index"] * len(_dyn_pos))})'
                f' -> tensor<{_type}>'
            )
            _changes += 1
    if _changes:
        lowered_text = '\n'.join(_lines)
        lowered_path.write_text(lowered_text)
        print(f"   Fixed {_changes} tensor.empty ops with dynamic sizes")

    # Fix arith.constant ops: scalar value + tensor result → dense attr
    from compiler.mlir_dialect.fixups import _fixup_arith_constant_scalar_tensor
    _before = lowered_text
    lowered_text = _fixup_arith_constant_scalar_tensor(lowered_text)
    if lowered_text != _before:
        lowered_path.write_text(lowered_text)

    # Verification gate: ensure lowered IR is valid before proceeding
    if DEBUG:
        print("  [debug] Step [4v/5] starting: verify lowered IR")
    print("[4v] Verifying lowered IR ...")
    try:
        _verify_lowered_ir(lowered_text)
        print("   Verification passed")
    except ValueError as e:
        print(f"   VERIFICATION FAILED:\n{e}")
        _save_failure_context("4v", "IR verification", compiled_path,
                              ir_text=lowered_text)
        print("   (continuing anyway for debugging purposes)")

    # Step 5: Lower LLVM dialect → LLVM IR (.ll) + compile to .dylib
    # lower_linalg_to_llvm_ir runs: bufferize → linalg→loops → scf→cf → memref→llvm → func→llvm
    if DEBUG:
        print("  [debug] Step [5/5] starting: lower to LLVM and compile .dylib")
    print("[5/5] Lowering linalg → LLVM + compiling to .dylib ...")
    if _no_verify:
        print("   [no-verify] Skipping BUILTIN_STAGES stage 1 canonicalize,cse")
    ctx_llvm = ir.Context()
    ctx_llvm.allow_unregistered_dialects = True
    with ctx_llvm:
        ir_mod = ir.Module.parse(lowered_text, ctx_llvm)
        try:
            lower_linalg_to_llvm_ir(ir_mod, skip_first_canonicalize=_no_verify)
        except Exception:
            _save_failure_context("5", "LLVM lowering", compiled_path,
                                  copy_source=str(lowered_path))
            raise
    print("   LLVM dialect lowering succeeded")

    dylib_path = compile_module_to_dylib(
        ir_mod,
        str(compiled_path),
        model_name=model_name,
    )

    print(f"\nCompilation complete: {dylib_path}")
    for fname in [f"lib{model_name}.dylib", "constants.bin"]:
        fpath = compiled_path / fname
        if fpath.exists():
            print(f"  ✓ {fpath} ({fpath.stat().st_size} bytes)")
        else:
            print(f"  ✗ {fpath} NOT FOUND")


if __name__ == "__main__":
    main()
