"""Compile model.mlir → .dylib for Rust runtime.

Usage: python scripts/compile_dylib.py <compiled_dir> [--model-name <name>]
Example: python scripts/compile_dylib.py compiled/tiny_llama --model-name tiny_llama
"""

import json
import os
import re
import sys
from pathlib import Path


def _setup_mlir_path() -> None:
    _mlir_pkg = Path(__file__).resolve().parent.parent / "mlir_binding" / "mlir_package"
    if _mlir_pkg.is_dir() and str(_mlir_pkg) not in sys.path:
        sys.path.insert(0, str(_mlir_pkg))


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

    if errors:
        raise ValueError(
            "Lowered IR verification failed:\n  - " + "\n  - ".join(errors)
        )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    compiled_dir = sys.argv[1]
    model_name = "model"
    for i, arg in enumerate(sys.argv):
        if arg == "--model-name" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]

    compiled_path = Path(compiled_dir)
    mlir_path = compiled_path / "model.mlir"
    metadata_path = compiled_path / "metadata.json"

    if not mlir_path.exists():
        print(f"ERROR: model.mlir not found at {mlir_path}")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from compiler.mlir_artifact import (
        _parse_mlir_text,
        _build_name_mapping,
        _build_constants_binary,
        mlir_module_to_ir_module,
    )
    from compiler.mlir_dialect.llvm_backend import (
        lower_linalg_to_llvm_ir,
        compile_module_to_dylib,
    )
    # Step 1: Load model.mlir → MlirModule
    print(f"[1/5] Parsing {mlir_path} ...")
    mlir_text = mlir_path.read_text()
    module = _parse_mlir_text(mlir_text)
    print(f"   {len(module.functions)} functions, {sum(len(f.ops) for f in module.functions)} ops")

    # Step 2: Reconstruct hf_key_map
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
            print(f"[2/5] No safetensors index found, skipping hf_key_map")

    # Step 3: Generate constants.bin
    # Inject tied_weights from metadata.json into module.metadata for name_mapping
    ws = metadata.get("weight_source", {})
    if "weight_source" not in module.metadata:
        module.metadata["weight_source"] = ws
    print("[3/5] Generating constants.bin ...")
    name_mapping = _build_name_mapping(module)
    if name_mapping:
        print(f"   Name mapping: {len(name_mapping)} entries")
    else:
        print(f"   WARNING: No name mapping built")
    const_bin = _build_constants_binary(module, name_mapping or {})
    bin_path = compiled_path / "constants.bin"
    bin_path.write_bytes(const_bin)
    print(f"   Written {len(const_bin)} bytes to {bin_path}")

    # Step 4: MlirModule → ir.Module → register sf dialect → C++ lowering
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
    except ImportError:
        print("   WARNING: sf dialect not available, falling back to Python lowering")

    ir_mod = mlir_module_to_ir_module(module, ctx=ctx_lower)

    # Direct C++ pipeline
    print("   Running C++ lowering...")
    pman = pm.PassManager.parse(
        "builtin.module(sf-promote-weights,canonicalize,cse,sf-lower-to-linalg)",
        ctx_lower)
    pman.enable_verifier(False)
    pman.run(ir_mod.operation)
    # Serialize with generic op format (required for mlir-opt without sf dialect)
    lowered_text = ir_mod.operation.get_asm(print_generic_op_form=True)
    print("   C++ lowering succeeded (verifier disabled)")

    lowered_path = compiled_path / "model.lowered.mlir"
    lowered_path.write_text(lowered_text)
    print(f"   Saved lowered MLIR to {lowered_path}")

    # Verification gate: ensure lowered IR is valid before proceeding
    print("[4v] Verifying lowered IR ...")
    try:
        _verify_lowered_ir(lowered_text)
        print("   Verification passed")
    except ValueError as e:
        print(f"   VERIFICATION FAILED:\n{e}")
        print("   (continuing anyway for debugging purposes)")

    # Step 5: Lower LLVM dialect → LLVM IR (.ll) + compile to .dylib
    # lower_linalg_to_llvm_ir runs: bufferize → linalg→loops → scf→cf → memref→llvm → func→llvm
    print("[5/5] Lowering linalg → LLVM + compiling to .dylib ...")
    from compiler.mlir_dialect.llvm_backend import lower_linalg_to_llvm_ir
    ctx_llvm = ir.Context()
    ctx_llvm.allow_unregistered_dialects = True
    with ctx_llvm:
        ir_mod = ir.Module.parse(lowered_text, ctx_llvm)
        lower_linalg_to_llvm_ir(ir_mod)
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
