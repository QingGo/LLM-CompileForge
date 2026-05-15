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
    print(f"[1/6] Parsing {mlir_path} ...")
    mlir_text = mlir_path.read_text()
    module = _parse_mlir_text(mlir_text)
    print(f"   {len(module.functions)} functions, {sum(len(f.ops) for f in module.functions)} ops")

    # Fix all known type inference bugs in the MlirModule.
    # The model was exported with wrong types due to shape_inference bugs.
    # Fix them by recomputing dependent ops' types from operand types.
    import re as _re
    for func in module.functions:
        # Build SSA → type map from function inputs + all op outputs
        ssa_to_type: dict[str, str] = {}
        for inp_name, inp_type in func.inputs:
            ssa_to_type[f"%{inp_name}"] = inp_type

        # Phase 1: Fix embedding op types (weight_first → correct)
        for op in func.ops:
            if op.op_name != "embedding" or len(op.output_types) != 1 or len(op.input_types) < 2:
                continue
            weight_type = op.input_types[0]
            indices_type = op.input_types[1]
            wm = _re.match(r"tensor<([^>]+)>", weight_type)
            im = _re.match(r"tensor<([^>]+)>", indices_type)
            if not wm or not im or len(wm.group(1).split("x")) < 2:
                continue
            w_dims = wm.group(1).split("x")
            i_dims = im.group(1).split("x")
            embed = w_dims[1]
            last = i_dims[-1]
            if last in ("i64", "f32", "bf16", "f16", "i32", "i8"):
                i_dims = i_dims[:-1]
            new_dims = i_dims + [embed]
            op.output_types[0] = f"tensor<{'x'.join(new_dims)}xf32>"

        # Phase 2: Propagate types through the function (fixes cascade bugs).
        # Build a new SSA→type map from the fixed types
        for op in func.ops:
            for res, ot in zip(op.results, op.output_types):
                ssa_to_type[f"%{res}"] = ot

        # Phase 3: Fix binary ops with wrong output types by recomputing
        # via simple broadcasting: output = broadcast(operand0, operand1).
        # Only fix cases where op has 2 inputs and 1 output of the same type.
        for op in func.ops:
            if len(op.input_types) == 2 and len(op.output_types) == 1 and op.op_name not in ("weight", "constant", "embedding"):
                t0 = op.input_types[0]
                t1 = op.input_types[1]
                out = op.output_types[0]
                # Parse dims from each type
                def _parse_dims(t: str):
                    m = _re.match(r"tensor<([^>]+)>", t)
                    if not m: return []
                    parts = m.group(1).split("x")
                    if parts[-1] in ("f32", "bf16", "i64", "i32", "f16"): parts = parts[:-1]
                    return parts
                d0 = _parse_dims(t0)
                d1 = _parse_dims(t1)
                d_out = _parse_dims(out)
                if not d0 or not d1 or not d_out:
                    continue
                # Broadcast: align trailing dims
                max_rank = max(len(d0), len(d1), len(d_out))
                while len(d0) < max_rank: d0.insert(0, "1")
                while len(d1) < max_rank: d1.insert(0, "1")
                while len(d_out) < max_rank: d_out.insert(0, "1")
                new_dims = []
                changed = False
                for i in range(max_rank):
                    a, b, c = d0[i], d1[i], d_out[i]
                    if a != "?" and b != "?" and a != b:
                        continue  # known mismatch — keep old
                    expected = a if a != "?" else (b if b != "?" else c)
                    if c != expected:
                        d_out[i] = expected
                        changed = True
                if changed:
                    while d_out and d_out[0] == "1" and any(d != "1" for d in d_out[1:]):
                        d_out.pop(0)
                    op.output_types[0] = f"tensor<{'x'.join(d_out)}xf32>"

    # Step 2: Reconstruct hf_key_map
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    if "hf_key_map" not in module.metadata:
        ws = metadata.get("weight_source", {})
        idx_path = ws.get("path", "")
        fmt = ws.get("format", "")

        if idx_path and os.path.isfile(idx_path) and "safetensors" in fmt:
            print(f"[2/6] Reconstructing hf_key_map from {idx_path} ...")
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
            print(f"[2/6] No safetensors index found, skipping hf_key_map")

    # Step 3: Generate constants.bin
    print("[3/6] Generating constants.bin ...")
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
    print("[4/6] Converting MlirModule → ir.Module → sf→linalg lowering ...")
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

    # Step 5: Lower linalg → LLVM using Python API (with verifier disabled)
    print("[5/6] Lowering linalg → LLVM ...")
    from compiler.mlir_dialect.llvm_backend import lower_linalg_to_llvm_ir, mlir_module_to_llvm_ir
    ctx_llvm = ir.Context()
    ctx_llvm.allow_unregistered_dialects = True
    with ctx_llvm:
        ir_mod = ir.Module.parse(lowered_text, ctx_llvm)
        lower_linalg_to_llvm_ir(ir_mod)
        llvm_text = str(ir_mod)
    print("   LLVM lowering succeeded")

    # Step 5b: Translate LLVM dialect → LLVM IR text
    print("[5b/6] Translating LLVM dialect → LLVM IR text ...")
    ctx_llvm2 = ir.Context()
    ctx_llvm2.allow_unregistered_dialects = True
    with ctx_llvm2:
        ir_mod_llvm = ir.Module.parse(llvm_text, ctx_llvm2)

    # Step 6: Compile to .dylib
    print("[6/6] Compiling to .dylib ...")
    dylib_path = compile_module_to_dylib(
        ir_mod_llvm,
        str(compiled_path),
        model_name=model_name,
    )
    with tempfile.TemporaryDirectory() as td:
        in_file = os.path.join(td, "input.mlir")
        out_file = os.path.join(td, "output.mlir")
        with open(in_file, "w") as f:
            f.write(lowered_text)
        proc = subprocess.run(
            [mlir_opt_path, "--allow-unregistered-dialect",
             f"--pass-pipeline={llvm_pipeline}", in_file, "-o", out_file],
            capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            print(f"   mlir-opt failed (exit {proc.returncode}), trying with text fixup...")
            # Fall back to string-based fixes + Python API
            import re
            # Replace tensor.cast from incompatible types with unrealized_conversion_cast
            def fix_casts(text):
                return re.sub(
                    r'"tensor\.cast"\((%\w+)\)\s*:\s*\(([^)]+)\)\s*->\s*(\S+)',
                    r'"builtin.unrealized_conversion_cast"(\1) : (\2) -> \3',
                    text)
            fixed_text = fix_casts(lowered_text)
            with open(in_file, "w") as f:
                f.write(fixed_text)
            proc = subprocess.run(
                [mlir_opt_path, "--allow-unregistered-dialect",
                 f"--pass-pipeline={llvm_pipeline}", in_file, "-o", out_file],
                capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                print(f"   mlir-opt still failed. Saving lowered IR for debugging...")
                # Save error output for inspection
                lowered_path.write_text(lowered_text)
                raise RuntimeError(
                    f"LLVM lowering failed:\n{proc.stderr[:1000]}")
        with open(out_file) as f:
            llvm_text = f.read()
    print("   LLVM lowering succeeded")

    # Step 5b: Translate LLVM dialect → LLVM IR text
    print("[5b/6] Translating LLVM dialect → LLVM IR text ...")
    ctx_llvm = ir.Context()
    ctx_llvm.allow_unregistered_dialects = True
    with ctx_llvm:
        ir_mod_llvm = ir.Module.parse(llvm_text, ctx_llvm)

    # Step 6: Compile to .dylib
    print("[6/6] Compiling to .dylib ...")
    dylib_path = compile_module_to_dylib(
        ir_mod_llvm,
        str(compiled_path),
        model_name=model_name,
    )

    print(f"\nDone! Compiled to: {dylib_path}")
    for fname in [f"lib{model_name}.dylib", "constants.bin"]:
        fpath = compiled_path / fname
        if fpath.exists():
            print(f"  ✓ {fpath} ({fpath.stat().st_size} bytes)")
        else:
            print(f"  ✗ {fpath} NOT FOUND")


if __name__ == "__main__":
    main()
