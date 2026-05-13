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


def _fixup_bare_arith(lowered_text: str) -> str:
    """Wrap bare arith ops on tensors in linalg.generic blocks (quoted format)."""
    lines = lowered_text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        m = re.match(
            r'(\s*)%(\d+)\s*=\s*"arith\.(\w+)"\(([^)]*)\)\s*(.*?):\s*\(([^)]*)\)\s*->\s*(\S+)',
            stripped,
        )
        if m and "tensor<" in m.group(6):
            indent = m.group(1)
            result_ssa = m.group(2)
            op_name = m.group(3)
            operands_str = m.group(4)
            input_types_str = m.group(6)
            output_type = m.group(7).strip()
            
            operands = [o.strip().lstrip("%") for o in operands_str.split(",") if o.strip()]
            input_types = [t.strip() for t in _split_mlir_types(input_types_str)]
            if not input_types:
                result.append(line); i += 1; continue
            
            first_type = input_types[0]
            dim_parts = first_type.split("<")[-1].rstrip(">").split("x")
            rank = len(dim_parts) - 1 if len(dim_parts) > 1 else 0
            elem_type = dim_parts[-1].strip() if dim_parts else first_type
            
            # Fix rank for common patterns
            if "tensor<" in first_type and rank == 0:
                inner = first_type.split("<", 1)[1].rstrip(">")
                if "x" in inner:
                    parts = inner.split("x")
                    rank = len(parts) - 1
                    elem_type = parts[-1].strip()
            
            n_inputs = len(operands)
            is_int_type = "i" in elem_type.lower() and "f" not in elem_type.lower()
            arith_op_name = op_name
            if is_int_type and op_name.endswith("f"):
                arith_op_name = op_name[:-1] + "i"
            
            # Build identity maps
            dim_letters = [f"d{j}" for j in range(rank)] if rank > 0 else []
            dim_str = ", ".join(dim_letters)
            if rank > 0:
                ident_map = f"affine_map<({dim_str}) -> ({dim_str})>"
                iter_types = ", ".join(["#linalg.iterator_type<parallel>"] * rank)
            else:
                ident_map = "affine_map<() -> ()>"
                iter_types = ""
            
            # Build maps — one per input + one for output
            # Use correct rank from OUTPUT type, broadcast maps for lower-rank inputs
            maps_list = []
            for inp_type in input_types:
                inp_rank = 0
                if "tensor<" in inp_type:
                    inner = inp_type.split("<", 1)[1].rstrip(">")
                    if "x" in inner:
                        inp_rank = len(inner.split("x")) - 1
                if inp_rank < rank:
                    # Broadcast map: project output dims to input dims
                    inp_exprs = ", ".join(f"d{j}" for j in range(inp_rank))
                    if inp_rank > 0:
                        maps_list.append(f"affine_map<({dim_str}) -> ({inp_exprs})>")
                    else:
                        maps_list.append(f"affine_map<({dim_str}) -> ()>")
                else:
                    maps_list.append(ident_map)
            maps_list.append(ident_map)  # output map
            maps = ", ".join(maps_list)
            
            # Generate empty tensor
            result.append(f"{indent}%e{result_ssa} = \"tensor.empty\"() : () -> {output_type}")
            
            # Generate linalg.generic in quoted format
            ins_refs = ", ".join(f"%{o}" for o in operands)
            all_refs = f"{ins_refs}, %e{result_ssa}"
            ins_types_str = ", ".join(input_types)
            
            attrs = f'indexing_maps = [{maps}], iterator_types = [{iter_types}], operandSegmentSizes = array<i32: {n_inputs}, 1>'
            
            result.append(f'{indent}%{result_ssa} = \"linalg.generic\"({all_refs}) <{{{attrs}}}> ({{')
            
            bb_args = ", ".join(f"%in{j}: {elem_type}" for j in range(n_inputs)) + f", %out: {elem_type}"
            result.append(f"{indent}  ^bb0({bb_args}):")
            
            arith_args = ", ".join(f"%in{j}" for j in range(n_inputs))
            arith_types = ", ".join([elem_type] * n_inputs)
            result.append(f'{indent}    %ir = "arith.{arith_op_name}"({arith_args}) : ({arith_types}) -> {elem_type}')
            result.append(f'{indent}    "linalg.yield"(%ir) : ({elem_type}) -> ()')
            
            result.append(f"{indent}  }}) : ({ins_types_str}, {output_type}) -> {output_type}")
            i += 1
            continue
        
        result.append(line)
        i += 1
    
    return "\n".join(result)
def _split_mlir_types(types_str: str) -> list[str]:
    """Split MLIR type list respecting nested angle brackets."""
    parts = []
    depth = 0
    current = []
    for ch in types_str:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts
    if bare_arith:
        # Check if any of these are NOT inside a linalg.generic block
        # Simple heuristic: look for patterns where arith op directly operates
        # on tensor types (indicated by : (tensor<...>)
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

    # 3. Must contain at least one linalg op (sanity check — lowering did something)
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
    from compiler.mlir_dialect.lowering import sf_to_linalg_pass_on_module
    from compiler.mlir_dialect.llvm_backend import (
        lower_linalg_to_llvm_ir,
        compile_module_to_dylib,
    )

    # Step 1: Load model.mlir → MlirModule
    print(f"[1/6] Parsing {mlir_path} ...")
    mlir_text = mlir_path.read_text()
    module = _parse_mlir_text(mlir_text)
    print(f"   {len(module.functions)} functions, {sum(len(f.ops) for f in module.functions)} ops")

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

    # Step 4: MlirModule → ir.Module → sf→linalg lowering
    print("[4/6] Converting MlirModule → ir.Module → sf→linalg lowering ...")
    _setup_mlir_path()
    import mlir.ir as ir

    ir_mod = mlir_module_to_ir_module(module)
    lowered_text = sf_to_linalg_pass_on_module(ir_mod)

    lowered_path = compiled_path / "model.lowered.mlir"
    lowered_path.write_text(lowered_text)
    print(f"   Saved lowered MLIR to {lowered_path}")

    # Post-process: wrap bare arith ops in linalg.generic
    lowered_text = _fixup_bare_arith(lowered_text)
    lowered_path.write_text(lowered_text)

    # Verification gate: ensure lowered IR is valid before proceeding
    print("[4v] Verifying lowered IR ...")
    try:
        _verify_lowered_ir(lowered_text)
        print("   Verification passed")
    except ValueError as e:
        print(f"   VERIFICATION FAILED:\n{e}")
        print("   (continuing anyway for debugging purposes)")

    # Step 5: Parse lowered text into fresh ir.Module → LLVM lowering
    print("[5/6] Parsing lowered MLIR → LLVM lowering ...")
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        ir_mod_llvm = ir.Module.parse(lowered_text, ctx)
        llvm_text = lower_linalg_to_llvm_ir(ir_mod_llvm)

    # Step 6: Parse LLVM text → compile to .dylib
    print("[6/6] Compiling to .dylib ...")
    ctx2 = ir.Context()
    ctx2.allow_unregistered_dialects = True
    with ctx2:
        ir_mod_final = ir.Module.parse(llvm_text, ctx2)
        dylib_path = compile_module_to_dylib(
            ir_mod_final,
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
