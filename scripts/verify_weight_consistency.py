#!/usr/bin/env python3
"""Weight consistency verification utilities.

Provides functions to verify weight promotion ordering and parameter
binding consistency between compute graph and lowered MLIR.
"""

import os
import re
import sys
from typing import Any


def parse_compute_graph_outputs(compiled_dir: str) -> list[dict]:
    """Parse the compute graph and return per-function output counts."""
    # Import from sibling script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.ctypes_forward import parse_sfcf_blob, parse_compute_graph  # noqa: I001

    bin_path = os.path.join(compiled_dir, "constants.bin")
    if not os.path.exists(bin_path):
        return []

    with open(bin_path, "rb") as f:
        blob = f.read()

    try:
        _, _, graph_pos, sfcf_version = parse_sfcf_blob(blob)
        graph, _ = parse_compute_graph(blob, graph_pos, version=sfcf_version)

        results = []
        for fi, func in enumerate(graph["functions"]):
            sym = re.sub(r'^_*mlir_ciface_', '', func["symbol"])
            # Count input binding types
            weight_inputs = 0
            ssa_inputs = 0
            global_inputs = 0
            weight_names: list[str] = []
            for inp in func["inputs"]:
                bt = inp["binding"][0]
                if bt == "weight":
                    weight_inputs += 1
                    weight_names.append(inp["binding"][1])  # the weight key name
                elif bt == "ssa":
                    ssa_inputs += 1
                elif bt == "global_input":
                    global_inputs += 1
            results.append({
                "index": fi,
                "symbol": sym,
                "num_inputs": func["num_inputs"],
                "num_outputs": func["num_outputs"],
                "weight_inputs": weight_inputs,
                "ssa_inputs": ssa_inputs,
                "global_inputs": global_inputs,
                "weight_names": weight_names,
                "inputs": func["inputs"],  # raw input bindings for SSA/topological checks
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def parse_lowered_ir_outputs(lowered_path: str) -> list[dict]:
    """Parse the lowered MLIR to count actual function outputs."""
    if not os.path.exists(lowered_path):
        return []

    with open(lowered_path) as f:
        content = f.read()

    # Extract function names in order
    sym_names = re.findall(r'sym_name = "([^"]+)"', content)

    # Extract return operand counts in order
    returns = re.finditer(r'"func\.return"\(([^)]*)\)', content)
    ret_counts = []
    for ret in returns:
        operands_str = ret.group(1)
        operands = [o.strip() for o in operands_str.split(",") if o.strip()]
        ret_counts.append(len(operands))

    # Extract declared output counts from function_type
    # Format: (args) -> (out1, out2)  or  (args) -> single_out
    ft_matches = re.findall(
        r'function_type = \([^)]*\) -> (?:\(([^)]*)\)|(\S+))',
        content, re.DOTALL
    )
    declared_counts = []
    for multi, _single in ft_matches:
        if multi:
            n = len(re.findall(r'tensor<[^>]+>', f"({multi})"))
        else:
            n = 1  # single output
        declared_counts.append(n)

    results = []
    for i, name in enumerate(sym_names):
        declared = declared_counts[i] if i < len(declared_counts) else 0
        actual = ret_counts[i] if i < len(ret_counts) else 0
        results.append({
            "symbol": name,
            "declared_outputs": declared,
            "actual_return_values": actual,
        })
    return results


def parse_llvm_ciface_params(ll_path: str) -> dict[str, int]:
    """Parse LLVM IR ciface function signatures and count ptr parameters."""
    if not os.path.exists(ll_path):
        return {}

    with open(ll_path) as f:
        content = f.read()

    results: dict[str, int] = {}
    pattern = re.compile(
        r'define void @_mlir_ciface_(\w+)\(([^)]*)\)',
    )
    for m in pattern.finditer(content):
        name = m.group(1)
        params = m.group(2)
        ptr_count = params.count('ptr ')
        results[name] = ptr_count
    return results


def parse_lowered_ir_function_inputs(lowered_path: str) -> dict[str, int]:
    """Parse lowered MLIR to count input tensors per function."""
    if not os.path.exists(lowered_path):
        return {}

    with open(lowered_path) as f:
        content = f.read()

    results: dict[str, int] = {}
    pattern = re.compile(
        r'function_type = \(([^)]*)\) -> (?:\(([^)]*)\)|(\S+)),[^}]*?sym_name = "([^"]+)"',
        re.DOTALL,
    )
    for m in pattern.finditer(content):
        name = m.group(4)
        input_types = m.group(1)
        n = len(re.findall(r'tensor<[^>]+>', input_types))
        results[name] = n
    return results


def _parse_tensor_shape(typ: str) -> tuple[int, ...]:
    """Parse a tensor type string into shape tuple. Dynamic dims ('?') → 0."""
    # typ is like "tensor<50272x768xf32>" or "tensor<f32>"
    inner = typ[7:-1]  # strip "tensor<" and ">"
    # Split off dtype (last segment after 'x')
    parts = inner.rsplit('x', 1)
    dims_str = parts[0]
    if not dims_str or dims_str == parts[-1]:
        # Scalar: no 'x' separator
        return ()
    dims = []
    for d in dims_str.split('x'):
        d = d.strip()
        if d == '?':
            dims.append(0)  # dynamic
        else:
            dims.append(int(d))
    return tuple(dims)


def parse_lowered_weight_names(lowered_path: str) -> dict[str, list[str]]:
    """Parse sf.weight_names attributes from lowered MLIR func ops.

    The C++ pass attaches sf.weight_names = ["name1", "name2", ...] on each
    func::FuncOp, listing the original weight names for promoted block args
    in declaration order. The MLIR is in generic op form:
      "func.func"() <{sym_name = "main_0", ...}> ({...}) {sf.weight_names = [...]}

    Returns: {"main_0": ["model_decoder_embed_tokens_weight", ...], ...}
    """
    results: dict[str, list[str]] = {}
    if not os.path.exists(lowered_path):
        return results

    with open(lowered_path) as f:
        content = f.read()

    # Collect all function names with their positions (in file order)
    sym_name_positions = list(
        re.finditer(r'sym_name\s*=\s*"([^"]+)"', content)
    )

    # Find each sf.weight_names and associate with the preceding sym_name
    for wm in re.finditer(r'sf\.weight_names\s*=\s*\[([^\]]*)\]', content):
        wpos = wm.start()
        func_name = None
        for sm in reversed(sym_name_positions):
            if sm.start() < wpos:
                func_name = sm.group(1)
                break
        if func_name:
            names_str = wm.group(1)
            names = re.findall(r'"([^"]+)"', names_str)
            results[func_name] = names

    return results


def _parse_weight_names_from_text(text: str) -> dict[str, list[str]]:
    """Parse sf.weight_names from lowered MLIR text string.

    Same logic as parse_lowered_weight_names but operates on a string
    instead of a file path. Used by verify_weight_promotion_order when
    the lowered IR text is already in memory.

    Returns: {"main_0": ["model_decoder_embed_tokens_weight", ...], ...}
    """
    results: dict[str, list[str]] = {}
    sym_name_positions = list(
        re.finditer(r'sym_name\s*=\s*"([^"]+)"', text)
    )
    for wm in re.finditer(r'sf\.weight_names\s*=\s*\[([^\]]*)\]', text):
        wpos = wm.start()
        func_name = None
        for sm in reversed(sym_name_positions):
            if sm.start() < wpos:
                func_name = sm.group(1)
                break
        if func_name:
            names_str = wm.group(1)
            names = re.findall(r'"([^"]+)"', names_str)
            results[func_name] = names
    return results


def verify_weight_promotion_order(module: Any, lowered_text: str) -> list[str]:
    """Check weight promotion preserves original op order.

    For each function in *module*, extract weight ops (op_name == "weight")
    in declaration order, filter out constant weights (name starts with
    ``_const_``), then compare against ``sf.weight_names`` in the lowered IR.

    Any name mismatch or ordering difference is reported as an error.

    Returns a list of error messages (empty = all functions pass).
    """
    errors: list[str] = []
    ir_weight_names = _parse_weight_names_from_text(lowered_text)

    for func in module.functions:
        func_name = func.name

        # Extract weight names in op declaration order (dict insertion order
        # from weights.items() in fx_to_mlir.py Phase 4).
        original_weight_names = [
            op.attributes.get("name", "")
            for op in func.ops
            if op.op_name == "weight"
            and not op.attributes.get("name", "").startswith("_const_")
        ]

        lowered_names = ir_weight_names.get(func_name, [])

        if not original_weight_names and not lowered_names:
            continue

        if len(original_weight_names) != len(lowered_names):
            errors.append(
                f"Function '{func_name}': expected {len(original_weight_names)} "
                f"weight names, found {len(lowered_names)} in lowered IR"
            )
            if original_weight_names:
                _show = original_weight_names[:10]
                errors.append(f"  Expected ({len(original_weight_names)}): {_show}")
            if lowered_names:
                _show = lowered_names[:10]
                errors.append(f"  Got ({len(lowered_names)}): {_show}")
            continue

        for i, (expected, actual) in enumerate(zip(
            original_weight_names, lowered_names, strict=True,
        )):
            if expected != actual:
                errors.append(
                    f"Function '{func_name}' arg[{i}]: expected '{expected}', "
                    f"got '{actual}' in lowered IR"
                )
    return errors


def parse_lowered_ir_input_shapes(lowered_path: str) -> dict[str, list[tuple[int, ...]]]:
    """Parse lowered MLIR and return (name -> list of tensor shapes) per function.
    Includes ALL params (including the first, which is input_ids in main_0).

    Returns: {"main_0": [(2, 4), (50272, 768), ...], ...}
    """
    if not os.path.exists(lowered_path):
        return {}

    with open(lowered_path) as f:
        content = f.read()

    results: dict[str, list[tuple[int, ...]]] = {}
    pattern = re.compile(
        r'function_type = \(([^)]*)\) -> (?:\(([^)]*)\)|(\S+)),[^}]*?sym_name = "([^"]+)"',
        re.DOTALL,
    )
    for m in pattern.finditer(content):
        name = m.group(4)
        input_types = m.group(1)
        tensors = re.findall(r'tensor<[^>]+>', input_types)

        shapes = [_parse_tensor_shape(t) for t in tensors]
        results[name] = shapes

    return results


def check_parameter_binding(cg_funcs: list[dict], lowered_path: str) -> int:
    """Compare compute graph weight+ssa count vs lowered MLIR non-input_ids param count.

    Also compares weight name strings when sf.weight_names is present.

    For each function:
      CG param = weight_inputs + ssa_inputs
      IR param = total_tensors - (1 if global_inputs > 0 else 0)

    Returns 0 if consistent, 1+ if issues found.
    """
    issues = 0

    ir_shapes = parse_lowered_ir_input_shapes(lowered_path)
    if not ir_shapes:
        print("❌ Cannot parse lowered IR input shapes")
        return 1

    ir_weight_names = parse_lowered_weight_names(lowered_path)

    print("\n🔗 Parameter binding check (CG compute graph vs lowered MLIR)")
    for cg in cg_funcs:
        if "error" in cg:
            continue

        name = f"main_{cg['index']}"
        cg_param_count = cg["weight_inputs"] + cg["ssa_inputs"]
        ir_shapes_list = ir_shapes.get(name, [])
        ir_total = len(ir_shapes_list)
        # Subtract input_ids (global_input) from IR count if present
        ir_param_count = ir_total - (1 if cg["global_inputs"] > 0 else 0)

        if cg_param_count != ir_param_count:
            print(f"  ❌ '{name}': CG expects {cg_param_count} params "
                  f"(w={cg['weight_inputs']}, ssa={cg['ssa_inputs']}), "
                  f"but lowered IR has {ir_param_count} non-input_ids params "
                  f"({ir_total} total - {cg['global_inputs']} global_input)")
            print(f"     Mismatch: {cg_param_count - ir_param_count:+d}")
            if cg["weight_names"]:
                n_show = min(8, len(cg["weight_names"]))
                print(f"     CG weight names ({len(cg['weight_names'])} total, first {n_show}):")
                for wn in cg["weight_names"][:n_show]:
                    print(f"       - {wn}")
            issues += 1
        else:
            print(f"  ✅ '{name}': {cg_param_count} params (consistent)")

        # New: compare weight name strings
        cg_w_names = cg.get("weight_names", [])
        ir_w_names = ir_weight_names.get(name, [])
        if cg_w_names and ir_w_names:
            if cg_w_names != ir_w_names:
                print(f"  ❌ '{name}': weight name mismatch")
                for i, (cg_n, ir_n) in enumerate(zip(cg_w_names, ir_w_names, strict=False)):
                    if cg_n != ir_n:
                        print(f"     arg[{i}]: CG='{cg_n}' vs IR='{ir_n}'")
                if len(cg_w_names) != len(ir_w_names):
                    print(f"     CG has {len(cg_w_names)} names, IR has {len(ir_w_names)} names")
                issues += 1
            elif cg_param_count == ir_param_count:
                # Only print name consistency when count is also consistent
                print(f"     Weight names: {len(cg_w_names)} (consistent)")

    return issues
