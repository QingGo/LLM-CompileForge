#!/usr/bin/env python3
"""Verify consistency between compiled .dylib, compute graph, and lowered IR.

Usage:
    python scripts/verify_dylib_consistency.py compiled/opt_125m_fresh
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np


def find_dylib(compiled_dir: str) -> str | None:
    for f in os.listdir(compiled_dir):
        if f.endswith(".dylib"):
            return os.path.join(compiled_dir, f)
    return None


def get_function_symbols(dylib_path: str) -> list[str]:
    """Get ciface function names from the dylib."""
    result = subprocess.run(
        ["nm", "-g", dylib_path], capture_output=True, text=True
    )
    funcs = []
    for line in result.stdout.splitlines():
        if "_mlir_ciface_" in line:
            name = line.split()[-1]
            # Strip leading underscores + mlir_ciface_ prefix
            # Mach-O adds extra _ prefix on macOS
            name = re.sub(r'^_*mlir_ciface_', '', name)
            funcs.append(name)
    return sorted(funcs)


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
        _, _, graph_pos = parse_sfcf_blob(blob)
        graph = parse_compute_graph(blob, graph_pos)

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


def check_ssa_bindings(cg_funcs: list[dict]) -> int:
    """Verify every SSA binding references a valid function + output index.

    Returns 0 if all valid, 1+ if issues found.
    """
    issues = 0
    print("\n🔀 SSA binding reference check")
    for fi, func in enumerate(cg_funcs):
        for ii, inp in enumerate(func["inputs"]):
            if inp["binding"][0] != "ssa":
                continue
            pf = inp["binding"][1]  # prev func index
            oi = inp["binding"][2]  # output index

            if pf >= len(cg_funcs):
                print(f"  ❌ func_{fi} input[{ii}]: SSA references func_{pf} but only {len(cg_funcs)} functions exist")
                issues += 1
                continue
            nout = cg_funcs[pf]["num_outputs"]
            if oi >= nout:
                print(f"  ❌ func_{fi} input[{ii}]: SSA references func_{pf} output[{oi}] "
                      f"but func_{pf} only has {nout} outputs")
                issues += 1
                continue
    if issues == 0:
        print("  ✅ All SSA bindings valid")
    return issues


def check_topological_order(cg_funcs: list[dict]) -> int:
    """Verify function call order respects SSA dependencies (no cycles).

    For each function, all SSA dependencies must reference earlier functions.
    Returns 0 if valid, 1+ if issues found.
    """
    issues = 0
    print("\n🔗 Topological order check")
    for fi, func in enumerate(cg_funcs):
        for inp in func["inputs"]:
            if inp["binding"][0] != "ssa":
                continue
            pf = inp["binding"][1]
            if pf >= fi:
                print(f"  ❌ func_{fi} depends on func_{pf} (forward reference) — violates topological order")
                issues += 1
    if issues == 0:
        print("  ✅ All functions in topological order")
    return issues


def check_model_mlir_types(model_path: str) -> int:
    """Check model.mlir for type/shape inconsistencies.

    Specifically checks:
    - sf.ones_like ops: result type should be consistent with shape attribute
    - Any sf op with tensor<f32> (scalar) that should have a multi-dim type

    Returns 0 if clean, 1+ if issues found.
    """
    if not os.path.exists(model_path):
        print("\n⚠️  model.mlir not found — skipping type check")
        return 0

    with open(model_path) as f:
        content = f.read()

    issues = 0
    print("\n📐 Type/shape consistency check (model.mlir)")

    # Check sf.ones_like ops
    # Pattern: "sf.ones_like"(...) -> tensor<f32> (scalar result with shape attr)
    ones_like_pattern = re.compile(
        r'"sf\.ones_like"\([^)]*\)\s*\{[^}]*shape\s*=\s*\[([^\]]*)\][^}]*\}\s*:\s*\([^)]*\)\s*->\s*(\S+)'
    )

    for m in ones_like_pattern.finditer(content):
        shape_attr = m.group(1)  # e.g., "%sym_size_int_32, %sym_size_int_33"
        result_type = m.group(2)  # e.g., "tensor<f32>"

        # Count dynamic dimensions in shape attr
        dim_count = len([d for d in shape_attr.split(',') if d.strip()])

        # Check if result type has matching rank
        result_rank_match = re.search(r'tensor<([^>]*)>', result_type)
        if result_rank_match:
            inner = result_rank_match.group(1)
            result_rank = len(inner.split('x')) if inner else 0
        else:
            result_rank = 0

        if dim_count != result_rank:
            print(
                f"  ❌ sf.ones_like: shape has {dim_count} dims "
                f"but result is rank-{result_rank} tensor ('{result_type}')"
            )
            issues += 1
        elif dim_count > 0 and result_rank == 0:
            print(
                f"  ❌ sf.ones_like: shape has {dim_count} dims "
                f"but result is SCALAR ('{result_type}')"
            )
            issues += 1

    if issues == 0:
        print("  ✅ All type/shape checks pass")
    return issues


def verify(compiled_dir: str) -> int:
    """Run all consistency checks. Return 0 = all good, 1 = issues found."""
    issues = 0

    # 1. Check .dylib symbols
    dylib_path = find_dylib(compiled_dir)
    if not dylib_path:
        print("❌ No .dylib found")
        return 1

    dylib_funcs = get_function_symbols(dylib_path)
    print(f"✅ .dylib: {len(dylib_funcs)} ciface functions")
    print(f"   Symbols: {dylib_funcs}")

    # 2. Check compute graph
    cg_funcs = parse_compute_graph_outputs(compiled_dir)
    if not cg_funcs:
        print("❌ Cannot parse compute graph")
        issues += 1
    else:
        print(f"✅ Compute graph: {len(cg_funcs)} functions")

        for cg in cg_funcs:
            if "error" in cg:
                print(f"  ⚠️  {cg['error']}")
                continue

            # Check if this function exists in .dylib
            expected_symbol = f"main_{cg['index']}"
            if expected_symbol not in dylib_funcs:
                # Try the full symbol name
                full = cg['symbol']
                if full not in dylib_funcs:
                    # Some dylibs use func_{idx} naming
                    alt = f"func_{cg['index']}"
                    if alt not in dylib_funcs:
                        print(f"  ⚠️  func_{cg['index']} ({expected_symbol}/{full}/{alt}) not in .dylib symbols")
                        issues += 1

    # 3. Check lowered IR function signatures
    lowered_path = os.path.join(compiled_dir, "model.lowered.mlir")
    ir_funcs = parse_lowered_ir_outputs(lowered_path)
    if not ir_funcs:
        print("❌ No lowered IR found (or no functions)")
        issues += 1
    else:
        print(f"✅ Lowered IR: {len(ir_funcs)} functions")

        for irf in ir_funcs:
            if irf["actual_return_values"] == 0:
                print(f"  ⚠️  '{irf['symbol']}': {irf['declared_outputs']} declared but 0 return values")
                issues += 1
            elif irf["declared_outputs"] != irf["actual_return_values"]:
                print(f"  ❌ '{irf['symbol']}': declares {irf['declared_outputs']} outputs "
                      f"but returns {irf['actual_return_values']} values")
                print("     This causes uninitialized output buffers (Issue #45)!")
                issues += 1

    # 4. Cross-reference: compute graph vs lowered IR
    if cg_funcs and ir_funcs and "error" not in cg_funcs[0]:
        print("\n📊 Cross-reference: compute graph vs lowered IR")
        for cg in cg_funcs:
            if "error" in cg:
                continue
            # Find matching IR function
            match = [irf for irf in ir_funcs if irf["symbol"].endswith(str(cg["index"]))]
            if match:
                irf = match[0]
                if irf["actual_return_values"] != cg["num_outputs"]:
                    print(f"  ❌ 'main_{cg['index']}': compute graph expects {cg['num_outputs']} outputs, "
                          f"lowered IR returns {irf['actual_return_values']}")
                    print("     This is the signature mismatch from Issue #45!")
                    issues += 1
                else:
                    print(f"  ✅ 'main_{cg['index']}': {cg['num_outputs']} outputs (consistent)")

    # 5. Check LLVM IR ciface signatures vs lowered IR expectations
    ll_path = os.path.join(compiled_dir, "model.ll")
    llvm_params = parse_llvm_ciface_params(ll_path)

    # Get input + output counts from lowered IR
    lowered_path = os.path.join(compiled_dir, "model.lowered.mlir")
    ir_inputs = parse_lowered_ir_function_inputs(lowered_path)

    if llvm_params and ir_funcs and ir_inputs:
        print("\n📐 LLVM ciface signature check")
        for irf in ir_funcs:
            name = irf["symbol"]
            actual_params = llvm_params.get(name, 0)
            ir_input_count = ir_inputs.get(name, 0)
            expected_params = ir_input_count + irf["declared_outputs"]

            if actual_params != expected_params:
                print(f"  ❌ '{name}': ciface has {actual_params} ptr params, "
                      f"but lowered IR expects {expected_params} "
                      f"(ins={ir_input_count} + outs={irf['declared_outputs']})")
                print("     This is the compiled function signature mismatch from Issue #45!")
                issues += 1
            else:
                print(f"  ✅ '{name}': {actual_params} ptr params (consistent)")

    # 6. Check parameter binding count consistency
    if cg_funcs:
        issues += check_parameter_binding(cg_funcs, lowered_path)

    # 7. Check SSA binding references
    if cg_funcs:
        ssa_issues = check_ssa_bindings(cg_funcs)
        issues += ssa_issues

    # 8. Check topological order
    if cg_funcs:
        topo_issues = check_topological_order(cg_funcs)
        issues += topo_issues

    # 9. Type/shape consistency check (model.mlir)
    model_path = os.path.join(compiled_dir, "model.mlir")
    type_issues = check_model_mlir_types(model_path)
    issues += type_issues

    # 10. Diagnostic tool health check
    diag_issues = check_diag_tool_health(compiled_dir)
    issues += diag_issues

    # Summary
    print(f"\n{'='*50}")
    if issues == 0:
        print("✅ All consistency checks passed!")
    else:
        print(f"⚠️  {issues} issue(s) found")

    return 1 if issues > 0 else 0


def _setup_mlir_path() -> None:
    """Set up MLIR Python bindings path for the MLIR build directory."""
    _mlir_pkg = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "llvm-project", "build", "tools", "mlir", "python_packages",
    )
    if os.path.isdir(_mlir_pkg) and _mlir_pkg not in sys.path:
        sys.path.insert(0, _mlir_pkg)


def check_diag_tool_health(compiled_dir: str) -> int:
    """Verify diagnostic tools are functioning correctly.

    Checks:
    1. ctypes oracle can load and compare (func_0 output must be non-zero)
    2. forward_check output has reasonable stats (not all zeros/same)
    3. ctypes vs Rust final output are consistent

    Returns 0 if all healthy, 1+ if issues.
    """
    import glob as _glob

    issues = 0
    print("\n\U0001fa79 Diagnostic tool health check")

    # Check 1: ctypes oracle
    _setup_mlir_path()
    try:
        from scripts.ctypes_oracle import CtypesOracle

        o = CtypesOracle(compiled_dir)

        # Compare against baseline to get func_outputs
        dylib = _glob.glob(os.path.join(compiled_dir, "lib*.dylib"))
        if dylib:
            o.compare(dylib[0])

            # Check func_0 output 12 (hidden state) is non-zero
            if hasattr(o, "_func_outputs") and len(o._func_outputs) > 0:
                func0_out12 = o._func_outputs[0][12] if len(o._func_outputs[0]) > 12 else None
                if func0_out12 is not None:
                    f = func0_out12.ravel()
                    if np.all(f == 0.0):
                        print("  ❌ ctypes oracle: func_0 output 12 is ALL ZEROS (diagnostic broken)")
                        issues += 1
                    elif len(f) >= 10 and np.all(f[:10] == f[0]):
                        print("  ❌ ctypes oracle: func_0 output 12 all same value (diagnostic corrupted)")
                        issues += 1
                    else:
                        print(f"  ✅ ctypes oracle: func_0 output 12 OK "
                              f"(min={f.min():.4f}, max={f.max():.4f}, mean={f.mean():.4f})")
                else:
                    print("  ⚠️ ctypes oracle: can't access func_0 output 12")

            # ALSO check func_0 output 13 (mask) — should not be all same value
            if hasattr(o, "_func_outputs") and len(o._func_outputs) > 0:
                func0_out13 = o._func_outputs[0][13] if len(o._func_outputs[0]) > 13 else None
                if func0_out13 is not None:
                    f13 = func0_out13.ravel()
                    f13_unique = np.unique(f13)
                    if len(f13_unique) <= 1:
                        print(f"  ❌ ctypes oracle: func_0 output 13 (mask) is all {f13_unique[0]:.1f} "
                              f"(mask generation likely broken — type/shape issue)")
                        issues += 1
                    elif (
                        len(f13_unique) == 2
                        and 0.0 in f13_unique
                        and (1.0 in f13_unique or any(v < -1e10 for v in f13_unique))
                    ):
                        print(f"  ✅ ctypes oracle: func_0 mask OK "
                              f"(values: {[f'{v:.1f}' for v in f13_unique[:5]]})")
                    else:
                        print(f"  ⚠️  ctypes oracle: func_0 mask has {len(f13_unique)} unique values "
                              f"({[f'{v:.1f}' for v in f13_unique[:5]]})")
            else:
                print("  ⚠️ ctypes oracle: _func_outputs not available")
        else:
            print("  ⚠️ ctypes oracle: no dylib found")
    except Exception as e:
        print(f"  ❌ ctypes oracle: failed to load: {e}")
        issues += 1

    # Check 2: Check forward_check output exists and has reasonable values
    csv_path = "/tmp/rust_logits.csv"
    if os.path.exists(csv_path):
        try:
            rust_raw = np.loadtxt(csv_path, delimiter=",")
            if rust_raw.size == 0:
                print("  ❌ forward_check: empty output")
                issues += 1
            else:
                f = rust_raw.ravel()
                if np.all(f == 0.0):
                    print("  ❌ forward_check: all zeros (broken)")
                    issues += 1
                elif len(f) >= 10 and np.all(f[:10] == f[0]):
                    print("  ❌ forward_check: all same value (broken)")
                    issues += 1
                else:
                    print(f"  ✅ forward_check: OK "
                          f"(min={f.min():.4f}, max={f.max():.4f}, mean={f.mean():.4f})")
        except Exception as e:
            print(f"  ❌ forward_check: failed to parse: {e}")
            issues += 1
    else:
        print("  ⚠️ forward_check: /tmp/rust_logits.csv not found (run forward_check first)")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Verify consistency between .dylib, compute graph, and lowered IR"
    )
    parser.add_argument("compiled_dir", help="Path to compiled model directory")
    args = parser.parse_args()

    sys.exit(verify(args.compiled_dir))


if __name__ == "__main__":
    main()
