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
            results.append({
                "index": fi,
                "symbol": sym,
                "num_inputs": func["num_inputs"],
                "num_outputs": func["num_outputs"],
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

    # Summary
    print(f"\n{'='*50}")
    if issues == 0:
        print("✅ All consistency checks passed!")
    else:
        print(f"⚠️  {issues} issue(s) found")

    return 1 if issues > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify consistency between .dylib, compute graph, and lowered IR"
    )
    parser.add_argument("compiled_dir", help="Path to compiled model directory")
    args = parser.parse_args()

    sys.exit(verify(args.compiled_dir))


if __name__ == "__main__":
    main()
