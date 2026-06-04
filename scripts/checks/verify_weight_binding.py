#!/usr/bin/env python3
import os
import re
import sys
from pathlib import Path


def parse_mlir_weight_names(lowered_path: str) -> dict[str, list[str]]:
    """Parse sf.weight_names from lowered MLIR's func.func attributes.

    MLIR uses custom syntax:
      func.func @main_0(...) attributes {sf.weight_names = [...]} {

    Returns: {"main_0": ["name1", ...], ...}
    """
    results: dict[str, list[str]] = {}
    if not os.path.exists(lowered_path):
        return results

    with open(lowered_path) as f:
        content = f.read()

    func_positions = list(
        re.finditer(r'func\.func\s+@(\w+)\s*\(', content)
    )
    for wm in re.finditer(r'sf\.weight_names\s*=\s*\[([^\]]*)\]', content):
        wpos = wm.start()
        func_name = None
        for fm in reversed(func_positions):
            if fm.start() < wpos:
                func_name = fm.group(1)
                break
        if func_name:
            names_str = wm.group(1)
            names = re.findall(r'"([^"]+)"', names_str)
            results[func_name] = names

    return results


def fmt_abbrev(names: list[str], max_show: int = 5) -> str:
    if not names:
        return "(none)"
    if len(names) <= max_show:
        return ", ".join(names)
    head = ", ".join(names[:max_show])
    return f"[{len(names)} names] {head}, ..."


def _mismatch_detail(mlir: list[str], cg: list[str]) -> list[str]:
    lines = []
    if len(mlir) != len(cg):
        lines.append(f"Count: MLIR has {len(mlir)}, SFCF has {len(cg)}")
    else:
        for idx, (m, c) in enumerate(zip(mlir, cg)):
            if m != c:
                lines.append(f"  arg[{idx}]: MLIR='{m}' vs SFCF='{c}'")
    if not lines:
        lines.append("Names differ (possibly order)")
        lines.append(f"  MLIR: {mlir}")
        lines.append(f"  SFCF: {cg}")
    return lines


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "compiled/opt_125m_fresh"
    model_dir = os.path.abspath(model_dir)

    lowered_path = os.path.join(model_dir, "model.lowered.mlir")
    constants_path = os.path.join(model_dir, "constants.bin")

    missing = []
    if not os.path.isdir(model_dir):
        missing.append(f"model_dir '{model_dir}'")
    if not os.path.exists(lowered_path):
        missing.append(f"model.lowered.mlir (expected at {lowered_path})")
    if not os.path.exists(constants_path):
        missing.append(f"constants.bin (expected at {constants_path})")
    if missing:
        print(f"Missing: {', '.join(missing)}")
        return 1

    mlir_names = parse_mlir_weight_names(lowered_path)
    if not mlir_names:
        print("No sf.weight_names found in lowered MLIR")
    else:
        print(f"MLIR: {len(mlir_names)} function(s) with weight names")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scripts.checks.verify_weight_consistency import parse_compute_graph_outputs

    cg_funcs = parse_compute_graph_outputs(model_dir)
    if not cg_funcs:
        print("Failed to parse compute graph from constants.bin")
        return 1

    cg_names: dict[str, list[str]] = {}
    for cg in cg_funcs:
        if "error" not in cg:
            cg_names[cg["symbol"]] = cg.get("weight_names", [])

    all_funcs = sorted(set(list(mlir_names.keys()) + list(cg_names.keys())),
                       key=lambda x: int(x.split("_")[-1]) if "_" in x else 0)

    header = f"{'Func':<14} {'MLIR weight names':<50} {'SFCF weight names':<50} Match?"
    sep = "-" * len(header)
    print(f"\n{'Weight Binding Verification':^60}")
    print(sep)
    print(header)
    print(sep)

    total_mismatches = 0
    for func_name in all_funcs:
        mlir_wn = mlir_names.get(func_name, [])
        cg_wn = cg_names.get(func_name, [])

        mlir_stripped = [s.strip() for s in mlir_wn]
        cg_stripped = [w.strip() for w in cg_wn]

        if mlir_stripped == cg_stripped:
            match_icon = "OK"
        else:
            match_icon = "MISMATCH"
            total_mismatches += 1

        mlir_fmt = fmt_abbrev(mlir_wn)
        cg_fmt = fmt_abbrev(cg_wn)

        print(f"{func_name:<14} {mlir_fmt:<50} {cg_fmt:<50} {match_icon}")

        if mlir_stripped != cg_stripped:
            detail_lines = _mismatch_detail(mlir_stripped, cg_stripped)
            for line in detail_lines:
                print(f"  {line}")

    print(sep)

    if total_mismatches == 0:
        print(f"\nPASS: All {len(all_funcs)} function(s) have matching weight bindings")
        return 0
    else:
        print(f"\nFAIL: {total_mismatches} function(s) have weight binding mismatches")
        return 1


if __name__ == "__main__":
    sys.exit(main())
