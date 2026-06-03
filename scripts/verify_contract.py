#!/usr/bin/env python3
"""
verify_contract.py — Automated contract obligation checker for LLM-CompileForge.

Verifies subproject contract obligations as defined in include/CONTRACT.md.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Type alias: (label, result, detail)
# result ∈ {"PASS", "FAIL", "SKIP"}
CheckResult = Tuple[str, str, str]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_SFA_MAGIC = 0x41464253
EXPECTED_SFA_VERSION = 1


def _collect_results(
    results: List[CheckResult], check_nums: set
) -> None:
    """Run checks and collect results."""
    checks = [
        (1, check1_engine_no_compiler_imports),
        (2, check2_hal_no_hardcoded_compiled_path),
        (3, check3_compiler_syspath_insert),
        (4, check4_proto_magic_version),
        (5, check5_hal_ir_schema),
        (6, check6_metadata_cache_policy),
        (7, check7_version_validation),
        (8, check8_num_inputs_validation),
        (9, check9_global_input_rank),
        (10, check10_buffer_rank_assert),
        (11, check11_op_catalog_completeness),
        (12, check12_cache_policy_proto_usage),
        (13, check13_kernel_op_trait),
        (14, check14_hal_ir_semantics),
    ]
    for num, fn in checks:
        if check_nums and num not in check_nums:
            continue
        label, passed, msg = fn()
        results.append((label, passed, msg))


# ── Check 1: engine/ must not import from compiler ────────────────────


def check1_engine_no_compiler_imports() -> Tuple[str, str, str]:
    """Verify no 'from compiler' imports in engine/ directory."""
    label = "Check 1: engine/ has no 'from compiler' imports"
    engine_dir = PROJECT_ROOT / "engine"
    if not engine_dir.is_dir():
        return label, "SKIP", "NO_ENGINE_DIR: engine/ directory not found"

    result = subprocess.run(
        ["grep", "-rn", "from compiler", str(engine_dir)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        details = "; ".join(
            l.replace(str(PROJECT_ROOT) + "/", "") for l in lines
        )
        return label, "FAIL", details
    elif result.returncode == 1:
        return label, "PASS", "no 'from compiler' imports found"
    else:
        return label, "FAIL", f"grep error: {result.stderr.strip()}"


# ── Check 2: runtime/src/hal/mod.rs must not have hardcoded compiled/ ─


def check2_hal_no_hardcoded_compiled_path() -> Tuple[str, str, str]:
    """Verify no hardcoded 'compiled/' path in hal/mod.rs."""
    label = "Check 2: runtime/src/hal/mod.rs has no hardcoded 'compiled/'"
    hal_mod = PROJECT_ROOT / "runtime" / "src" / "hal" / "mod.rs"
    if not hal_mod.is_file():
        return label, "SKIP", "NO_HAL_MOD: runtime/src/hal/mod.rs not found"

    result = subprocess.run(
        ["grep", "-n", "compiled/", str(hal_mod)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        details = "; ".join(
            f"hal/mod.rs:{l.strip()}" for l in lines
        )
        return label, "FAIL", details
    elif result.returncode == 1:
        return label, "PASS", "no hardcoded 'compiled/' path found"
    else:
        return label, "FAIL", f"grep error: {result.stderr.strip()}"


# ── Check 3: compiler/ sys.path.insert references sf-dialect ──────────


def check3_compiler_syspath_insert() -> Tuple[str, str, str]:
    """Verify sys.path.insert calls in compiler/ reference sf-dialect."""
    label = "Check 3: compiler/ sys.path.insert references sf-dialect"
    compiler_dir = PROJECT_ROOT / "compiler"
    if not compiler_dir.is_dir():
        return label, "SKIP", "NO_COMPILER_DIR: compiler/ not found"

    result = subprocess.run(
        ["grep", "-rn", "sys.path.insert", str(compiler_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return label, "PASS", "no sys.path.insert calls (may use other setup)"

    lines = result.stdout.strip().split("\n")
    # Allowed patterns in sys.path.insert calls:
    #   - literal sf-dialect or gen/proto in path
    #   - variable _gen_proto (resolves to gen/proto/python)
    #   - variable _sf_candidate (resolves to sf-dialect/.../sf)
    #   - variable _mlir_pkg (resolves to mlir_binding/mlir_package — needed by compiler)
    #   - calls in test files (legitimate test setup)
    def _is_allowed(hit: str) -> bool:
        if "sf-dialect" in hit:
            return True
        if "gen/proto" in hit or "_gen_proto" in hit:
            return True
        if "_sf_candidate" in hit:
            return True
        if "_mlir_pkg" in hit or "mlir_binding" in hit:
            return True
        if "test_" in hit or "/tests/" in hit:
            return True
        return False

    issues = []
    for line in lines:
        rel = line.replace(str(PROJECT_ROOT) + "/", "")
        if not _is_allowed(line):
            issues.append(rel)

    if issues:
        return (
            label,
            "FAIL",
            f"sys.path.insert without recognized subproject reference: {'; '.join(issues)}",
        )
    # Verify at least one insert references sf-dialect
    has_sf_dialect = any("sf-dialect" in l or "_sf_candidate" in l for l in lines)
    if not has_sf_dialect:
        return label, "FAIL", "no sys.path.insert referencing sf-dialect found"
    return label, "PASS", f"{len(lines)} sys.path.insert call(s) referencing allowed subproject paths"


# ── Check 4: proto SfaAbiHeader magic/version match sfa.h ────────────


def check4_proto_magic_version() -> Tuple[str, str, str]:
    """Verify SFA_MAGIC and SFA_VERSION in compiler/sfa_abi.py match expected values."""
    label = "Check 4: Proto SfaAbiHeader magic/version match sfa.h"
    sfa_abi_path = PROJECT_ROOT / "compiler" / "sfa_abi.py"
    if not sfa_abi_path.is_file():
        return label, "SKIP", "NO_SFA_ABI: compiler/sfa_abi.py not found"

    # Parse constants from compiler/sfa_abi.py
    magic_val = None
    version_val = None
    with open(sfa_abi_path) as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith("SFA_MAGIC"):
                try:
                    raw = line_stripped.split("=")[1].split("#")[0].strip()
                    magic_val = int(raw, 0)  # base=0 auto-detects 0x prefix
                except (ValueError, IndexError):
                    pass
            elif line_stripped.startswith("SFA_VERSION"):
                try:
                    raw = line_stripped.split("=")[1].split("#")[0].strip()
                    version_val = int(raw)
                except (ValueError, IndexError):
                    pass
            if magic_val is not None and version_val is not None:
                break

    failures = []
    if magic_val is None:
        failures.append("SFA_MAGIC not found in compiler/sfa_abi.py")
    elif magic_val != EXPECTED_SFA_MAGIC:
        failures.append(f"SFA_MAGIC={magic_val:#x}, expected {EXPECTED_SFA_MAGIC:#x}")
    if version_val is None:
        failures.append("SFA_VERSION not found in compiler/sfa_abi.py")
    elif version_val != EXPECTED_SFA_VERSION:
        failures.append(f"SFA_VERSION={version_val}, expected {EXPECTED_SFA_VERSION}")

    if failures:
        return label, "FAIL", "; ".join(failures)

    # Also verify proto import works
    gen_proto = str(PROJECT_ROOT / "gen" / "proto" / "python")
    if gen_proto not in sys.path:
        sys.path.insert(0, gen_proto)
    try:
        import sfa_abi_pb2  # type: ignore[import-untyped]
        _ = sfa_abi_pb2.SfaAbiHeader
        return label, "PASS", (
            f"SFA_MAGIC=0x{magic_val:08X}, SFA_VERSION={version_val}; "
            "proto SfaAbiHeader importable"
        )
    except ImportError as e:
        return label, "FAIL", f"Cannot import SfaAbiHeader: {e}"


# ── Check 5: hal_ir.json validates against schema ─────────────────────


def check5_hal_ir_schema() -> Tuple[str, str, str]:
    """Validate hal_ir.json against include/hal_ir.schema.json."""
    label = "Check 5: hal_ir.json validates against schema"
    schema_path = PROJECT_ROOT / "include" / "hal_ir.schema.json"
    if not schema_path.is_file():
        return label, "SKIP", "NO_SCHEMA: include/hal_ir.schema.json not found"

    # Find hal_ir.json — check compiled/ directories
    hal_ir_candidates = sorted(
        PROJECT_ROOT.glob("compiled/*/hal_ir.json")
    )
    if not hal_ir_candidates:
        return label, "SKIP", "NO_HAL_IR_SKIP: no hal_ir.json found under compiled/"

    # Use the first (or most recent) hal_ir.json
    hal_ir_path = hal_ir_candidates[-1]

    # Validate JSON parseable
    try:
        with open(hal_ir_path) as f:
            hal_ir_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return label, "FAIL", f"Cannot parse {hal_ir_path.relative_to(PROJECT_ROOT)}: {e}"

    # Validate against JSON Schema
    try:
        import jsonschema
    except ImportError:
        return label, "SKIP", "NO_JSONSCHEMA: jsonschema package not installed"

    try:
        with open(schema_path) as f:
            schema = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return label, "FAIL", f"Cannot load schema: {e}"

    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(hal_ir_data), key=lambda e: e.path)
    if errors:
        error_msgs = "; ".join(
            f"{'/'.join(str(p) for p in e.absolute_path) or '(root)'}: {e.message}"
            for e in errors[:5]
        )
        if len(errors) > 5:
            error_msgs += f" ... and {len(errors) - 5} more"
        return label, "FAIL", error_msgs

    func_count = len(hal_ir_data.get("functions", []))
    rel_path = hal_ir_path.relative_to(PROJECT_ROOT)
    return label, "PASS", f"{rel_path} validates ({func_count} functions)"


# ── Check 6: cache_policy in metadata.json parseable ──────────────────


def check6_metadata_cache_policy() -> Tuple[str, str, str]:
    """Verify cache_policy key in metadata.json is parseable."""
    label = "Check 6: cache_policy in metadata.json parseable"

    # Find metadata.json
    metadata_candidates = sorted(
        PROJECT_ROOT.glob("compiled/*/metadata.json")
    )
    if not metadata_candidates:
        return label, "SKIP", "NO_METADATA_SKIP: no metadata.json found under compiled/"

    metadata_path = metadata_candidates[-1]
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return label, "FAIL", f"Cannot parse {metadata_path.relative_to(PROJECT_ROOT)}: {e}"

    if "cache_policy" not in metadata:
        return label, "PASS", "no cache_policy key (not required for all models)"

    cache_policy = metadata["cache_policy"]
    # Must be a dict (object) to be considered parseable
    if isinstance(cache_policy, dict):
        rel_path = metadata_path.relative_to(PROJECT_ROOT)
        required_keys = {"max_blocks", "block_tokens", "kv_dtype"}
        missing = required_keys - set(cache_policy.keys())
        if missing:
            return label, "FAIL", (
                f"{rel_path}: cache_policy missing keys: {', '.join(sorted(missing))}"
            )
        return label, "PASS", f"{rel_path}: cache_policy valid with keys {sorted(cache_policy.keys())}"
    else:
        return label, "FAIL", (
            f"{metadata_path.relative_to(PROJECT_ROOT)}: "
            f"cache_policy is {type(cache_policy).__name__}, expected dict"
        )


# ── Check 7: runtime/src/abi.rs must validate SFA ABI version ──────────


def check7_version_validation() -> Tuple[str, str, str]:
    """Verify runtime/src/abi.rs validates SFA AbiHeader.version against SFA_VERSION."""
    label = "Check 7 (G4): runtime/src/abi.rs validates SFA ABI version"
    abi_rs = PROJECT_ROOT / "runtime" / "src" / "abi.rs"
    if not abi_rs.is_file():
        return label, "SKIP", "NO_ABI_RS: runtime/src/abi.rs not found"

    # Verify SFA_VERSION constant exists
    result_const = subprocess.run(
        ["grep", "-n", "pub const SFA_VERSION:", str(abi_rs)],
        capture_output=True, text=True,
    )
    if result_const.returncode != 0:
        return label, "FAIL", "SFA_VERSION constant not defined in abi.rs"

    # Verify version comparison exists in load_sfa_abi()
    result_cmp = subprocess.run(
        ["grep", "-n", r"version.*SFA_VERSION", str(abi_rs)],
        capture_output=True, text=True,
    )
    if result_cmp.returncode != 0:
        return label, "FAIL", "no version comparison against SFA_VERSION found in abi.rs"

    const_line = result_const.stdout.strip().split("\n")[0].strip()
    cmp_lines = "; ".join(
        l.strip() for l in result_cmp.stdout.strip().split("\n")
    )
    return label, "PASS", (
        f"SFA_VERSION defined (abi.rs:{const_line}); "
        f"version comparison(s) found (abi.rs:{cmp_lines})"
    )


# ── Check 8: runtime/src/abi.rs must validate num_inputs ──────────────


def check8_num_inputs_validation() -> Tuple[str, str, str]:
    """Verify runtime/src/abi.rs validates num_inputs == inputs.len()."""
    label = "Check 8 (G3): runtime/src/abi.rs validates num_inputs == inputs.len()"
    abi_rs = PROJECT_ROOT / "runtime" / "src" / "abi.rs"
    if not abi_rs.is_file():
        return label, "SKIP", "NO_ABI_RS: runtime/src/abi.rs not found"

    # Search for any ensure!/bail! in abi.rs (may span multiple lines;
    # the num_inputs check uses anyhow::ensure! across 3 lines)
    result_ensure = subprocess.run(
        ["grep", "-n", "ensure!", str(abi_rs)],
        capture_output=True, text=True,
    )
    if result_ensure.returncode != 0:
        return label, "FAIL", "no ensure!() found in abi.rs"

    # Verify the ensure! context mentions num_inputs or inputs.len()
    result_context = subprocess.run(
        [
            "grep", "-nE",
            r"inputs\.len\(\)|num_inputs",
            str(abi_rs),
        ],
        capture_output=True, text=True,
    )

    # Find an ensure! line that is near a num_inputs/inputs.len() line
    ensure_lines = [
        int(l.split(":")[0])
        for l in result_ensure.stdout.strip().split("\n")
        if l.strip()
    ]
    ctx_lines = [
        int(l.split(":")[0])
        for l in result_context.stdout.strip().split("\n")
        if l.strip()
    ]

    # Check if any ensure! is within 5 lines of a num_inputs/inputs.len() reference
    for eline in ensure_lines:
        for cline in ctx_lines:
            if abs(eline - cline) <= 5:
                return label, "PASS", (
                    f"ensure! at line {eline} validates num_inputs "
                    f"(context line {cline})"
                )

    return label, "FAIL", (
        "ensure! found but not near num_inputs/inputs.len() context; "
        f"ensure! lines: {ensure_lines}, ctx lines: {ctx_lines}"
    )


# ── Check 9: SfaInputGlobal must use field.rank (not hardcoded) ──────


def check9_global_input_rank() -> Tuple[str, str, str]:
    """Verify SfaInputGlobal match arm uses field.rank, not hardcoded rank: 2."""
    label = "Check 9 (G1): SfaInputGlobal uses field.rank (not hardcoded)"
    abi_rs = PROJECT_ROOT / "runtime" / "src" / "abi.rs"
    if not abi_rs.is_file():
        return label, "SKIP", "NO_ABI_RS: runtime/src/abi.rs not found"

    with open(abi_rs) as f:
        content_lines = f.readlines()

    # Scan ALL SfaInputGlobal occurrences; one is InputBinding (no field.rank
    # nearby), the other is IOTensorDef construction (should use field.rank)
    for i, line in enumerate(content_lines):
        if "SfaInputGlobal" not in line:
            continue
        # Check next 12 lines for field.rank usage
        for j in range(i, min(i + 12, len(content_lines))):
            if "field.rank" in content_lines[j]:
                return label, "PASS", (
                    f"SfaInputGlobal at line {i + 1} "
                    f"uses field.rank (line {j + 1})"
                )

    return label, "FAIL", (
        "no SfaInputGlobal match arm found that uses field.rank — "
        "rank may be hardcoded"
    )


# ── Check 10: compute_graph_runner.rs must assert buffer rank ─────────


def check10_buffer_rank_assert() -> Tuple[str, str, str]:
    """Verify compute_graph_runner.rs has debug_assert! checking buffer rank vs io_def.rank."""
    label = "Check 10 (G2): compute_graph_runner.rs asserts buffer rank matches io_def.rank"
    runner_rs = PROJECT_ROOT / "runtime" / "src" / "compute_graph_runner.rs"
    if not runner_rs.is_file():
        return label, "SKIP", "NO_RUNNER_RS: runtime/src/compute_graph_runner.rs not found"

    result = subprocess.run(
        ["grep", "-n", r"debug_assert.*rank", str(runner_rs)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        details = "; ".join(
            f"compute_graph_runner.rs:{l.strip()}" for l in lines
        )
        return label, "PASS", f"debug_assert with rank found: {details}"
    elif result.returncode == 1:
        return label, "FAIL", "no debug_assert! checking buffer rank vs io_def.rank found"
    else:
        return label, "FAIL", f"grep error: {result.stderr.strip()}"


# ── Check 11: SfaOpCatalog must have ≥30 ops ─────────────────────────


def check11_op_catalog_completeness() -> Tuple[str, str, str]:
    """Verify SfaOpCatalog has at least 30 registered HAL operators."""
    label = "Check 11: SfaOpCatalog ≥30 SfaOpDef entries"
    gen_proto = str(PROJECT_ROOT / "gen" / "proto" / "python")
    compiler_dir = str(PROJECT_ROOT / "compiler")

    if gen_proto not in sys.path:
        sys.path.insert(0, gen_proto)
    if compiler_dir not in sys.path:
        sys.path.insert(0, compiler_dir)

    try:
        from mlir_dialect.op_catalog import build_op_catalog  # type: ignore[import-untyped]
    except ImportError as e:
        return label, "SKIP", f"NO_OP_CATALOG: cannot import build_op_catalog: {e}"

    try:
        catalog = build_op_catalog()
    except Exception as e:
        return label, "FAIL", f"Cannot build SfaOpCatalog: {e}"

    op_count = len(catalog.ops)
    if op_count >= 30:
        return label, "PASS", f"{op_count} ops registered in SfaOpCatalog"
    else:
        return label, "FAIL", (
            f"Only {op_count} ops registered, expected ≥30"
        )


# ── Check 12: runtime CachePolicy must use SfaCachePolicy proto ───────


def check12_cache_policy_proto_usage() -> Tuple[str, str, str]:
    """Verify runtime/src/kv_cache.rs has from_proto method using SfaCachePolicy."""
    label = "Check 12: runtime/src/kv_cache.rs has from_proto + SfaCachePolicy"
    kv_cache = PROJECT_ROOT / "runtime" / "src" / "kv_cache.rs"
    if not kv_cache.is_file():
        return label, "SKIP", "NO_KV_CACHE: runtime/src/kv_cache.rs not found"

    # Check for fn from_proto
    result_fn = subprocess.run(
        ["grep", "-n", "fn from_proto", str(kv_cache)],
        capture_output=True, text=True,
    )
    if result_fn.returncode != 0:
        return label, "FAIL", "no 'fn from_proto' found in kv_cache.rs"

    # Check for SfaCachePolicy usage
    result_policy = subprocess.run(
        ["grep", "-n", "SfaCachePolicy", str(kv_cache)],
        capture_output=True, text=True,
    )
    if result_policy.returncode != 0:
        return label, "FAIL", "no 'SfaCachePolicy' found in kv_cache.rs"

    fn_lines = "; ".join(
        l.strip() for l in result_fn.stdout.strip().split("\n")
    )
    policy_lines = "; ".join(
        l.strip() for l in result_policy.stdout.strip().split("\n")
    )
    return label, "PASS", (
        f"from_proto found: kv_cache.rs:{fn_lines}; "
        f"SfaCachePolicy refs: kv_cache.rs:{policy_lines}"
    )


# ── Check 13: KernelOp trait exists with 5+ impl blocks ─────────────


def check13_kernel_op_trait() -> Tuple[str, str, str]:
    """Verify runtime/src/hal/primitives/traits.rs has pub trait KernelOp and ≥5 impl blocks."""
    label = "Check 13: KernelOp trait exists with ≥5 impl blocks"
    traits_rs = PROJECT_ROOT / "runtime" / "src" / "hal" / "primitives" / "traits.rs"
    if not traits_rs.is_file():
        return label, "SKIP", "NO_TRAITS_RS: runtime/src/hal/primitives/traits.rs not found"

    # Check for pub trait KernelOp
    result_trait = subprocess.run(
        ["grep", "-c", r"pub trait KernelOp", str(traits_rs)],
        capture_output=True, text=True,
    )
    if result_trait.returncode != 0:
        return label, "FAIL", f"grep error: {result_trait.stderr.strip()}"
    trait_count = int(result_trait.stdout.strip())

    if trait_count < 1:
        return label, "FAIL", "no 'pub trait KernelOp' found in traits.rs"

    # Check for impl KernelOp for
    result_impl = subprocess.run(
        ["grep", "-c", r"impl KernelOp for", str(traits_rs)],
        capture_output=True, text=True,
    )
    if result_impl.returncode != 0:
        return label, "FAIL", f"grep error: {result_impl.stderr.strip()}"
    impl_count = int(result_impl.stdout.strip())

    if impl_count >= 5:
        return label, "PASS", (
            f"pub trait KernelOp found ({trait_count}); "
            f"{impl_count} impl KernelOp for blocks (≥5 required)"
        )
    else:
        return label, "FAIL", (
            f"pub trait KernelOp found ({trait_count}), "
            f"but only {impl_count} impl blocks (need ≥5)"
        )


# ── Check 14: hal_ir semantics proto with 20+ entries ────────────────


def check14_hal_ir_semantics() -> Tuple[str, str, str]:
    """Verify SfaHalOpSemantics defined in proto and default_hal_op_semantics() has ≥20 entries."""
    label = "Check 14: SfaHalOpSemantics defined in proto, ≥20 entries in default impl"
    proto_path = PROJECT_ROOT / "include" / "sfa_abi.proto"
    hal_runner_mod = PROJECT_ROOT / "runtime" / "src" / "hal_runner" / "mod.rs"

    if not proto_path.is_file():
        return label, "SKIP", "NO_PROTO: include/sfa_abi.proto not found"
    if not hal_runner_mod.is_file():
        return label, "SKIP", "NO_HAL_RUNNER: runtime/src/hal_runner/mod.rs not found"

    # Check proto defines SfaHalOpSemantics message
    result_proto = subprocess.run(
        ["grep", "-c", r"message SfaHalOpSemantics", str(proto_path)],
        capture_output=True, text=True,
    )
    if result_proto.returncode != 0:
        return label, "FAIL", f"grep error on proto: {result_proto.stderr.strip()}"
    proto_matches = int(result_proto.stdout.strip())

    if proto_matches < 1:
        return label, "FAIL", "no 'message SfaHalOpSemantics' found in sfa_abi.proto"

    # Check default_hal_op_semantics() has ≥20 SfaHalOpSemanticEntry instances
    result_entries = subprocess.run(
        ["grep", "-c", "SfaHalOpSemanticEntry {", str(hal_runner_mod)],
        capture_output=True, text=True,
    )
    if result_entries.returncode != 0:
        return label, "FAIL", f"grep error on hal_runner: {result_entries.stderr.strip()}"
    entry_count = int(result_entries.stdout.strip())

    if entry_count >= 20:
        return label, "PASS", (
            f"SfaHalOpSemantics message defined in proto ({proto_matches}); "
            f"{entry_count} SfaHalOpSemanticEntry instances in default_hal_op_semantics() (≥20 required)"
        )
    else:
        return label, "FAIL", (
            f"SfaHalOpSemantics defined in proto, "
            f"but only {entry_count} entries in default_hal_op_semantics() (need ≥20)"
        )


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify LLM-CompileForge contract obligations"
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        default=False,
        help="Run all 14 checks (default)",
    )
    parser.add_argument(
        "--check",
        type=int,
        nargs="+",
        choices=range(1, 15),
        metavar="N",
        help="Run specific checks (1-14)",
    )
    args = parser.parse_args()

    # Determine which checks to run
    check_nums: set = set()
    if args.check:
        check_nums = set(args.check)
    else:
        # --check-all (default)
        check_nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

    results: List[CheckResult] = []
    _collect_results(results, check_nums)

    # Print results
    passed = 0
    failed = 0
    skipped = 0
    for label, result, detail in results:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "○"}[result]
        marker = f"[{icon} {result}]"
        print(f"{marker} {label}")
        if detail:
            print(f"       {detail}")
        if result == "PASS":
            passed += 1
        elif result == "FAIL":
            failed += 1
        elif result == "SKIP":
            skipped += 1

    # Summary
    total = passed + failed + skipped
    print()
    print(f"Summary: {passed} passed, {failed} failed, {skipped} skipped (of {total} total)")

    # Exit code
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
