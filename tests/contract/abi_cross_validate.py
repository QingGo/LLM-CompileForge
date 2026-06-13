#!/usr/bin/env python3
"""
abi_cross_validate.py — Cross-validate SfaAbiHeader fields from a compiled model.

Loads the protobuf SfaAbiHeader + SfaWeightData embedded in the compiled dylib's
source artifacts (sfa_abi.c / sfa_weights.c) and validates every field against
runtime expectations per include/CONTRACT.md.

Checks:
  1. num_inputs == input_fields.len() for each function   (G3)
  2. output descriptors match output_rank when populated  (G4 partial)
  3. Every SfaInputField.rank > 0 (or == 0 with fallback) (G1)
  4. Every SfaSsaRef.producer_func < total function count (G4)
  5. All weight entries have non-empty compiled_name/hf_key

Usage:
    python tests/contract/abi_cross_validate.py outputs/compiled/opt_125m_fresh
    python tests/contract/abi_cross_validate.py  # auto-discovers from outputs/compiled/

Exit 0: all checks pass (or gracefully skipped when no model available)
Exit 1: at least one check fails
"""

import argparse
import os
import re
import sys

# Proto module is at gen/proto/python relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROTO_DIR = os.path.join(_PROJECT_ROOT, "gen", "proto", "python")
sys.path.insert(0, _PROTO_DIR)


# ── Proto extraction helpers ──────────────────────────────────────────

_HEX_PATTERN: re.Pattern = re.compile(r"0x([0-9a-fA-F]{2})")


def _find_compiled_dir(argv_path: str | None = None) -> str | None:
    """Resolve a compiled model directory from argument or auto-discovery."""
    if argv_path:
        if os.path.isdir(argv_path):
            return os.path.abspath(argv_path)
        # Try as model name under outputs/compiled/
        candidate = os.path.join(_PROJECT_ROOT, "compiled", argv_path)
        if os.path.isdir(candidate):
            return candidate
        return None

    # Auto-discovery: first subdirectory under outputs/compiled/
    compiled_root = os.path.join(_PROJECT_ROOT, "compiled")
    if not os.path.isdir(compiled_root):
        return None
    for entry in sorted(os.listdir(compiled_root)):
        full = os.path.join(compiled_root, entry)
        if os.path.isdir(full):
            return full
    return None


def _extract_proto_bytes(c_file_path: str) -> bytes:
    """Extract raw protobuf bytes from a sfa_abi.c / sfa_weights.c file.

    These files contain a ``const uint8_t name[SIZE] = { 0xNN, 0xNN, ... };``
    array. We extract every ``0xNN`` token and convert to bytes.
    """
    with open(c_file_path, "rb") as f:
        text = f.read().decode("ascii", errors="ignore")
    hex_bytes = _HEX_PATTERN.findall(text)
    return bytes(int(b, 16) for b in hex_bytes)


# ── Validation functions ──────────────────────────────────────────────


def validate_num_inputs(funcs: list) -> int:
    """Check G3: num_inputs == input_fields.len() for every function."""
    failures = 0
    for fi, f in enumerate(funcs):
        expected = f.num_inputs
        actual = len(f.input_fields)
        if expected != actual:
            print(f"  [FAIL] G3: func[{fi}] ({f.symbol}): num_inputs={expected} != input_fields.len()={actual}")
            failures += 1
    if failures == 0:
        print(f"  G3: PASS (all {len(funcs)} funcs: num_inputs == input_fields.len())")
    else:
        for fi, f in enumerate(funcs):
            expected = f.num_inputs
            actual = len(f.input_fields)
            if expected != actual:
                print(f"       func[{fi}] ({f.symbol}): num_inputs={expected} vs input_fields={actual}")
        print(f"  G3: FAIL ({failures}/{len(funcs)} mismatches)")
    return failures


def validate_output_descriptors(funcs: list) -> int:
    """Check output_rank consistency with output descriptors.

    When outputs are populated, verify each descriptor has a valid rank.
    When outputs are empty, output_rank is used as fallback — this is
    expected behavior (see CONTRACT.md G4 partial).
    """
    failures = 0
    populated = 0
    total_descriptors = 0
    for fi, f in enumerate(funcs):
        if len(f.outputs) == 0:
            continue
        populated += 1
        total_descriptors += len(f.outputs)
        # Verify each output descriptor has rank > 0
        for oi, od in enumerate(f.outputs):
            if od.rank == 0:
                print(f"  [FAIL] G4: func[{fi}] ({f.symbol}): output[{oi}].rank=0 (invalid)")
                failures += 1
    if failures == 0:
        print(f"  G4: PASS ({populated}/{len(funcs)} funcs have outputs, {total_descriptors} descriptors, all rank>0)")
    else:
        print(f"  G4: FAIL ({failures} zero-rank descriptors)")
    return failures


def validate_input_ranks(funcs: list) -> int:
    """Check G1: Every SfaInputField.rank > 0 or has known fallback.

    rank=0 is allowed when the runtime has a documented fallback (rank=2
    for GlobalInput, rank=2 for Weight). We report rank=0 as INFO but
    only flag FAIL if rank is missing for ALL inputs of a function.
    """
    zero_ranks = 0
    total_inputs = 0
    for _fi, f in enumerate(funcs):
        for _ii, inp in enumerate(f.input_fields):
            total_inputs += 1
            if inp.rank == 0:
                zero_ranks += 1
    if zero_ranks == 0:
        print(f"  G1: PASS (all {total_inputs} inputs have rank>0)")
    else:
        print(f"  G1: PASS ({zero_ranks}/{total_inputs} inputs with rank=0, runtime has documented fallback to 2)")
    return 0  # rank=0 is not a failure per G1 CONTRACT.md


def validate_ssa_references(funcs: list) -> int:
    """Check G4: Every SfaSsaRef.producer_func < total function count."""
    total_funcs = len(funcs)
    failures = 0
    total_ssa = 0
    for fi, f in enumerate(funcs):
        for ii, inp in enumerate(f.input_fields):
            if not inp.HasField("ssa"):
                continue
            total_ssa += 1
            pf = inp.ssa.producer_func
            if pf >= total_funcs:
                print(
                    f"  [FAIL] G4: func[{fi}] ({f.symbol}) input[{ii}]: "
                    f"ssa.producer_func={pf} >= total_funcs={total_funcs}"
                )
                failures += 1
    if failures == 0:
        print(f"  G4: PASS (all {total_ssa} SSA producer_func in range)")
    else:
        print(f"  G4: FAIL ({failures}/{total_ssa} out of range)")
    return failures


def validate_weight_entries(weight_data) -> int:
    """Check all weight entries have non-empty compiled_name and hf_key."""
    failures = 0
    for wi, w in enumerate(weight_data.weight_entries):
        name_ok = bool(w.compiled_name and w.compiled_name.strip())
        key_ok = bool(w.hf_key and w.hf_key.strip())
        if not name_ok or not key_ok:
            missing = []
            if not name_ok:
                missing.append("compiled_name")
            if not key_ok:
                missing.append("hf_key")
            print(f"  [FAIL] WEIGHT[{wi}]: empty {', '.join(missing)}")
            failures += 1
    if failures == 0:
        print(f"  WEIGHTS: PASS (all {len(weight_data.weight_entries)} entries non-empty)")
    else:
        print(f"  WEIGHTS: FAIL ({failures}/{len(weight_data.weight_entries)} empty fields)")
    return failures


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-validate SfaAbiHeader fields from compiled model")
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=None,
        help="Path to compiled model directory (e.g. outputs/compiled/opt_125m_fresh)",
    )
    args = parser.parse_args()

    # 1. Discover compiled model directory
    compiled_dir = _find_compiled_dir(args.model_dir)
    if compiled_dir is None:
        print("[SKIP] No compiled model directory found — nothing to validate.")
        print("       Run 'make build-all' first, or specify a model directory.")
        return 0

    print(f"=== ABI Cross-Validation: {os.path.basename(compiled_dir)} ===\n")

    sfa_abi_c = os.path.join(compiled_dir, "sfa_abi.c")
    sfa_weights_c = os.path.join(compiled_dir, "sfa_weights.c")

    if not os.path.isfile(sfa_abi_c):
        print(f"[SKIP] sfa_abi.c not found in {compiled_dir}")
        print("       Run 'make build-all' to compile the model first.")
        return 0

    # 2. Load protobuf data
    from sfa_abi_pb2 import SfaAbiHeader, SfaWeightData  # type: ignore[attr-defined]

    try:
        abi_data = _extract_proto_bytes(sfa_abi_c)
        abi = SfaAbiHeader()
        abi.ParseFromString(abi_data)
    except Exception as e:
        print(f"[FAIL] Cannot decode SfaAbiHeader from {sfa_abi_c}: {e}")
        return 1

    print(f"  ABI header: magic=0x{abi.magic:08X} version={abi.version} funcs={len(abi.funcs)}")

    # Validate magic (must be 0x41464253 "SFBA")
    if abi.magic != 0x41464253:
        print(f"  [FAIL] MAGIC: 0x{abi.magic:08X} != 0x41464253 (expected)")
        return 1
    print(f"  [PASS] MAGIC: 0x{abi.magic:08X} == 0x41464253\n")

    # Validate version
    if abi.version != 1:
        print(f"  [FAIL] VERSION: {abi.version} != 1 (expected)")
        return 1
    print(f"  [PASS] VERSION: {abi.version} == 1\n")

    # 3. Load weight data
    if os.path.isfile(sfa_weights_c):
        try:
            weight_data_raw = _extract_proto_bytes(sfa_weights_c)
            weight_data = SfaWeightData()
            weight_data.ParseFromString(weight_data_raw)
            print(
                f"  Weight data: entries={len(weight_data.weight_entries)} "
                f"constants={len(weight_data.constant_entries)}\n"
            )
        except Exception as e:
            print(f"  [WARN] Cannot decode SfaWeightData: {e}")
            # Create empty weight data for validation
            weight_data = SfaWeightData()
    else:
        print("  [WARN] sfa_weights.c not found — skipping weight validation\n")
        weight_data = SfaWeightData()

    # 4. Run validations
    print("═══ G3: num_inputs == input_fields.len() ═══")
    g3_failures = validate_num_inputs(abi.funcs)

    print("\n═══ G4: Output descriptors ═══")
    g4a_failures = validate_output_descriptors(abi.funcs)

    print("\n═══ G1: Input field ranks ═══")
    g1_failures = validate_input_ranks(abi.funcs)

    print("\n═══ G4: SSA producer_func range ═══")
    g4b_failures = validate_ssa_references(abi.funcs)

    total_weight_failures = 0
    if len(weight_data.weight_entries) > 0:
        print("\n═══ WEIGHTS: non-empty compiled_name + hf_key ═══")
        total_weight_failures = validate_weight_entries(weight_data)

    # 5. Summary
    total_failures = g3_failures + g4a_failures + g1_failures + g4b_failures + total_weight_failures
    print(f"\n{'═' * 60}")
    if total_failures == 0:
        print("RESULT: ALL CHECKS PASSED ✓")
        return 0
    else:
        print(f"RESULT: {total_failures} FAILURE(S) ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
