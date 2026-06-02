#!/usr/bin/env bash
# memref_3way.sh — 3-way MemRef layout contract test orchestrator
#
# Runs sizeof/offsetof assertions in C, Rust, and Python; compares
# all three outputs; exits 0 only if every key matches across all
# three languages.
#
# Usage:   ./tests/contract/memref_3way.sh
# Expect:  SfaMemRef2 = 56 bytes in all three languages.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

red="$(tput setaf 1 2>/dev/null || echo '')"
green="$(tput setaf 2 2>/dev/null || echo '')"
reset="$(tput sgr0   2>/dev/null || echo '')"

ALL_PASS=0  # 0=pass, 1=fail

# ── C ────────────────────────────────────────────────────────────────

C_BIN="$SCRIPT_DIR/memref_3way_c"
cc -I"$PROJECT_ROOT/include" -o "$C_BIN" "$SCRIPT_DIR/memref_3way.c"
C_OUT="$("$C_BIN")"
rm -f "$C_BIN"

# ── Rust ─────────────────────────────────────────────────────────────

RUST_OUT=$(cd "$PROJECT_ROOT/rust" && cargo test --lib test_memref_3way_contract -- --nocapture 2>&1 \
    | grep '^Rust:' || true)

# ── Python ───────────────────────────────────────────────────────────

source "$PROJECT_ROOT/.venv/bin/activate" 2>/dev/null || true
PY_OUT=$(python3 "$SCRIPT_DIR/memref_3way.py")

# ── Compare ──────────────────────────────────────────────────────────
# Each output has lines: Prefix:value
# We extract value for each key and cross-compare.

check_key() {
    local key="$1"       # e.g. "56" or "allocated=0"
    local c_val="" py_val="" rust_val=""

    c_val=$(echo "$C_OUT"    | grep "^C:${key}$"    | cut -d: -f2-)
    rust_val=$(echo "$RUST_OUT" | grep "^Rust:${key}$" | cut -d: -f2-)
    py_val=$(echo "$PY_OUT"   | grep "^Python:${key}$"| cut -d: -f2-)

    if [[ "$c_val" == "$key" && "$rust_val" == "$key" && "$py_val" == "$key" ]]; then
        echo "${green}C:${key} Rust:${key} Python:${key} — PASS${reset}"
    else
        ALL_PASS=1
        echo "${red}FAIL for ${key}: C='${c_val}' Rust='${rust_val}' Python='${py_val}'${reset}"
    fi
}

for key in "40" "56" "72" "88" "allocated=0" "aligned=8" "offset=16"; do
    check_key "$key"
done

echo ""
if [[ "$ALL_PASS" -eq 0 ]]; then
    echo "${green}ALL 3-WAY CONTRACT TESTS PASSED${reset}"
    exit 0
else
    echo "${red}SOME 3-WAY CONTRACT TESTS FAILED${reset}"
    exit 1
fi
