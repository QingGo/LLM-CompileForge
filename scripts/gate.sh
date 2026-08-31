#!/usr/bin/env bash
# gate.sh — executable discipline gate for every change.
#
# Runs:
#   1. make lint
#   2. Python engine/HAL fast tests (KV contract + HAL)
#   3. Rust fast unit subset (runner config + main_0 pass-through contract)
#   4. KV contract source assertions (TRAPS.md K13-K23)
#
# Budget: < 5 minutes.  Do not commit when this script fails.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

source scripts/env.sh

GATE_START=$SECONDS
BUDGET_SECS=300

step() {
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "▶ $1"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

fail_if_over_budget() {
  local elapsed=$((SECONDS - GATE_START))
  if (( elapsed > BUDGET_SECS )); then
    echo "❌ gate.sh exceeded ${BUDGET_SECS}s budget (${elapsed}s)" >&2
    exit 1
  fi
}

step "lint (ruff + mypy)"
make lint
fail_if_over_budget

step "Python engine/HAL + compiler seam fast tests"
"$PROJECT_ROOT/.venv/bin/pytest" -q \
  python_runtime/engine/tests/test_generate_kv.py \
  python_runtime/engine/tests/test_kv_intercept.py \
  python_runtime/hal/tests/test_hal.py \
  python_runtime/hal/tests/test_pytorch_backend.py \
  compiler/tests/test_linalg_blas_rewrite.py
fail_if_over_budget

step "Rust fast unit subset"
(cd runtime && cargo test --lib test_runner_config_with_cache_policy_enables_kv -- --quiet)
(cd runtime && cargo test --lib main0_weight_passthrough_map -- --quiet)
fail_if_over_budget

step "KV contract assertions (K13-K23)"
"$PROJECT_ROOT/.venv/bin/python" scripts/checks/verify_kv_contract.py
fail_if_over_budget

elapsed=$((SECONDS - GATE_START))
echo
echo "✅ gate.sh PASSED in ${elapsed}s (budget ${BUDGET_SECS}s)"
