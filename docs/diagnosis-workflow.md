# Bug Diagnosis Workflow

## Overview

When compiled model output diverges from the HuggingFace reference, this 4-phase
workflow isolates the root cause from the full model down to a single MLIR op.

| Phase | Tool | Granularity | Question Answered |
|-------|------|-------------|-------------------|
| 1 | `test_per_function_cos.py` | Function (16 per model) | Which function diverges? |
| 2 | `reduce_mlir.py` | MLIR module | Which functions are needed to reproduce? |
| 3 | `reduce_function_ops.py` | Op within function | Which op causes it? |
| 4 | `test_precision_contract.py` | Single-op dylib | Can the op reproduce in isolation? |

Phases 1, 2, and 4 are implemented.  Phase 3 requires an SSA-rewriting model
and is tracked in a separate plan.

## Phase 1: Locate the Diverging Function

`test_per_function_cos.py` runs the compiled dylib function-by-function
against pre-generated golden `.npz` files with a 4-gate precision check:
cosine >= 0.9999, zero-mean relative error < 0.01, max outlier / std < 10,
and top-10 Jaccard >= 0.5.  A cos > 0.99 that fails the gate is a logic bug.

```bash
# All 64 tests (4 seq configs x 16 functions). -x stops at first failure.
# --timeout=0 disables the project-global 1s timeout (fixture setup needs more).
pytest compiler/tests/test_per_function_cos.py -v -x --timeout=0

# Narrow by case or function.
pytest compiler/tests/test_per_function_cos.py -v -x -k "seq6" --timeout=0
pytest compiler/tests/test_per_function_cos.py -v -k "main_3" --timeout=0
```

**Key observations from first failure**: which function name, which gate failed,
and whether the failure is seq_len-dependent (seq=1 catches causal mask bugs;
seq=32 catches overflow/stride bugs).

## Phase 2: Function-Level MLIR Reduction

`reduce_mlir.py` removes non-failing functions from `model.mlir`, shrinking
the reproducer by 90%+.  It uses the MLIR API for robust parsing of both
`func.func @name(...)` and generic `"func.func"() { }` serialization formats.

```bash
# One-at-a-time function deletion (most aggressive).
python scripts/diagnostics/reduce_mlir.py outputs/compiled/opt_125m_fresh/model.mlir \
  --strategy function \
  --interestingness "./check_divergence.sh {}" \
  --output /tmp/reduced.mlir

# Binary search for first failing function in the chain.
python scripts/diagnostics/reduce_mlir.py outputs/compiled/opt_125m_fresh/model.mlir \
  --strategy binary \
  --interestingness "./check_divergence.sh {}" \
  --output /tmp/reduced_first.mlir

# Track smallest interesting variant by line count.
python scripts/diagnostics/reduce_mlir.py outputs/compiled/opt_125m_fresh/model.mlir \
  --strategy function --metric lines \
  --interestingness "./check_divergence.sh {}" \
  --output /tmp/reduced.mlir
```

The `--interestingness` script is written per bug.  It receives a temp MLIR
file path (replacing `{}` in the command), compiles it to a dylib, runs the
target function test, and exits 0 if the bug is still present.  The function
strategy deletes functions one at a time and retries from the start; the
binary strategy finds the first failing function index, keeping only the prefix.

## Phase 3: Op-Level Reduction (Future)

`reduce_function_ops.py` will bisect individual ops within the failing
function.  Requires SSA use-def chain rewriting after op deletion.
**Status**: not yet implemented.

## Phase 4: Minimal Reproduction

Add a test case to `test_precision_contract.py` that compiles a single
`sf.*` op through the full sf→linalg→LLVM→dylib pipeline and verifies
output against NumPy/PyTorch reference.

```bash
pytest compiler/tests/test_precision_contract.py -v --timeout=300
pytest compiler/tests/test_precision_contract.py -v -k "TestOpPrecision" --timeout=300
```

To add a new case: add a `NumericalTestCase` to `tests/contract/fixtures/precision_cases.pb`
with input/weight/expected data and a `min_cosine` threshold, then add a
`_make_mlir_for_case` branch for the op type.

## Worked Example: Non-KV Logit Bias Bug

**Symptoms**: `main_1` (layer_0) produces cos=0.99975 at seq=6.  cos > 0.99
indicates a logic bug, not floating-point error.  Only the cosine gate fails.

### Phase 1: Identification

```bash
pytest compiler/tests/test_per_function_cos.py -v -x --timeout=0
# FAILED test_function_matches_golden[seq6-main_1]
# Function 'main_1' FAILED 4-gate check (case=seq6, seq_len=6):
#     cos          = 0.9997500000  [OK=False]
#     mean_rel_err = 5.23e-05      [OK=True]
#     max_outlier  = 1.42e-02      [OK=True]
#     top-10 jaccard = 0.800       [OK=True]
```

Tests for seq=1, seq=2, and seq=32 all pass for `main_1`.  The divergence
is seq_len-dependent and surfaces at the first transformer layer.

### Phase 2: MLIR Function Reduction

Write an interestingness script that compiles the reduced MLIR and targets
only `main_1`:

```bash
# check_func1_divergence.sh
#!/bin/bash
set -e
M="$1"
# Compile M → dylib, run: pytest -k "seq6-main_1"
# Exit 0 if the test still FAILS.
```

```bash
python scripts/diagnostics/reduce_mlir.py outputs/compiled/opt_125m_fresh/model.mlir \
  --strategy function \
  --interestingness "./check_func1_divergence.sh {}" \
  --output /tmp/reduced_func1.mlir
```

Expected: keeps `main_0` (embedding) and `main_1` (layer_0); deletes `main_2`..`main_15`.

### Phase 4 Plan

Once the op-level reducer identifies the exact op within `main_1`, add a
precision contract test for that single op to confirm whether the bug is in
compiler lowering or the kernel.

## Reference

| Path | Purpose |
|------|---------|
| `compiler/tests/test_per_function_cos.py` | Per-function 4-gate precision test |
| `compiler/tests/generate_golden_outputs.py` | Deterministic golden `.npz` generator |
| `compiler/tests/test_precision_contract.py` | Single-op precision contract tests |
| `scripts/diagnostics/reduce_mlir.py` | MLIR function-level reducer |
| `tests/data/golden/npy/opt_125m/configs.json` | Test matrix (seq=1,2,6,32 x 16 functions) |
| `tests/contract/fixtures/precision_cases.pb` | Precision contract test vectors |
| `.omo/plans/function-level-testing.md` | Full infrastructure plan |
