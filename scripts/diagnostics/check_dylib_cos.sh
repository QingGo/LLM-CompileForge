#!/bin/bash
# check_dylib_cos.sh — Compile reduced MLIR → dylib → compare cosine vs Python reference
#
# Pre-built interestingness test for reduce_mlir.py when debugging
# correctness regressions (Issue #45: cos=0.525).
#
# Usage with reduce_mlir.py:
#   reduce_mlir.py model.lowered.mlir \
#     --interestingness "./check_dylib_cos.sh {}" --output reduced.mlir
#
# Env vars:
#   CHECK_DYLIB_ARTIFACT_DIR — compiled model dir (default: outputs/compiled/opt_125m_fresh)
#   CHECK_DYLIB_THRESHOLD    — cosine threshold (default: 0.999)
#   CHECK_DYLIB_TIMEOUT      — per-stage timeout (default: 120)
#
# Exit: 0 = cos below threshold (bug present = interesting)
#       1 = cos above threshold or compilation failed (bug gone)
#       2 = MLIR unparseable

set -euo pipefail

MLIR_FILE="${1:-}"
ARTIFACT_DIR="${CHECK_DYLIB_ARTIFACT_DIR:-outputs/compiled/opt_125m_fresh}"
THRESHOLD="${CHECK_DYLIB_THRESHOLD:-0.999}"
TIMEOUT="${CHECK_DYLIB_TIMEOUT:-120}"

if [ -z "$MLIR_FILE" ] || [ ! -f "$MLIR_FILE" ]; then
    echo "Usage: $0 <mlir_file>" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_DIR="$(mktemp -d /tmp/check_dylib_cos_XXXXXX)"
trap "rm -rf $WORK_DIR" EXIT

cd "$PROJECT_DIR"

# Step 1: Parse and compile reduced MLIR → .dylib
python -c "
import sys, os
sys.path.insert(0, '.')
from compiler.backend.compile_utils import _setup_mlir_path
_setup_mlir_path()

import mlir.ir as ir
ctx = ir.Context()
ctx.allow_unregistered_dialects = True
ctx.load_all_available_dialects()
try:
    from mlir._mlir_libs import _mlirRegisterEverything
    reg = ir.DialectRegistry()
    _mlirRegisterEverything.register_dialects(reg)
    ctx.append_dialect_registry(reg)
except: pass

with ir.Location.unknown(ctx):
    mod = ir.Module.parse(open('$MLIR_FILE').read(), ctx)

from compiler.pipeline.stages import BUILTIN_STAGES, run_stages
results = run_stages(mod, ctx, BUILTIN_STAGES, log_dir='')
failed = [r for r in results if not r.success]
if failed:
    sys.exit(1)  # compilation failed = bug may be gone

from compiler.backend.compile_utils import emit_llvm_ir_to_file, llc_compile, link_dylib
llvm_ir = os.path.join('$WORK_DIR', 'model.ll')
emit_llvm_ir_to_file(mod, llvm_ir)
obj = llc_compile(llvm_ir, output=os.path.join('$WORK_DIR', 'model.o'))
dylib_path = os.path.join('$WORK_DIR', 'libmodel.dylib')
link_dylib([obj], dylib_path)
print(dylib_path)
" > "$WORK_DIR/dylib_path.txt" 2>/dev/null || {
    # Compilation failed — bug might be gone, or IR is malformed
    exit 1
}

DYLIB_PATH="$(cat "$WORK_DIR/dylib_path.txt")"
if [ ! -f "$DYLIB_PATH" ]; then
    exit 1
fi

# Step 2: Compare cosine via CtypesOracle
python -c "
import sys
sys.path.insert(0, '.')
from scripts.ctypes_oracle import CtypesOracle
o = CtypesOracle('$ARTIFACT_DIR')
cos = o.compare('$DYLIB_PATH')
sys.exit(0 if cos < $THRESHOLD else 1)
" 2>/dev/null
