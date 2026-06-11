#!/bin/bash
# check_lowering.sh — Pre-built interestingness test for reduce_mlir.py
#
# Checks if the sf→linalg lowering pipeline fails on a given MLIR file.
# Can be used directly or as a reduce_mlir.py interestingness test:
#
#   reduce_mlir.py source.mlir --interestingness "./check_lowering.sh {}"
#
# Exit codes: 0 = lowering failed (bug present=interesting)
#             1 = lowering succeeded (bug gone)
#             2 = MLIR unparseable (invalid reduction)

set -euo pipefail

MLIR_FILE="${1:-}"
PASS_PIPELINE="${CHECK_LOWERING_PIPELINE:-sf-promote-weights,canonicalize,cse,sf-chain-wrapper,sf-lower-to-linalg}"
TIMEOUT="${CHECK_LOWERING_TIMEOUT:-60}"

if [ -z "$MLIR_FILE" ]; then
    echo "Usage: $0 <mlir_file>" >&2
    exit 1
fi

if [ ! -f "$MLIR_FILE" ]; then
    echo "File not found: $MLIR_FILE" >&2
    exit 1
fi

# Use check_pass.py as the backend
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python "$SCRIPT_DIR/check_pass.py" "$PASS_PIPELINE" "$MLIR_FILE" --timeout "$TIMEOUT"
