#!/usr/bin/env bash
# Build _sfDialectsNanobind.so with proper linking.
#
# Uses -undefined dynamic_lookup for MLIR/LLVM symbols.  This is correct
# because MLIR Python bindings (libMLIRPythonCAPI.dylib) are loaded into
# the process by _mlir.cpython-310-darwin.so at import time, making all
# MLIR/LLVM symbols available globally.  Explicit linking against
# libMLIRPythonCAPI.dylib would create a SECOND copy of the MLIR global
# state (dialect registry, pass registry), causing pass lookup to fail.
#
# Python symbol is linked explicitly to avoid undefined symbol errors.
#
# Usage: bash scripts/build_sf_extension.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

SF_DIR="$PROJECT_ROOT/sf-dialect"
BUILD_DIR="$SF_DIR/build"
LLVM_BUILD="$PROJECT_ROOT/llvm-project/build"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python3"

# Locate MLIR Python shared libraries — use the LOCAL BUILD's copy
# (The .venv's copy from the release wheel has PassManager issues)
LOCAL_MLIR_LIBS_DIR="$LLVM_BUILD/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"
if [ ! -d "$LOCAL_MLIR_LIBS_DIR" ]; then
    echo "ERROR: Local MLIR Python libs not found at $LOCAL_MLIR_LIBS_DIR"
    echo "       (has setup.sh been run?)"
    exit 1
fi
echo "Using MLIR Python libs from: $LOCAL_MLIR_LIBS_DIR"

# Rebuild sf-dialect static libraries (if C++ code changed)
echo "==> Rebuilding sf-dialect..."
cmake --build "$BUILD_DIR" --target SfDialect 2>&1 | tail -3

# Get Python include and nanobind include paths
_PY_INC=$($VENV_PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')
_PY_LIB_DIR=$($VENV_PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
_NB_INC=$($VENV_PYTHON -c 'import nanobind, os; print(os.path.dirname(nanobind.__file__))')/include
_PY_LIB=$(ls ~/.local/share/uv/python/cpython-3.10.19-macos-x86_64-none/lib/libpython3.10.dylib 2>/dev/null || echo "")

if [ -z "$_PY_LIB" ]; then
    echo "ERROR: libpython3.10.dylib not found"
    exit 1
fi

echo "==> Building _sfDialectsNanobind.so with -undefined dynamic_lookup..."
echo "    (MLIR symbols resolved from already-loaded libMLIRPythonCAPI.dylib)"
OUTPUT_SO="$LOCAL_MLIR_LIBS_DIR/_sfDialectsNanobind.cpython-310-darwin.so"
clang++ -std=c++17 -fPIC -shared -O2 \
    -o "$OUTPUT_SO" \
    -DMLIR_BINDINGS_PYTHON_DOMAIN=mlir_sf \
    -I"$SF_DIR/include" \
    -I"$LLVM_BUILD/include" \
    -I"$LLVM_BUILD/tools/mlir/include" \
    -I"$PROJECT_ROOT/llvm-project/llvm/include" \
    -I"$PROJECT_ROOT/llvm-project/mlir/include" \
    -I"$SF_DIR" \
    -I"$BUILD_DIR/include" \
    -I"$_PY_INC" \
    -isystem "$_NB_INC" \
    "$SF_DIR/python/SfExtensionNanobind.cpp" \
    "$SF_DIR/lib/CAPI/Dialects.cpp" \
    "$BUILD_DIR/lib/Sf/libSfDialect.a" \
    "$BUILD_DIR/lib/CAPI/libSfCAPI.a" \
    "$_PY_LIB" \
    -undefined dynamic_lookup

echo "==> Copying .so to source tree (backup)..."
mkdir -p "$SF_DIR/python/mlir_sf/_mlir_libs"
cp "$OUTPUT_SO" "$SF_DIR/python/mlir_sf/_mlir_libs/"

echo "==> Setting up sf_dialect.pth to point to LOCAL build..."
SITE_PKG="$PROJECT_ROOT/.venv/lib/python3.10/site-packages"
LOCAL_SF_PKG="$BUILD_DIR/python_packages/sf"
echo "$LOCAL_SF_PKG" > "$SITE_PKG/sf_dialect.pth"
# Ensure local MLIR Python takes precedence over release wheel
echo "$LLVM_BUILD/tools/mlir/python_packages/mlir_core" >> "$SITE_PKG/sf_dialect.pth"

echo "==> Verifying import works (against LOCAL build)..."
$VENV_PYTHON -c "
import sys
sys.path.insert(0, '$LLVM_BUILD/tools/mlir/python_packages/mlir_core')
import mlir.ir as ir
import mlir.passmanager as pm
from mlir_sf._mlir_libs._sfDialectsNanobind import sf

# Test PassManager works
ctx = ir.Context()
sf.register_dialects(ctx._CAPIPtr, load=True)
with ir.Location.unknown(ctx):
    m = ir.Module.parse('func.func @f(%a: tensor<4x8xf32>) -> tensor<4x8xf32> { %0 = arith.addf %a, %a : tensor<4x8xf32> return %0 : tensor<4x8xf32> }', ctx)
    p = pm.PassManager.parse('builtin.module(canonicalize)', ctx)
    p.run(m.operation)
print('OK: PassManager works')
"

echo "==> Build complete."
