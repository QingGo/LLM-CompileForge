#!/usr/bin/env bash
# build_so.sh — Build _sfDialectsNanobind.so with dependency tracking.
#
# Part of the build dependency chain (fix-systemic-issues plan §1.2):
#   1. Locates sf-dialect build directory
#   2. Runs cmake --build for SfDialect static library
#   3. Links the .so extension with ABI compatibility check
#   4. Installs .so to Python package path
#
# Usage: bash scripts/build_so.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SF_DIR="$PROJECT_ROOT/sf-dialect"
BUILD_DIR="$SF_DIR/build"
LLVM_BUILD="$PROJECT_ROOT/llvm-project/build"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python3"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── [1/4] Locate build directory ──────────────────────────────
echo -e "${CYAN}==> [1/4] Locating sf-dialect build directory...${NC}"

if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}ERROR: sf-dialect build directory not found at $BUILD_DIR${NC}"
    echo "       Run: bash scripts/setup.sh"
    exit 1
fi
echo "       Found: $BUILD_DIR"

LOCAL_MLIR_LIBS_DIR="$LLVM_BUILD/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"
if [ ! -d "$LOCAL_MLIR_LIBS_DIR" ]; then
    echo -e "${RED}ERROR: Local MLIR Python libs not found at $LOCAL_MLIR_LIBS_DIR${NC}"
    echo "       Has setup.sh been run?"
    exit 1
fi
echo "       MLIR Python libs: $LOCAL_MLIR_LIBS_DIR"

# ── [2/4] Build SfDialect static library ──────────────────────
echo -e "${CYAN}==> [2/4] Building SfDialect static library...${NC}"
if ! cmake --build "$BUILD_DIR" --target SfDialect 2>&1; then
    echo -e "${RED}ERROR: cmake --build SfDialect failed${NC}"
    exit 2
fi

A_FILE="$BUILD_DIR/lib/Sf/libSfDialect.a"
if [ -f "$A_FILE" ]; then
    A_SIZE=$(stat -f "%z" "$A_FILE" 2>/dev/null || stat -c "%s" "$A_FILE" 2>/dev/null || echo "unknown")
    echo "       ✓ libSfDialect.a ($A_SIZE bytes)"
else
    echo -e "${RED}ERROR: $A_FILE not found after build${NC}"
    exit 2
fi

# ── [3/4] Build .so with ABI compatibility check ──────────────
echo -e "${CYAN}==> [3/4] Building _sfDialectsNanobind.so...${NC}"

# ABI compatibility check: detect LLVM ABI breaking check symbols
# in the static library. If present, reconfigure cmake with FORCE_OFF.
echo "       Checking ABI compatibility..."
if nm "$A_FILE" 2>/dev/null | grep -qi "DisableABI\|ABIBreaking\|LLVM_ABI_BREAKING"; then
    echo -e "       ${CYAN}⚠ ABI breaking checks detected in static library${NC}"
    echo "       → Reconfiguring cmake with LLVM_ABI_BREAKING_CHECKS=FORCE_OFF..."
    cmake -G Ninja \
        -S "$SF_DIR" \
        -B "$BUILD_DIR" \
        -DPython3_EXECUTABLE="$VENV_PYTHON" \
        -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
        -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ABI_BREAKING_CHECKS=FORCE_OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --target SfDialect 2>&1 | tail -3
    echo "       ✓ Rebuild complete (ABI-fixed)"
fi

# Python include paths
_PY_INC=$($VENV_PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')
_NB_INC=$($VENV_PYTHON -c 'import nanobind, os; print(os.path.dirname(nanobind.__file__))')/include
_PY_LIB=$(ls ~/.local/share/uv/python/cpython-3.10.19-macos-x86_64-none/lib/libpython3.10.dylib 2>/dev/null || echo "")

if [ -z "$_PY_LIB" ]; then
    echo -e "${RED}ERROR: libpython3.10.dylib not found${NC}"
    echo "       Expected at: ~/.local/share/uv/python/cpython-3.10.19-macos-x86_64-none/lib/"
    exit 1
fi

OUTPUT_SO="$LOCAL_MLIR_LIBS_DIR/_sfDialectsNanobind.cpython-310-darwin.so"
echo "       Linking with -undefined dynamic_lookup..."
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
    "$A_FILE" \
    "$BUILD_DIR/lib/CAPI/libSfCAPI.a" \
    "$_PY_LIB" \
    -undefined dynamic_lookup

SO_SIZE=$(stat -f "%z" "$OUTPUT_SO" 2>/dev/null || stat -c "%s" "$OUTPUT_SO" 2>/dev/null || echo "unknown")
echo "       ✓ $OUTPUT_SO ($SO_SIZE bytes)"

# ── [4/4] Install .so to Python package path ──────────────────
echo -e "${CYAN}==> [4/4] Installing .so to Python package path...${NC}"

# Copy to source tree for version control compatibility
mkdir -p "$SF_DIR/python/mlir_sf/_mlir_libs"
cp "$OUTPUT_SO" "$SF_DIR/python/mlir_sf/_mlir_libs/"
echo "       → $SF_DIR/python/mlir_sf/_mlir_libs/"

# Update .pth files so .venv can find mlir_sf and local MLIR build
SITE_PKG="$PROJECT_ROOT/.venv/lib/python3.10/site-packages"
LOCAL_SF_PKG="$BUILD_DIR/python_packages/sf"
echo "$LOCAL_SF_PKG" > "$SITE_PKG/sf_dialect.pth"
echo "$LLVM_BUILD/tools/mlir/python_packages/mlir_core" >> "$SITE_PKG/sf_dialect.pth"
echo "       ✓ .pth files updated"

# Quick verification
echo -e "       Verifying import..."
$VENV_PYTHON -c "
import sys
sys.path.insert(0, '$LLVM_BUILD/tools/mlir/python_packages/mlir_core')
import mlir.ir as ir
import mlir.passmanager as pm
from mlir_sf._mlir_libs._sfDialectsNanobind import sf
ctx = ir.Context()
sf.register_dialects(ctx._CAPIPtr, load=True)
with ir.Location.unknown(ctx):
    m = ir.Module.parse('func.func @f(%a: tensor<4x8xf32>) -> tensor<4x8xf32> { return %a : tensor<4x8xf32> }', ctx)
    p = pm.PassManager.parse('builtin.module(canonicalize)', ctx)
    p.run(m.operation)
print('       ✓ Import & PassManager OK')
" 2>&1

echo -e "${GREEN}==> Build complete. _sfDialectsNanobind.so is ready.${NC}"
