#!/usr/bin/env bash
# setup.sh — Bootstrap LLM-CompileForge development environment.
#
# One-command setup for any platform:
#   bash scripts/setup.sh
#
# Options:
#   --with-models     Also compile tiny-llama for e2e tests (~2 min extra)
#   --lint-only       Skip torch install (lint + unit tests still work)
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WITH_MODELS=false
LINT_ONLY=false
QUICK=false

for arg in "$@"; do
    case "$arg" in
        --with-models) WITH_MODELS=true ;;
        --lint-only)   LINT_ONLY=true ;;
        --quick)        QUICK=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── 0. Helper: build LLVM/MLIR from source ──────────────────────
build_llvm() {
    echo -e "${CYAN}[LLVM] Building MLIR from source...${NC}"
    source "$PROJECT_ROOT/.llvm-version"
    LLVM_DIR="$PROJECT_ROOT/llvm-project"
    BUILD_DIR="$LLVM_DIR/build"

    if [ ! -d "$LLVM_DIR" ]; then
        echo "  Cloning $FORK ..."
        git clone "$FORK" "$LLVM_DIR"
    fi

    cd "$LLVM_DIR"
    git fetch origin
    git checkout "$COMMIT"
    echo "  LLVM at $(git rev-parse --short HEAD)"

    if [ -d "$BUILD_DIR" ] && [ -f "$BUILD_DIR/build.ninja" ]; then
        echo "  Build directory exists — running ninja (incremental)..."
    else
        echo "  Configuring cmake ..."
        cmake -G Ninja \
            -S llvm \
            -B build \
            -DLLVM_ENABLE_PROJECTS=mlir \
            -DLLVM_TARGETS_TO_BUILD=Native \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_ENABLE_ASSERTIONS=ON \
            -DLLVM_INSTALL_UTILS=ON \
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
            -DPython3_EXECUTABLE="$PROJECT_ROOT/.venv/bin/python3" \
            -DLLVM_CCACHE_BUILD=ON \
            -DCMAKE_C_COMPILER=clang \
            -DCMAKE_CXX_COMPILER=clang++ \
            -DLLVM_USE_LINKER=lld
    fi

    ninja -C build tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/_mlir.cpython-310-darwin.so
    echo -e "  ${GREEN}MLIR Python bindings built.${NC}"

    # Generate a minimal setup.py so this directory is pip-installable
    MLIR_PKG="$BUILD_DIR/tools/mlir/python_packages/mlir_core"
    cat > "$MLIR_PKG/setup.py" << 'SETUPEOF'
from setuptools import setup
setup(
    name="mlir-core",
    version="23.0.0",
    description="MLIR Python bindings (self-compiled)",
    packages=["mlir", "mlir._mlir_libs", "mlir._mlir_libs._mlir",
              "mlir.dialects", "mlir.extras"],
    package_data={"mlir": ["_mlir_libs/*", "_mlir_libs/_mlir/*",
                           "_mlir_libs/_mlir/dialects/*",
                           "py.typed"]},
    include_package_data=True,
    zip_safe=False,
)
SETUPEOF

    $PROJECT_ROOT/.venv/bin/pip install --force-reinstall --no-deps -e "$MLIR_PKG" 2>&1 | tail -3
    echo -e "  ${GREEN}mlir-core installed in .venv (editable).${NC}"
}

# ── 0b. Helper: build sf-dialect ─────────────────────────────────
build_sf_dialect() {
    echo -e "${CYAN}[sf] Building sf-dialect...${NC}"
    SF_DIR="$PROJECT_ROOT/sf-dialect"
    BUILD_DIR="$SF_DIR/build"
    LLVM_BUILD="$PROJECT_ROOT/llvm-project/build"
    VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python3"

    mkdir -p "$BUILD_DIR"
    cmake -G Ninja \
        -S "$SF_DIR" \
        -B "$BUILD_DIR" \
        -DPython3_EXECUTABLE="$VENV_PYTHON" \
        -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
        -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_BUILD_TYPE=Release

    # Build the static library and Python package structure (sources, tablegen).
    ninja -C "$BUILD_DIR" SfDialect SfPythonModules 2>&1 | tail -3

    # Build the extension .so manually — cmake's built-in extension links
    # MLIR core static libraries which cause dialect re-registration with
    # the LLVM build's libMLIRPythonCAPI.dylib.
    # We link only libSfDialect.a and resolve all MLIR core symbols at
    # runtime via -undefined dynamic_lookup.
    _PY_INC=$($VENV_PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')
    _NB_INC=$($VENV_PYTHON -c 'import nanobind, os; print(os.path.dirname(nanobind.__file__))')/include

    clang++ -std=c++17 -fPIC -shared \
        -o "$BUILD_DIR/python_packages/sf/mlir_sf/_mlir_libs/_sfDialectsNanobind.cpython-310-darwin.so" \
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
        -undefined dynamic_lookup

    # Write .pth file so .venv can find mlir_sf
    SITE_PKG="$PROJECT_ROOT/.venv/lib/python3.10/site-packages"
    echo "$BUILD_DIR/python_packages/sf" > "$SITE_PKG/sf_dialect.pth"

    echo -e "  ${GREEN}sf-dialect built and mlir_sf available.${NC}"
}

echo -e "${CYAN}=== LLM-CompileForge Dev Setup ===${NC}"
echo ""

# ── 1. Python version check ──────────────────────────────────
echo -e "${CYAN}[1/5] Checking Python >= 3.10...${NC}"
PYTHON=""
for cmd in python3.10 python3.11 python3.12 python3.13 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$("$cmd" -c "import sys; print(sys.version_info.major)")
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            echo -e "  ${GREEN}Found $PYTHON $ver${NC}"
            break
        fi
    fi
done
if [ -z "$PYTHON" ]; then
    echo -e "${RED}Python >= 3.10 required. Install from https://python.org${NC}"
    exit 1
fi

# ── 2. Install uv if missing ──────────────────────────────────
echo -e "${CYAN}[2/5] Checking uv package manager...${NC}"
if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo -e "${RED}Failed to install uv. Install manually: https://docs.astral.sh/uv/${NC}"
        exit 1
    fi
fi
echo -e "  ${GREEN}uv $(uv --version)${NC}"

# ── 3. Virtual environment ────────────────────────────────────
echo -e "${CYAN}[3/5] Creating virtual environment...${NC}"
cd "$PROJECT_ROOT"
uv venv --python "$PYTHON" 2>/dev/null || true

# ── 4. Platform-aware dependency install ──────────────────────
echo -e "${CYAN}[4/5] Installing dependencies...${NC}"
OS="$(uname -s)"
ARCH="$(uname -m)"

if $LINT_ONLY; then
    echo "  Lint-only mode: installing lint + test tools (no torch)."
    uv sync --extra lint --extra test 2>&1 | tail -5
    echo -e "  ${GREEN}Lint-only dependencies installed.${NC}"
    # Skip unit tests (many need torch); verify lint instead.
    echo -e "${CYAN}[5/5] Verifying lint...${NC}"
    if .venv/bin/ruff check hal/ compiler/ engine/ --quiet 2>/dev/null; then
        echo -e "  ${GREEN}Lint: PASS${NC}"
    fi
    if .venv/bin/mypy hal/ compiler/ engine/ server/ --config-file pyproject.toml --no-error-summary 2>/dev/null; then
        echo -e "  ${GREEN}Mypy: PASS${NC}"
    fi
    echo ""
    echo -e "${GREEN}=== Lint-only setup complete ===${NC}"
    echo ""
    echo "  Lint passes.  Install torch to run unit tests:"
    echo "    On macOS Intel: see instructions above (conda + symlink)"
    echo "    On other platforms: uv sync --extra dev"
    echo ""
    exit 0
fi
    # ── macOS Intel ──────────────────────────────────
    echo -e "  ${YELLOW}Platform: macOS x86_64 — no PyPI torch wheel available.${NC}"
    echo ""

    # Check if conda is available
    CONDA_FOUND=false
    if command -v conda &>/dev/null; then
        CONDA_FOUND=true
    fi

    # Check if torch is already importable (symlinked from conda)
    TORCH_OK=false
    if .venv/bin/python -c "import torch" 2>/dev/null; then
        ver=$( .venv/bin/python -c "import torch; print(torch.__version__)" )
        echo -e "  ${GREEN}torch ${ver} already available (symlinked from conda).${NC}"
        TORCH_OK=true
    fi

    if ! $TORCH_OK; then
        echo ""
        echo "  ── macOS Intel torch setup ──"
        echo "  Two options:"
        echo ""
        if $CONDA_FOUND; then
            echo "  [Option A] Use conda (recommended if you already have conda):"
            echo ""
            echo "    # 1. Create a conda env with torch:"
            echo "    conda create -n serveforge python=3.10 pytorch -c pytorch -y"
            echo ""
            echo "    # 2. Symlink torch + deps into .venv:"
            echo "    CONDA_SITE=\$(conda info --base)/envs/serveforge/lib/python3.10/site-packages"
            echo "    ln -s \"\$CONDA_SITE/torch\" .venv/lib/python3.10/site-packages/torch"
            echo "    for pkg in torchgen sympy networkx filelock fsspec jinja2 mpmath; do"
            echo "      [ -e \"\$CONDA_SITE/\$pkg\" ] && ln -s \"\$CONDA_SITE/\$pkg\" .venv/lib/python3.10/site-packages/\$pkg"
            echo "    done"
            echo ""
        fi
        echo "  [Option B] Development without torch (lint only):"
        echo ""
        echo "    bash scripts/setup.sh --lint-only"
        echo ""
        echo "  After setting up torch, run 'uv sync --extra lint --extra test' to install remaining deps."
        echo ""
        exit 0
    fi
    # torch is available, proceed with uv sync (no torch in extras)
    uv sync --extra lint --extra test 2>&1 | tail -5
elif [ "$OS" = "Linux" ]; then
    # Check for NVIDIA GPU
    if command -v nvidia-smi &>/dev/null; then
        echo "  Platform: Linux $(uname -m) with NVIDIA GPU — using CUDA torch."
        uv sync --extra dev 2>&1 | tail -5
    else
        echo "  Platform: Linux $(uname -m) CPU-only — using PyTorch CPU index."
        uv sync --extra dev --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -5
    fi
else
    # macOS arm64, Windows, etc.
    echo "  Platform: $OS $ARCH — using default PyPI index."
    uv sync --extra dev 2>&1 | tail -5
fi

echo -e "  ${GREEN}Dependencies installed.${NC}"

# ── 4b. Install MLIR Python bindings ─────────────────────────
echo -e "${CYAN}[4b/6] Installing MLIR Python bindings...${NC}"
if .venv/bin/python -c "import mlir.ir; import mlir_sf" 2>/dev/null; then
    echo -e "  ${GREEN}mlir + mlir_sf already available.${NC}"
elif $QUICK; then
    echo "  --quick: installing mlir-core from GitHub Release (no local build)."
    .venv/bin/python -m pip install --no-deps \
        'https://github.com/QingGo/llvm-project/releases/download/llvmorg-22.1.5/mlir_core-22.1.5-cp310-cp310-macosx_11_0_x86_64.whl' \
        2>&1 | tail -3
    echo -e "  ${YELLOW}mlir_sf not available. C++ lowering will fall back to Python.${NC}"
else
    build_llvm
    build_sf_dialect
fi

# ── 4c. Build Rust runtime ───────────────────────────────────
echo -e "${CYAN}[4c/6] Building Rust runtime...${NC}"
if command -v cargo &>/dev/null; then
    if .venv/bin/python -c "import llm_serveforge_runtime" 2>/dev/null; then
        echo -e "  ${GREEN}Rust runtime already available.${NC}"
    else
        echo "  Building with maturin..."
        .venv/bin/pip install maturin 2>&1 | tail -1
        unset CONDA_PREFIX 2>/dev/null || true
        .venv/bin/maturin develop -r --manifest-path rust/Cargo.toml 2>&1 | tail -3
    fi
else
    echo -e "  ${YELLOW}cargo not found — skip Rust runtime. Install rustup: https://rustup.rs${NC}"
fi

# ── 5. Verify ─────────────────────────────────────────────────
echo -e "${CYAN}[5/6] Verifying setup...${NC}"

# Run lint
if .venv/bin/ruff check hal/ compiler/ engine/ --quiet 2>/dev/null; then
    echo -e "  ${GREEN}Lint: PASS${NC}"
else
    echo -e "  ${YELLOW}Lint: see warnings above${NC}"
fi

# Run unit tests
echo "  Running unit tests..."
if .venv/bin/pytest tests/ -m unit -q --tb=line 2>/dev/null; then
    echo -e "  ${GREEN}Unit tests: PASS${NC}"
else
    echo -e "  ${YELLOW}Unit tests: some failures (see above)${NC}"
fi

echo ""

# ── Models ────────────────────────────────────────────────────
if [ -d "$PROJECT_ROOT/compiled/tiny_llama" ]; then
    echo -e "${GREEN}compiled/tiny_llama found — e2e tests enabled.${NC}"
else
    echo -e "${YELLOW}No compiled models found.${NC}"
    echo "  To enable e2e tests, compile a model:"
    echo ""
    echo "    .venv/bin/python scripts/compile.py tiny-llama"
    echo ""
    echo "  Or re-run setup with:"
    echo "    bash scripts/setup.sh --with-models"
fi

if $WITH_MODELS; then
    echo ""
    echo -e "${CYAN}Compiling tiny-llama model for e2e tests...${NC}"
    .venv/bin/python scripts/compile.py tiny-llama 2>&1
    echo ""
fi

echo -e "${GREEN}=== Setup complete ===${NC}"
echo ""
echo "  Development commands:"
echo "    source .venv/bin/activate"
echo "    make lint              # ruff + mypy"
echo "    make test-unit         # 244 unit tests (~6s)"
echo "    cargo test             # 46 Rust tests (rust/)"
echo "    make test-integration  # e2e tests (requires compiled models)"
echo "    make smoke             # environment sanity check"
echo ""
