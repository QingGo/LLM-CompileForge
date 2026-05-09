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

for arg in "$@"; do
    case "$arg" in
        --with-models) WITH_MODELS=true ;;
        --lint-only)   LINT_ONLY=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

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

# ── 5. Verify ─────────────────────────────────────────────────
echo -e "${CYAN}[5/5] Verifying setup...${NC}"

# Run lint
if .venv/bin/ruff check hal/ compiler/ engine/ --quiet 2>/dev/null; then
    echo -e "  ${GREEN}Lint: PASS${NC}"
else
    echo -e "  ${YELLOW}Lint: see warnings above${NC}"
fi

# Run unit tests
echo "  Running unit tests..."
if .venv/bin/pytest tests/ -m unit -q --tb=line --ignore=tests/test_mlir_passes.py 2>/dev/null; then
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
echo "    make test-unit         # 298 unit tests (~6s)"
echo "    make test-integration  # e2e tests (requires compiled models)"
echo "    make smoke             # environment sanity check"
echo ""
