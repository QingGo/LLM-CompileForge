# Contributing to LLM-CompileForge

See @AGENTS.md for project conventions and skill routing.
See @.opencode/CONTEXT.md for shared language and architecture overview.

## One-command setup

```bash
bash scripts/setup.sh
```

This handles Python version checks, installs `uv`, detects your platform, and
sets up the virtual environment correctly.  After it completes:

```bash
source .venv/bin/activate
make test-unit    # 298 tests, ~6 seconds
```

## Platform notes

| Platform | Torch source | Notes |
|---|---|---|
| Linux x86\_64 (GPU) | PyPI default (CUDA) | `uv sync` works out of the box |
| Linux x86\_64 (CPU) | PyTorch CPU index | `scripts/setup.sh` auto-detects this |
| macOS arm64 (M1/M2/M3) | PyPI default | Native MPS support |
| macOS x86\_64 (Intel) | github fork wheel | qingo/pytorch release |

**macOS Intel developers** — PyTorch does not publish a PyPI wheel for this
platform.  The setup script will print clear instructions for the conda +
symlink approach.  If you only need to contribute lint fixes or unit tests,
use `bash scripts/setup.sh --lint-only` to skip torch entirely.

## Development workflow

```
make lint          # ruff + mypy (<2s, 41 files)
make test-unit     # pytest -m unit (298 tests, ~6s)
make test-fast     # lint + unit + smoke (<15s)
make smoke         # environment sanity check
make test-all      # lint + unit + integration + smoke
```

Bug fix workflow:
1. Write a unit test that reproduces the bug (must fail before the fix)
2. Implement the fix
3. `make lint && make test-unit`
4. `make smoke`

## Running integration tests

Integration and baseline tests require compiled model artifacts:

```bash
# Compile the smallest test model (~30 seconds)
.venv/bin/python scripts/compile.py tiny-llama

# Run integration tests
make test-integration
```

Pre-compiled models are available for: `tiny-llama`, `opt-125m`, `qwen`.
Qwen requires the model weights in `models/Qwen/Qwen3.5-0.8B/`.

## Project structure

```
compiler/       # Model export, IR, MLIR emission, optimization passes
engine/         # Runtime inference: executor, scheduler, KV cache
hal/            # Hardware abstraction layer (PyTorch backend)
cache/          # Radix tree prefix cache
server/         # FastAPI inference server
configs/        # Per-model YAML configuration files
scripts/        # CLI tools (setup, compile, benchmarks)
tests/          # Test suite (unit + integration + baseline)
```

## Code conventions

Subproject-specific conventions are documented in each subproject's AGENTS.md:

- Python compiler: @compiler/AGENTS.md
- Rust runtime:   @runtime/AGENTS.md
- C++ sf-dialect: @sf-dialect/AGENTS.md
- Tests:          @tests/AGENTS.md

Shared conventions:
- **Imports**: use `from __future__ import annotations` at the top of every Python file.
- **Type annotations**: project is `mypy --strict` (excluding tests).
- **Line length**: 120 characters (configured in `pyproject.toml`).

## PR checklist

- [ ] `make lint` passes (ruff + mypy, zero warnings)
- [ ] `make test-unit` passes (298 tests)
- [ ] New bug fixes include a reproducing unit test
- [ ] No new HAL interface method signature changes (add only)
- [ ] Functionality milestone? Run `make profile` and record baseline
