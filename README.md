# LLM-ServeForge

Hardware-agnostic LLM inference compiler & runtime — Phase 1 MVP.

**Core idea**: Build vLLM/SGLang-level serving capabilities on a minimal HAL (Hardware Abstraction Layer) so any AI accelerator can get production-grade inference by implementing just 3 interfaces: Device, Buffer, OpExecutor.

## Quick Start

```bash
# Install
source .venv/bin/activate
uv sync

# Run tests
make lint && make test-unit

# Start server
python -c "
from server.app import create_app, create_engine
import uvicorn
engine = create_engine()
app = create_app(engine)
uvicorn.run(app, port=8000)
"

# Call the API
curl http://localhost:8000/health
```

## Architecture

```
PyTorch Model → torch.export → FX Graph → Custom IR → Optimized IR
                                                         │
API Server → Scheduler → BlockManager → Executor → HAL → CPU/GPU
```

### Modules

| Module | Purpose |
|--------|---------|
| `hal/` | Hardware Abstraction Layer — Device, Buffer, OpExecutor (16 ops) |
| `compiler/` | AOT compiler: FX Graph → IR → optimization passes (CSE, DCE, ConstantFold, FuseRMSNorm, FuseSiLU) |
| `engine/` | Runtime: Scheduler (Continuous Batching + Chunked Prefill), BlockManager (PagedAttention), Sampler |
| `server/` | FastAPI server with OpenAI-compatible `/v1/completions` and `/v1/chat/completions` |

### Compile a Model

```bash
# Compile facebook/opt-125m from local HF cache
python scripts/compile.py opt-125m

# Compile tiny test Llama model  
python scripts/compile.py tiny-llama
```

Artifacts are saved to `./compiled/<model>/`:
- `model.mlir` — Standard MLIR text (canonical artifact format)
- `weights.pth` — Model weights
- `metadata.json` — Compilation metadata

## Development

```bash
make lint          # ruff + mypy
make test-unit     # pytest -m unit (171 tests)
make test-integration  # pytest -m integration
make smoke         # Quick health check
make profile       # Performance baseline
```

## Known Limitations (Phase 1)

- Static shape AOT compilation — models run at the export input shape
- Single-machine, single-process execution
- PyTorch backend only (custom Triton/CUDA kernels planned for Phase 2)
- Server concurrency via `asyncio.Lock` (background engine loop planned for Phase 2)

## License

MIT
