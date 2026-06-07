# python_runtime/AGENTS.md — Python 运行时

## 环境

```bash
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE
unset CONDA_PREFIX
export DYLD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.10/site-packages/torch/lib:$(pwd)/sf-dialect/build/python_packages/sf/mlir_sf/_mlir_libs:$(pwd)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"
```

Python 3.10 only。**绝不使用 conda 环境** — uv 独立管理。

## 包结构

```
python_runtime/
├── hal/             # 硬件抽象层: Device, Buffer, OpExecutor
│   ├── interface.py # 核心 ABC
│   ├── pytorch_backend/  # PyTorch 后端 (28 ops)
│   └── hardware_specs/   # 硬件规格 (A100, M2 Pro, ...)
├── engine/          # 推理引擎: LLMEngine, MlirExecutor
│   ├── llm_engine.py     # 引擎入口
│   ├── mlir_executor.py  # 模型 forward + 权重加载
│   ├── _inference_loop.py # 推理循环
│   ├── speculative/      # 投机解码
│   └── cache/            # Radix 缓存
├── server/          # REST API: FastAPI OpenAI-compatible
│   ├── app.py
│   └── routes.py
└── tests/           # python_runtime 测试 (pytest)
```

## 依赖方向

```
Server → Engine → HAL
```
- `server/` imports from `python_runtime.engine` and `python_runtime.hal`
- `engine/` imports from `python_runtime.hal` and `compiler.utils`
- `hal/` is a leaf — no internal cross-project dependencies

## 关键模块

| 路径 | 职责 |
|------|------|
| `python_runtime/hal/interface.py` | Device, Buffer, OpExecutor ABCs |
| `python_runtime/hal/pytorch_backend/__init__.py` | `PyTorchBackend` — 28 ops 实现 |
| `python_runtime/hal/pytorch_backend/_ops_attention.py` | `_AttentionOps`: SDPA, flash_attention, paged_attention |
| `python_runtime/engine/llm_engine.py` | `LLMEngine` — 服务主入口 |
| `python_runtime/engine/mlir_executor.py` | `MlirExecutor` — 模型 forward, dylib/Ctypes 调用 |
| `python_runtime/engine/_inference_loop.py` | `_inference_loop` — 推理调度循环 |
| `python_runtime/engine/sampler.py` | Token 采样 |
| `python_runtime/engine/cache_manager.py` | KV cache 管理 (protobuf CachePolicy) |
| `python_runtime/engine/cache/radix_cache.py` | Radix 树前缀缓存 |
| `python_runtime/server/app.py` | FastAPI app + `create_engine()` |
| `python_runtime/server/routes.py` | OpenAI-compatible `/v1/completions`, `/v1/chat/completions` |

## 测试

```bash
pytest python_runtime/hal/tests/       # HAL 测试 (test_hal, test_hardware_sim, test_pytorch_backend)
pytest python_runtime/engine/tests/    # Engine 测试 (test_batch, test_cache_manager, test_sampler)
pytest python_runtime/hal/tests/ -m unit --tb=short  # 快速单元测试
```

## 文件组织

- `tests/` 在子包内 — `python_runtime/hal/tests/`, `python_runtime/engine/tests/`
- 无 server 测试目录 — server 测试在根 `tests/test_server.py` (集成测试)

## 已知限制

- PyTorch backend only — 仅 CPU 执行
- 静态 shape: 模型在导出时的输入 shape 下运行
- 单机单进程
- Triton/CUDA 后端计划中

## 技能路由

| 问题 | 技能 |
|------|------|
| Engine 调度/性能 | @.opencode/skills/grill-me/SKILL.md |
| HAL 算子实现 | @.opencode/skills/add-hal-op/SKILL.md |
| 调试运行时问题 | @.opencode/skills/debug-rust-forward/SKILL.md |
