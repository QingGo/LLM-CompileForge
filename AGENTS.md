# AGENTS.md — LLM-CompileForge

## 环境

```bash
source scripts/env.sh
```

Python 3.10 only。**绝不使用 conda 环境** — uv 独立管理。

## 编译管线

| Step | 命令 | 产物 |
|------|------|------|
| 0. sf-dialect build | `make build-so` | `_sfDialectsNanobind.so` |
| 1. 导出模型 MLIR | `compiler/compile.py opt-125m` | `model.mlir` + `metadata.json` |
| 2. 编译 .dylib | `compiler/compile_dylib.py outputs/compiled/<model>` | `lib<model>.dylib` |
| 3. Rust 构建 | `make build` | `target/release/serveforge` |

**完整流程**: `make build-all` 或 `make rebuild`

`compile_mlir()` 必须传 `dynamic_shapes={"input_ids": {0: Dim("batch"), 1: Dim("seq")}}`。`cache_policy=CachePolicy.for_llama(...)` 触发 SDPA 边界切分 (28 functions)；无则为按层切分 (16 functions)。

## 关键文件索引

| 路径 | 内容 |
|------|------|
| `AGENTS.md` | **当前文件** — 项目级铁律 |
| `.opencode/CONTEXT.md` | 共享语言 (四层架构 / Path A vs B / 术语表) |
| `.opencode/TRAPS.md` | 全部已知陷阱 |
| `compiler/pipeline/__init__.py` | 编译入口: `compile_mlir()` |
| `compiler/compile.py` | 模型导出 CLI |
| `compiler/compile_dylib.py` | dylib 编译 CLI |
| `python_runtime/hal/interface.py` | HAL 核心接口 (Device, Buffer, OpExecutor) |
| `python_runtime/engine/llm_engine.py` | 推理引擎入口 |
| `runtime/src/engine/runner.rs` | InferenceRunner — Path A 推理主循环 (step/generate) |
| `runtime/src/engine/compute_graph_runner.rs` | ComputeGraphRunner — 加载 dylib 执行 function graph |
| `.omo/plans/` | 实现计划 |

## 根目录结构

| 目录 | 类型 | 作用 |
|------|------|------|
| `compiler/` | 子项目 | Python 编译器：FX Graph → sf dialect → linalg → LLVM → .dylib |
| `python_runtime/` | 子项目 | Python 运行时：HAL → Engine → Server 四层架构的 Python 端 |
| `runtime/` | 子项目 | Rust 运行时：Path A (dylib) 推理执行 |
| `sf-dialect/` | 子项目 | C++ MLIR dialect：sf ops 定义 + sf→linalg lowering pass |
| `include/` | 契约层 | 跨项目接口：`sfa.h` (热路径) + `sfa_abi.proto` (冷路径) |
| `gen/` | 生成代码 | Proto 编译产物 (Python + Rust) |
| `kernels/` | 独立模块 | PyTorch 算子实现：flash_attention, rms_norm, quantize |
| `scripts/` | 工具 | 项目级脚本：diagnostics/, checks/, 入口工具 |
| `tests/` | 测试 | 端到端/集成测试 (单元测试在各子项目内) |
| `outputs/` | 产物 | 编译产物 (compiled/, logs/, benchmark_results.json) |
| `models/` | 数据 | 下载的 HF 模型文件 |
| `docs/` | 文档 | 设计文档、教程 |
| `.opencode/` | 配置 | 共享语言 (CONTEXT.md), 陷阱 (TRAPS.md), 技能 |
| `.omo/` | 工作流 | 实现计划 (.omo/plans/), 证据 (.omo/evidence/) |

## 子项目约定 (渐进式展开)

- Python compiler: @compiler/AGENTS.md
- Rust runtime:   @runtime/AGENTS.md
- Python runtime: @python_runtime/AGENTS.md
- C++ sf-dialect: @sf-dialect/AGENTS.md
- 测试:          @tests/AGENTS.md
- 公共接口:      @include/sfa.h (热路径) + @include/sfa_abi.proto (冷路径)
- 贡献指南:      @CONTRIBUTING.md

## 反馈环

| 命令 | 用途 | 预算 |
|------|------|------|
| `make lint` | ruff + mypy | <5s |
| `./scripts/gate.sh` | t + engine/HAL/compiler  + Rust  + KV  | <5min |
| `make test-fixup && make test-pipeline-quick && make test-rust-unit` | fixup + IR + Rust 单元 | test-fixup <2s, pipeline-quick <60s, rust-unit 2-4min (含 dylib E2E) |
| `make test-forward-smoke` | forward 无 NaN | <5s |
| `make test-pipeline-smoke && make test-rust-integ` | pipeline + Rust 集成 | <90s |
| `make verify-dylib-fresh` | dylib 比 model.mlir 新 | <2s |
| `make test-forward-cos` | cosine regression | <60s |
| `make test-function-golden` | per-function golden (release, 含 seq6/seq32 重量级) | <3min |

## 技能路由 (按需加载)

| 问题 | 技能 |
|------|------|
| 架构/设计讨论 | @.opencode/skills/grill-me/SKILL.md |
| TDD 开发 | @.opencode/skills/tdd/SKILL.md |
| Rust SIGSEGV/NaN | @.opencode/skills/debug-rust-forward/SKILL.md |
| Pipeline 挂死 | @.opencode/skills/pipeline-hang-debug/SKILL.md |
| Pipeline 故障 | @.opencode/skills/pipeline-debug/SKILL.md |
| LLVM lowering cast 残留 | @.opencode/skills/fix-unrealized-casts/SKILL.md |
| 向量化 IR 爆炸 | @.opencode/skills/debug-vectorization/SKILL.md |
| HF cosine 精度退化 | @.opencode/skills/debug-correctness/SKILL.md |
| 模型导出失败 | @.opencode/skills/debug-model-export/SKILL.md |
| weight not found | @.opencode/skills/trace-weight-mapping/SKILL.md |
| 内存调试 (lldb/ASAN/malloc) | @.opencode/skills/debug-tools/SKILL.md |
| 新 MLIR pass | @.opencode/skills/add-mlir-pass/SKILL.md |
| 新 HAL op | @.opencode/skills/add-hal-op/SKILL.md |
| 架构改进 | @.opencode/skills/improve-codebase-architecture/SKILL.md |
| 设计+文档 | @.opencode/skills/grill-with-docs/SKILL.md |
| 结构化调试循环 | @.opencode/skills/diagnose/SKILL.md |
| 会话交接 | @.opencode/skills/handoff/SKILL.md |
| 精简模式 | @.opencode/skills/caveman/SKILL.md |
| 反馈优化 | @.opencode/skills/feedback-optimization/SKILL.md |

## 开发范式 — 契约驱动、子项目独立

**核心原则**：每个子项目只依赖 `include/sfa.h` + `include/sfa_abi.proto` 定义的契约，不依赖其他子项目的实现。

```
1. 定义接口     — include/sfa.h + include/sfa_abi.proto (唯一真相来源)
2. 改造 sf-dialect — 独立测试，验证输出符合契约
3. 改造 compiler   — 独立测试，不依赖 sf-dialect 的具体实现
4. 改造 runtime    — 独立测试，不依赖 compiler 的具体实现
5. E2E 验证        — 此阶段不应出问题；出问题说明 2/3/4 的测试不扎实
```

**E2E 失败的处理**：定位到具体的子项目 → 只在那个子项目内用 TDD 复现和修复。一次只改一个子项目。

**子项目依赖方向**：

```
include/sfa.h  ←── 热路径契约 (SfaMemRef, 零拷贝)
include/sfa_abi.proto ←── 冷路径契约 (SfaFuncMeta, protobuf)
        │
        ├── sf-dialect (C++)     ← 依赖 .h/.proto，不依赖 Python/Rust
        ├── compiler  (Python)   ← 依赖 .h/.proto + sf-dialect .so，不依赖 Rust
        └── runtime   (Rust)     ← 依赖 .h/.proto + dylib 导出符号，不依赖 Python
```

**关键规则**：
- 子项目正确性测试**不应依赖**其他子项目的实现——只依赖契约定
- 子项目接口**窄而深**——暴露最少必要信息，内部实现可任意换
- 任何满足契约的实现都可以替换——不同 compiler 生成兼容 dylib，不同 runtime 加载同一个 dylib
- E2E 测试是**最后验证手段**，不是开发时的调试工具

## TDD 铁律

**修复 bug 必须先写单元测试复现，再修代码，测试通过后验证 E2E。** 禁止直接跑 E2E 猜测 fix。
单测优先（<0.1s 反馈），E2E 最后验证。先 `cargo check` 再 `cargo build`。ASAN 最后用。

## 反馈环优化

| 原则 | 做法 |
|------|------|
| **先 check 再 build** | `cargo check` <1s，`cargo build` ~10s |
| **输出到日志文件** | `./binary > /tmp/log.txt 2>&1`，再 `grep` 按需读取 |
| **单测优先** | 单元测试 <0.1s，E2E ~3min。先 TDD 定位再 E2E 验证 |
| **ASAN 最后** | 先 debug build + lldb 缩小范围，ASAN 精确定位 |

## 调试方法论 (摘要)

详见 `.opencode/CONTEXT.md` 调试章节。核心原则：

1. **变更归因**: `git stash` → checkout 旧 commit → 复现 → 回退。不可凭 `nm`/`grep` 推断。
2. **Token 偏差**: 精度是最后排除的假设。diff > 10 先查输入值/数据流，精度累积最后考虑。
3. **精度门控**: cos_sim 高不等于精度问题。必须通过 4 项检查 (零均值/对称分布/无离群/top-N 重叠)。
4. **三路对比验证**: numpy ←→ Path A (Python ctypes dylib) ←→ Path A (Rust server)。禁止自洽性检验替代跨路径对比。

## HAL 规则 (摘要)

- 所有 kernel 通过 `executable.execute(op_name, stream, &inputs, &outputs)`。禁止裸调 ciface。
- Accelerate BLAS: `ldb >= max(K,1)`。窄矩阵 fallback naive matmul。
- Unsafe audit: `bash scripts/audit-unsafe.sh`。全部 `unsafe {}` 必须有 `// SAFETY:` 注释。
- 完整 HAL 规则、陷阱表、BLAS 细节 → @runtime/AGENTS.md
