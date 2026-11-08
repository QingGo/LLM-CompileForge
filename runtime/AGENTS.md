# runtime/AGENTS.md — Rust 运行时

## 构建

`Cargo.toml` 位于 `runtime/`。默认 features: `["cli"]`。

```bash
cargo build --release          # 产物: target/release/serveforge
make build                     # 等同 cargo build --release
```

## 测试

```bash
cargo test --lib               # Rust 单元测试 (fast)
make test-rust-unit            # 同上
make test-rust-integ           # Rust 集成测试
```

## 关键模块

| 路径 | 职责 |
|------|------|
| `runtime/src/main.rs` | CLI 入口: `serveforge` binary |
| `runtime/src/model/abi.rs` | Proto 解析: `sfa_abi.proto` → Rust structs |
| `runtime/src/hal/sfa.rs` | `SfaMemRef` — MLIR memref 到 Rust 内存的映射 |
| `runtime/src/hal/primitives/` | 底层 kernel: matmul, attention, element_wise, reduce |
| `runtime/src/engine/compute_graph_runner.rs` | Path A runner: 加载 `.dylib` 执行 function graph |
| `runtime/src/engine/executor.rs` | ModelExecutor: dylib 加载 + forward pass |
| `runtime/src/engine/runner.rs` | InferenceRunner: step() 循环, sampler, tokenizer |
| `runtime/src/engine/sampler.rs` | Token 采样 (greedy, temperature, top-k, top-p) |
| `runtime/src/engine/tokenizer.rs` | HuggingFace tokenizer.json 封装 |
| `runtime/src/kv_cache.rs` | KV cache 管理 |
| `runtime/src/model/weight_loader.rs` | 权重加载: WeightProvider + hf_key_map |

### 已废弃模块 (Path B)

Path B (HAL IR op-by-op dispatch) 已在 commit `495ca08` 中移除（`hal/rust/`、`hal_runner/` 不再存在）；旧实现见 git history。

## HAL 规则 (强制)

1. **所有 kernel 调用通过 `executable.execute(op_name, stream, &inputs, &outputs)`**。禁止裸调 ciface。
2. 新 kernel: C++ lowering → dylib → Rust execute dispatch。
3. 调试: `HAL_TRACE=1|2|3` 控制 op 级日志；`HAL_DUMP_FUNC=0,1` 保存中间 tensor。

## macOS BLAS 陷阱

Accelerate BLAS 要求 `ldb >= max(K,1)`。attention 窄矩阵 `[64,4]` 场景 `ldb=4 < K=64` → **必须 fallback naive matmul**。

触发条件: `runtime/src/hal/primitives/matmul.rs:ldb < k`
症状: SIGSEGV
规避: 窄矩阵自动检测并 fallback，无需手动干预。添加新 matmul 路径时注意此约束。

## 已知陷阱 (Path B)

Path B 的旧陷阱已随实现移除，仅保留在 git history 中。

## Unsafe Audit

```bash
bash scripts/audit-unsafe.sh
```

全部 `unsafe {}` block 必须有 `// SAFETY:` 注释说明理由。

## 调试

- SIGSEGV: `lldb -- target/debug/serveforge ...` + ASAN (`RUSTFLAGS="-Zsanitizer=address"`)
- NaN: `HAL_TRACE=3 HAL_DUMP_FUNC=0,1` 逐 op 保存中间结果
- 完整调试流程 → `docs/diagnosis-workflow.md`
