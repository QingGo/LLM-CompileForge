# tests/AGENTS.md — 测试约定

## 测试金字塔

| 层级 | 位置 | 命令 | 预算 |
|------|------|------|------|
| **Contract** (跨子项目 ABI) | `tests/contract/` | `make test-contract` | <1s |
| **Unit** (子项目内) | 各子项目 test/ | `make test-rust-unit`, `pytest compiler/tests/` | <5s |
| **Integration** | `tests/` | `make test-pipeline-smoke`, `make test-rust-integ` | <90s |
| **E2E** | `tests/` | `make test-e2e-forward`, `make test-e2e-forward-hal` | <3min |
| **Correctness** | `tests/` | `make test-forward-cos` | <60s |

## Contract 测试

`tests/contract/memref_3way.{c,py,sh}` — 跨 C/Rust/Python 三方的 MemRef 布局验证。

- 验证 `SFATensorRaw` struct 的 sizeof/offsetof 三方一致
- 不依赖模型编译，纯 ABI 契约
- 三方输出必须逐 key 匹配

## E2E 测试

```bash
make test-e2e-forward        # Path A (dylib) forward 正确性
make test-e2e-forward-hal    # Path B (HAL IR) forward 正确性
```

E2E 测试需要已编译的模型 artifact (`compiled/<model>/model.mlir` + `.dylib`)。

## 新增测试

- **新 compiler pass**: `compiler/tests/` 下添加 pytest，测试 IR 变换
- **新 HAL op**: `runtime/tests/` 下添加 Rust 测试
- **新 lowering**: `sf-dialect/test/Sf/` 下添加 `.mlir` FileCheck 测试
- **跨子项目行为变更**: 在 `tests/contract/` 添加 contract test，再补 E2E

## TDD 铁律

修复 bug: **先写单元测试复现 → 修代码 → 单测通过 → E2E 验证**。禁止跳过单测直接 E2E。
