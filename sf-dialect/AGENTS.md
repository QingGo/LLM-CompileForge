# sf-dialect/AGENTS.md — C++ MLIR Dialect

## 构建

```bash
make build-so        # Python nanobind 扩展: _sfDialectsNanobind.so
make build-plugin    # sf-opt 工具 (standalone MLIR pass runner)
make build-sf        # 完整 C++ 构建 (tablegen + .a + .so + sf-opt)
```

CMake 配置由根 `Makefile` 自动处理 (`sf-dialect/build/CMakeCache.txt` target)。
前提: `llvm-project/build/` 已编译 (MLIR + LLVM)。

## 测试

```bash
cd sf-dialect/build && lit -v test/Sf/   # FileCheck 测试
```

测试文件: `sf-dialect/test/Sf/sfa_contract_verify.mlir`

## 关键文件

| 路径 | 职责 |
|------|------|
| `include/Sf/SfOps.td` | sf dialect op 定义 (TableGen) |
| `include/Sf/SfPasses.td` | Pass 声明 (TableGen) |
| `include/Sf/SfPasses.h` | Pass 注册头文件 (auto-generated) |
| `lib/Sf/SfOps.cpp` | Op 实现 (verifiers, builders, fold) |
| `lib/Sf/SfDialect.cpp` | Dialect 注册 |
| `lib/Sf/SfLowerToLinalg.cpp` | sf→linalg 主 lowering 入口 |
| `lib/Sf/SfaContractPass.cpp` | lowering 验证 pass (输出结构检查) |
| `lib/CAPI/Dialects.cpp` | C API force-linkage: 注册 sf dialect 到 MLIR context |

## Pass 注册流程

1. 在 `include/Sf/SfPasses.td` 声明 pass
2. 在 `include/Sf/SfPasses.h` 自动生成声明 (tablegen)
3. 在 `lib/CAPI/Dialects.cpp` 添加 `force_linksf()` 条目 — **遗漏则 pass 不可见**

## Lowering 组织

```
sf dialect ops
  → SfLowerToLinalg.cpp (主调度: 按 op 类型分发)
    → SfLowerMatmul.cpp      (matmul → linalg.matmul)
    → SfLowerAttention.cpp   (attention → linalg ops)
    → SfLowerNormalization.cpp (rms_norm/layer_norm)
    → SfLowerReduce.cpp      (reduce_sum/mean)
    → SfLowerActivation.cpp  (silu/gelu/relu)
    → SfLowerGenOps.cpp      (通用 op: element_wise, reshape)
    → SfLowerShape.cpp       (shape ops)
    → SfLowerSeqOps.cpp      (序列 op)
    → SfLowerCompare.cpp     (比较 op)
    → SfLowerFused.cpp       (融合 op)
    → SfFusionPasses.cpp     (算子融合 pass)
  → linalg dialect
    → 后续 MLIR lowering pipeline (canonicalize → tile → bufferize → ... → LLVM)
```
