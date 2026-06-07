# compiler/AGENTS.md — Python 编译管线

## 环境

Python 3.10 only, `.venv` by uv。绝不使用 conda。
激活: `source .venv/bin/activate`

## 入口

| 脚本 | 用途 |
|------|------|
| `compiler/compile.py` | 模型导出: `torch.export` → FX Graph → sf dialect → `model.mlir` + `metadata.json` |
| `compiler/compile_dylib.py` | `.dylib` 编译: `model.mlir` → lowering pipeline → LLVM IR → `.dylib` |

## 编译管线概览

```
torch.export → FX Graph → compiler/fx/converter.py (sf dialect + hf_key_map)
  → sf_to_linalg_pass (C++ sf→linalg lowering)
  → compiler/backend/llvm_backend.py (canonicalize → tile → bufferize → loops → cf → llvm)
  → mlir-translate → llc → cc -shared → .dylib
```

`dynamic_shapes={"input_ids": {0: Dim("batch"), 1: Dim("seq")}}` 必须传。
`cache_policy=CachePolicy.for_llama(...)` → SDPA 边界切分 (28 functions)；无 → 按层切分 (16 functions)。

## Proto 生成

```bash
make proto-gen  # sfa_abi.proto → gen/proto/{python,rust}
```

## 包结构

```
compiler/
├── fx/             FX Graph → MlirModule 转换
│   ├── converter.py   主转换入口: fx_graph_to_mlir()
│   ├── split.py       按层/SDPA边界切分为多函数
│   └── utils.py       ATen→HAL映射, shape/type解析
├── artifact/       MlirModule I/O
│   ├── ir.py          MlirOp/MlirFunction/MlirModule 数据类
│   ├── parse.py       model.mlir 文本解析
│   ├── serialize.py   model.mlir 序列化
│   ├── binary.py      constants.bin 读写
│   └── load.py        权重加载
├── pipeline/       编译编排
│   ├── __init__.py    compile_mlir() 入口
│   ├── stages.py      BUILTIN_STAGES, run_stages
│   └── stages_utils.py Stage/StageResult 类型
├── backend/        LLVM 后端
│   ├── compile_utils.py 外部工具 (mlir-translate, llc, cc)
│   ├── llvm_backend.py  linalg→LLVM lowering 编排
│   ├── fixups.py        IR 文本修复
│   ├── dylib.py         dylib 重链接 + 新鲜度检查
│   └── verify.py        降级 IR 验证 + 错误上下文
├── dialect/        sf dialect 定义
│   ├── sf.py           sf op 类
│   ├── builder.py      SfModule 构建器
│   ├── _op_defs.py     OpDef 表 + ATen→HAL 表
│   └── op_catalog.py   算子目录
├── shape/          Shape 推导
│   ├── shape_inference.py   shape 推导表 + 入口
│   ├── shape_inference_pure.py  纯 shape 函数
│   └── shape_inference_utils.py 广播/类型工具
├── hal/            HAL IR 生成
│   ├── hal_ir_builder.py  MlirModule → hal_ir.json
│   ├── sf_decompose.py    sf op 分解
│   └── op_lowering/       op→HAL lowering
├── passes/         MLIR pass (CSE, DCE, FuseRMSNorm, FuseSiLU)
├── quantize/       量化策略 (AWQ, SmoothQuant, FP8, mixed)
├── tp/             张量并行
├── rwkv/           RWKV 方言
├── utils/          工具 (logging, errors, lazy_imports)
└── tests/          测试 (镜像源码结构)
```

## 关键模块

| 路径 | 职责 |
|------|------|
| `compiler/pipeline/__init__.py` | 编译入口: `compile_mlir()` |
| `compiler/sfa_abi.py` | Proto 序列化/反序列化 (model→binary) |
| `compiler/artifact/binary.py` | 二进制 artifact: `constants.bin` 读写 |
| `compiler/pipeline/stages.py` | Lowering stage 定义与编排 |
| `compiler/fx/converter.py` | FX Graph → sf dialect 转换 |
| `compiler/passes/` | MLIR pass: fusion, quantize |
| `compiler/sfa_weights.py` | 权重序列化与 weight mapping |

## 测试

```bash
pytest compiler/tests/           # 编译器测试 (按源码结构分层)
make test-pipeline-quick         # 管线快速 IR 验证 (<5s)
make test-pipeline-smoke         # 管线完整烟雾测试 (<90s)
```

## Lint

```bash
make lint  # ruff + mypy (<2s)
```
