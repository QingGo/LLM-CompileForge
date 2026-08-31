# Dylib cos 对齐复现步骤

**commit**: `157af46`

## 复现步骤

### 1. 删除旧产物，从零编译
```bash
cd /Users/zeng/code/LLM-CompileForge
rm -rf outputs/compiled
source .venv/bin/activate
export KMP_DUPLICATE_LIB_OK=TRUE
unset CONDA_PREFIX
export DYLD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.10/site-packages/torch/lib:$(pwd)/sf-dialect/build/python_packages/sf/mlir_sf/_mlir_libs:$(pwd)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"
make build-all MODEL=opt-125m
```

### 2. 构建 forward_check 并运行
```bash
cd runtime && cargo build --bin forward_check
cd ..
FORWARD_CHECK_TOKENS="1,2,3,4" ./runtime/target/debug/forward_check
```

### 3. 对比 HF
```python
import numpy as np, torch
from numpy.linalg import norm
from transformers import OPTForCausalLM

def cos(a, b):
    a32 = a.astype(np.float32); b32 = b.astype(np.float32)
    return float(np.dot(a32, b32) / (norm(a32) * norm(b32) + 1e-30))

rust_logits = np.loadtxt('/tmp/rust_logits.csv', delimiter=',', dtype=np.float32).reshape(1, 4, 50272)
model = OPTForCausalLM.from_pretrained('facebook/opt-125m', local_files_only=True).eval()
with torch.no_grad():
    hf_logits = model(torch.tensor([[1, 2, 3, 4]]).long()).logits.numpy().astype(np.float32)

print(f'cos(HF, dylib) = {cos(hf_logits.flatten(), rust_logits.flatten()):.10f}')
```

## 复现结果

| 配置 | cos(HF, dylib) |
|------|:---:|
| batch=1, seq=4 | **0.9999953508** |
| batch=1, seq=8 | **0.9999967217** |
| batch=2, seq=4 | **0.9999962449** |

所有 argmax 匹配。

## 修复的 Bug

1. **sf.arange lowering** (`sf-dialect/lib/Sf/SfLowerGenOps.cpp`): `startVal` 使用了 input 值而非常量 0
2. **SDPA mask broadcast** (`sf-dialect/lib/Sf/SfLowerAttention.cpp`): identity map 无法处理 size-1 维度（mask head dim=1 vs QKV head dim=12）
3. **Golden 生成** (`compiler/tests/generate_golden_outputs.py`): 使用 dylib self-comparison 而非 MlirExecutor
