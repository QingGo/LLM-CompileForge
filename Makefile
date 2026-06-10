.PHONY: lint lint-clippy lint-ruff lint-mypy test-unit test-integration test-fast test-all test-model test-patterns test-smoke profile smoke clean clean-logs test-fixup test-ctypes-oracle test-pipeline-smoke test-rust test-rust-unit test-rust-integ test-rust-cov test-pipeline-quick test-changed test-pipeline-timing test-pipeline-debug test-pipeline-validate test-vec test-lower test-baseline test-compile-full test-forward-smoke test-forward-smoke-rust test-forward-cos test-weight-consistency test-consistency verify-dylib verify-dylib-fresh verify-consistency verify-diag verify-preflight check-op-consistency test-contract build-rust install-rust build-so test-dylib-cos test-dylib-cos-quick clean-compiled test-dylib-quick debug-cos diagnose test-kv-compiler test-kv-rust test-kv-python-e2e test-kv-all build-sf rebuild-clean rebuild-mlir rebuild-dylib rebuild-test rebuild build-all build build-plugin serve serve-py run-prompt test-e2e-forward

# ═══════════════════════════════════════════════════════════════
#  环境
# ═══════════════════════════════════════════════════════════════
VENV := .venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest
PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# 编译器/运行时 关键路径
LLVM_BUILD_BIN := $(PROJECT_ROOT)/llvm-project/build/bin
MLIR_LIBS_PATH := $(PROJECT_ROOT)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs
SF_MLIR_LIBS := $(PROJECT_ROOT)/sf-dialect/build/python_packages/sf/mlir_sf/_mlir_libs
TORCH_LIB_PATH := $(PROJECT_ROOT)/$(VENV)/lib/python3.10/site-packages/torch/lib
DYLIB_ENV := DYLD_LIBRARY_PATH="$(SF_MLIR_LIBS):$(TORCH_LIB_PATH):$(MLIR_LIBS_PATH)"

$(VENV):
	uv venv --python 3.10 && uv sync

# ═══════════════════════════════════════════════════════════════
#  编译管线 (3 步)
#  1. build-so —— sf-dialect Python 扩展 (C++/tablegen 变更后需重跑)
#  2. compile.py —— torch.export → model.mlir (FX→MLIR artifact)
#  3. compile_dylib.py —— model.mlir → .dylib (lowering + llc + link)
# ═══════════════════════════════════════════════════════════════

# ---- proto 代码生成 ----
PROTO_DIR := $(PROJECT_ROOT)/include
GEN_DIR := $(PROJECT_ROOT)/gen/proto

proto-gen:
	mkdir -p $(GEN_DIR)/python $(GEN_DIR)/rust
	# Python
	protoc --proto_path=$(PROTO_DIR) --python_out=$(GEN_DIR)/python $(PROTO_DIR)/sfa_abi.proto $(PROTO_DIR)/sfa_precision.proto
	# Rust (prost: package sfa → sfa/sfa.rs — single invocation for shared package)
	protoc --proto_path=$(PROTO_DIR) --prost_out=$(GEN_DIR)/rust $(PROTO_DIR)/sfa_abi.proto $(PROTO_DIR)/sfa_precision.proto
	# init for Python import
	touch $(GEN_DIR)/python/__init__.py

# ---- step 0: sf-dialect 构建 ----
.PHONY: doctor configure-sf-dialect build-sf build-so

doctor:
	@echo "=== Environment Check ==="
	@echo "PROJECT_ROOT: $(PROJECT_ROOT)"
	@echo -n "  MLIR_DIR: "; \
	if [ -f "$(PROJECT_ROOT)/llvm-project/build/lib/cmake/mlir/MLIRConfig.cmake" ]; then \
		echo "$(PROJECT_ROOT)/llvm-project/build/lib/cmake/mlir (compiled) ✓"; \
	else echo "MISSING — run: bash scripts/setup.sh"; fi
	@echo -n "  LLVM_DIR: "; \
	if [ -f "$(PROJECT_ROOT)/llvm-project/build/lib/cmake/llvm/LLVMConfig.cmake" ]; then \
		echo "$(PROJECT_ROOT)/llvm-project/build/lib/cmake/llvm (compiled) ✓"; \
	else echo "MISSING"; fi
	@echo -n "  mlir-tblgen: "; \
	if [ -x "$(PROJECT_ROOT)/llvm-project/build/bin/mlir-tblgen" ]; then \
		echo "compiled ✓"; else echo "MISSING"; fi
	@echo -n "  sf-dialect .a: "; \
	if [ -f "$(PROJECT_ROOT)/sf-dialect/build/lib/Sf/libSfDialect.a" ]; then \
		ls -lh "$(PROJECT_ROOT)/sf-dialect/build/lib/Sf/libSfDialect.a" | awk '{print $$5}'; \
	else echo "not built — run: make build-sf"; fi

# sf-dialect CMake 配置 (ABI 对齐 LLVM 构建)
# 仅在 CMakeCache.txt 不存在或 CMakeLists.txt 变更时才重配置
sf-dialect/build/CMakeCache.txt:
	@LLVM_ABI=$$(grep LLVM_ABI_BREAKING_CHECKS $(PROJECT_ROOT)/llvm-project/build/CMakeCache.txt | cut -d= -f2); \
	if [ -z "$$LLVM_ABI" ]; then LLVM_ABI=WITH_ASSERTS; fi; \
	echo "  LLVM_ABI_BREAKING_CHECKS=$$LLVM_ABI"; \
	mkdir -p sf-dialect/build && \
	cmake -G Ninja -S sf-dialect -B sf-dialect/build \
		-DPython3_EXECUTABLE=$(PROJECT_ROOT)/$(VENV)/bin/python3 \
		-DMLIR_DIR=$(PROJECT_ROOT)/llvm-project/build/lib/cmake/mlir \
		-DLLVM_DIR=$(PROJECT_ROOT)/llvm-project/build/lib/cmake/llvm \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_ABI_BREAKING_CHECKS=$$LLVM_ABI \
		-DCMAKE_BUILD_TYPE=Release

# C++ 静态库 (sf-dialect/lib/Sf/ → libSfDialect.a, lib/CAPI/ → libSfCAPI.a)
# SfDialect: ops/dialect/pass 实现
# SfCAPI: C API 桥接 + 强制 pass 符号链接 (保证 Python binding 能在运行时找到 pass)
sf-dialect/build/lib/Sf/libSfDialect.a: sf-dialect/build/CMakeCache.txt
	@echo "==> Building SfDialect static library..."
	cmake --build sf-dialect/build --target SfDialect SfCAPI

build-sf: sf-dialect/build/lib/Sf/libSfDialect.a

# Python 扩展 .so (_sfDialectsNanobind.cpython-310-darwin.so)
# clang++ 编译 SfExtensionNanobind.cpp + Dialects.cpp + libSfDialect.a + libSfCAPI.a
# 用 -undefined dynamic_lookup 使 MLIR 符号在运行时解析
# 触发条件: C++/tablegen 代码变更 → 需要重新 build-so
build-so: build-sf proto-gen
	@mkdir -p sf-dialect/build/python_packages/sf/mlir_sf/_mlir_libs && \
	ln -sf $(PROJECT_ROOT)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/_sfDialectsNanobind.cpython-310-darwin.so \
		sf-dialect/build/python_packages/sf/mlir_sf/_mlir_libs/_sfDialectsNanobind.cpython-310-darwin.so && \
	echo "$(PROJECT_ROOT)/sf-dialect/build/python_packages/sf" > $(PROJECT_ROOT)/$(VENV)/lib/python3.10/site-packages/sf_dialect.pth && \
	scripts/build_so.sh

# 独立 mlir-opt 插件 (无需 Python 即可加载 sf dialect)
# 由 CMake (sf-dialect/tools/CMakeLists.txt) 管理
# 使用: mlir-opt --load-dialect-plugin=sf-dialect/build/tools/libSfDialectPlugin.dylib
.PHONY: build-plugin
build-plugin: build-sf
	cmake --build sf-dialect/build --target SfDialectPlugin

# ---- step 1: 导出模型 (Python) ----
# compiler/compile.py: torch.export → FX Graph → MlirModule → MLIR artifact
#   cache_policy 参数: CachePolicy.for_llama(...) 触发 SDPA 边界切分
#   输出: model.mlir + metadata.json + constants.bin + constants.pth
rebuild-mlir: $(VENV)
	$(PYTHON) compiler/compile.py opt-125m --output-dir ./outputs/compiled/opt_125m_fresh

# ---- step 2: 编译 .dylib (C++ lowering + llc) ----
# compiler/compile_dylib.py: MLIR → sf→linalg → LLVM IR → .o → .dylib
#   需要 sf dialect Python bindings (需先 make build-so)
#   产物: lib<model-name>.dylib (嵌入 constants.bin)
rebuild-dylib: $(VENV)
	$(DYLIB_ENV) $(PYTHON) compiler/compile_dylib.py outputs/compiled/opt_125m_fresh --model-name opt_125m

# ---- 全量: step 1 + step 2 + Rust 测试 ----
rebuild: rebuild-clean rebuild-mlir rebuild-dylib rebuild-test

rebuild-clean:
	rm -rf outputs/compiled/opt_125m_fresh

rebuild-test:
	cd runtime && cargo test test_opt_125m_forward_runs -- --nocapture 2>&1 | tail -5

# ---- 从头完整构建 (新机器适用) ----
# Usage: make build-all                    # 编译 opt-125m → outputs/compiled/opt_125m_fresh
#        make build-all MODEL=opt-125m      # 等同默认
#        make build-all MODEL=tiny-llama    # 编译 tiny-llama → outputs/compiled/tiny_llama
# 约定: compiler/compile.py 的 CLI 参数用 dash (opt-125m), 输出目录用 underscore (opt_125m_fresh)
.PHONY: build-all
build-all: $(VENV) build-so build-plugin
	@echo "=== 完整编译流水线 ==="
	@echo ""
	@echo "▸ 检查外部依赖..."
	@if [ ! -f "$(PROJECT_ROOT)/llvm-project/build/lib/cmake/mlir/MLIRConfig.cmake" ]; then \
		echo "  ❌ MLIR 未编译。请运行: bash scripts/setup.sh"; exit 1; fi
	@if [ ! -f "$(PROJECT_ROOT)/llvm-project/build/bin/mlir-opt" ]; then \
		echo "  ❌ mlir-opt 缺少。运行: cd llvm-project/build && ninja mlir-opt"; exit 1; fi
	@echo "  ✓ LLVM/MLIR build"
	@echo "  ✓ sf-dialect build"
	@echo ""
	@echo "Step 1: 编译模型 ($(MODEL))"
	$(PYTHON) compiler/compile.py $(MODEL) --output-dir ./outputs/compiled/$(MODEL_FRESH)
	@echo ""
	@echo "Step 2: lowering + .dylib"
	rm -f outputs/compiled/$(MODEL_FRESH)/model.lowered.mlir
	PATH="$(LLVM_BUILD_BIN):$(PATH)" $(DYLIB_ENV) $(PYTHON) compiler/compile_dylib.py outputs/compiled/$(MODEL_FRESH) --model-name $(MODEL_FRESH)
	@echo ""
	@echo "Step 3: tokenizer files"
	$(DYLIB_ENV) $(PYTHON) -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('facebook/opt-125m' if '$(MODEL)' == 'opt-125m' else '$(MODEL)'); tok.backend_tokenizer.save('outputs/compiled/$(MODEL_FRESH)/tokenizer.json')"
	@if [ "$(MODEL)" = "opt-125m" ]; then \
		cp $$(python3 -c "from pathlib import Path; p = Path.home() / '.cache/huggingface/hub/models--facebook--opt-125m/snapshots'; print(sorted(p.glob('*/tokenizer_config.json'))[0])") outputs/compiled/$(MODEL_FRESH)/tokenizer_config.json; \
	fi
	@echo "Step 4: Rust 二进制"
	$(MAKE) build
	@echo ""
	@echo "✅ build-all complete"

# 模型名 → 编译产物目录名映射
# compiler/compile.py 的 targets 字典定义了默认路径, 这里统一映射
MODEL ?= opt-125m
MODEL_FRESH = $(patsubst opt-125m,opt_125m_fresh,$(patsubst tiny-llama,tiny_llama,$(MODEL)))

# ---- 快速版: dylib 编译 + cosine 测试 ----
# 两步合并, 跳过 compile.py (需要已有 model.mlir)
# debug-cos 与 test-dylib-cos-quick 几乎相同, 保留为专用诊断入口
test-dylib-cos: build-so rebuild-mlir
	rm -f outputs/compiled/opt_125m_fresh/model.lowered.mlir
	PATH="$(LLVM_BUILD_BIN):$(PATH)" $(DYLIB_ENV) $(PYTHON) compiler/compile_dylib.py outputs/compiled/opt_125m_fresh --model-name opt_125m
	$(PYTHON) tests/test_dylib_cosine.py -v

test-dylib-cos-quick: build-so
	rm -f outputs/compiled/opt_125m_fresh/model.lowered.mlir
	PATH="$(LLVM_BUILD_BIN):$(PATH)" $(DYLIB_ENV) $(PYTHON) compiler/compile_dylib.py outputs/compiled/opt_125m_fresh --model-name opt_125m
	$(PYTHON) tests/test_dylib_cosine.py -v

debug-cos: build-so
	rm -f outputs/compiled/opt_125m_fresh/model.lowered.mlir
	PATH="$(LLVM_BUILD_BIN):$(PATH)" $(DYLIB_ENV) $(PYTHON) compiler/compile_dylib.py outputs/compiled/opt_125m_fresh --model-name opt_125m
	$(PYTHON) scripts/diagnostics/diagnose_cos.py --layer 12

clean-compiled:
	rm -f outputs/compiled/opt_125m_fresh/model.mlir
	rm -f outputs/compiled/opt_125m_fresh/model.lowered.mlir
	rm -f outputs/compiled/opt_125m_fresh/model.lowered.readable.mlir
	rm -f outputs/compiled/opt_125m_fresh/model.ll
	rm -f outputs/compiled/opt_125m_fresh/lib*.dylib

# ═══════════════════════════════════════════════════════════════
#  Rust 构建 & 服务
# ═══════════════════════════════════════════════════════════════
.PHONY: build build-rust install-rust serve serve-py run-prompt

build: proto-gen
	cd runtime && cargo build --release

build-rust: $(VENV)
	@echo "  🔧 构建 Python 绑定 (maturin develop --uv --features python-bindings)..."
	@if [ -n "$$CONDA_PREFIX" ]; then echo "  ⚠️  检测到 CONDA_PREFIX，自动 unset..."; fi
	cd runtime && source ../.venv/bin/activate && unset CONDA_PREFIX && maturin develop --features python-bindings --uv 2>&1

install-rust: build-rust

PORT ?= 8000
serve: build-all
	./runtime/target/release/serveforge serve $(MODEL_FRESH) --port $(PORT)

serve-py:
	@echo "Python 后端需要 runtime/ 的 python-bindings feature 成功编译。"
	python -c "from server.app import create_app,create_engine; import uvicorn; \
	  uvicorn.run(create_app(create_engine()), port=$(PORT))" 2>&1 || \
	  echo "❌ Python 后端不可用 (llm_serveforge_runtime import 失败)"

PROMPT ?= "Hello"
MAX_TOKENS ?= 8
TEMPERATURE ?= 0.0
run-prompt: build
	$(DYLIB_ENV) ./runtime/target/release/serveforge run $(MODEL) \
	  --prompt "$(PROMPT)" --max-tokens $(MAX_TOKENS) --temperature $(TEMPERATURE) --no-chat-template

# ═══════════════════════════════════════════════════════════════
#  测试 (L0-L4 分层)
# ═══════════════════════════════════════════════════════════════

# L0: 静态检查
lint: lint-ruff lint-mypy
lint-clippy:
	cargo clippy -- -D warnings
lint-ruff:
	$(RUFF) check python_runtime/ compiler/ kernels/ tests/
lint-mypy:
	$(MYPY) python_runtime/ compiler/ kernels/ --config-file pyproject.toml
check-op-consistency:
	$(PYTHON) scripts/checks/check_op_consistency.py
test-ddr:
	@echo "=== Verifying DRR patterns ==="; \
	TBLGEN=$(PROJECT_ROOT)/llvm-project/build/bin/mlir-tblgen; \
	INCLUDES="-I$(PROJECT_ROOT)/sf-dialect/include/Sf -I$(PROJECT_ROOT)/llvm-project/mlir/include \
	          -I$(PROJECT_ROOT)/llvm-project/build/tools/mlir/include -I$(PROJECT_ROOT)/llvm-project/llvm/include \
	          -I$(PROJECT_ROOT)/llvm-project/build/include -I$(PROJECT_ROOT)/sf-dialect/include"; \
	if $$TBLGEN $$INCLUDES -gen-rewriters $(PROJECT_ROOT)/sf-dialect/include/Sf/SfLoweringPatterns.td > /dev/null 2>&1; then \
		echo "  ✅ DRR patterns compile"; \
	else echo "  ❌ DRR patterns failed"; exit 1; fi

# L0: Contract — cross-validate proto ABI fields in compiled model
TEST_CONTRACT_MODEL := outputs/compiled/opt_125m_fresh
test-contract: $(VENV)
	PYTHONPATH="$(PROJECT_ROOT)/gen/proto/python:$$PYTHONPATH" $(PYTHON) tests/contract/abi_cross_validate.py $(TEST_CONTRACT_MODEL)

# L1: 单元测试
test-unit: $(VENV)
	$(PYTEST) tests/ -m unit -v --tb=short --timeout=1
test-model: $(VENV)
	$(PYTHON) -m pytest tests/test_model_artifact.py -v --tb=short --timeout=10
test-patterns: $(VENV)
	$(PYTHON) -m pytest tests/test_lowering_patterns.py -v --tb=short --timeout=2
test-smoke: lint test-unit test-model test-patterns
test-fixup: $(VENV)
	$(PYTEST) compiler/tests/backend/test_fixup_casts.py -v --tb=short --timeout=10
test-ctypes-oracle: $(VENV)
	$(PYTEST) tests/test_ctypes_oracle.py -v --tb=short --timeout=30
test-rust-unit: $(VENV)
	mkdir -p outputs/logs/rust
	cd runtime && cargo test --lib > ../outputs/logs/rust/test_unit_$$(date +%Y%m%d_%H%M%S).log 2>&1
test-rust: test-rust-unit test-rust-integ
test-e2e-forward:
	cd runtime && cargo test --test integration_tests
test-rust-cov:
	cargo llvm-cov --lib --summary-only
test-pipeline-timing: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" $(PYTHON) scripts/diagnostics/pipeline_timing.py outputs/compiled/opt_125m_fresh
test-pipeline-debug: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" $(PYTHON) scripts/diagnostics/pipeline_debug.py outputs/compiled/opt_125m_fresh --out /tmp/pipeline_debug
test-changed: $(VENV)
	@if [ -f tests/run_related_tests.py ]; then \
		$(PYTHON) tests/run_related_tests.py; \
	else echo "⚠️  tests/run_related_tests.py 不存在，回退到 test-fast"; $(MAKE) test-fast; fi

# L1.5: 正确性
test-forward-smoke: $(VENV)
	mkdir -p outputs/logs/test
	@for model in outputs/compiled/opt_125m_fresh outputs/compiled/tiny_llama; do \
		if [ -d $$model ]; then \
			echo "[$$model]"; \
			LLM_SERVEFORGE_LOG=INFO $(PYTHON) scripts/checks/check_forward_smoke.py $$model > outputs/logs/test/forward_smoke_$$(date +%Y%m%d_%H%M%S).log 2>&1 || exit 1; \
		fi; \
	done

# L1.5: Rust forward smoke (uses forward_check binary)
test-forward-smoke-rust: verify-dylib-fresh
	cd runtime && cargo build --bin forward_check
	@echo "[forward_check] Running Rust forward pass..."
	$(DYLIB_ENV) ./runtime/target/debug/forward_check

# L1.5: HAL IR forward smoke (removed — Path B deprecated)

test-forward-cos: test-forward-smoke
	@for model in outputs/compiled/opt_125m_fresh; do \
		if [ -d $$model ]; then \
			echo "[$$model]"; \
			$(PYTHON) scripts/checks/check_forward_cos.py $$model || exit 1; \
		fi; \
	done

test-vec: $(VENV)
	$(PYTHON) tests/test_transform_vec.py
	$(PYTHON) -m pytest tests/test_pipeline_bugs.py -v --tb=short --timeout=60
test-lower: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" $(PYTHON) tests/test_lowering_diag.py
test-pipeline-quick: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" \
	$(PYTEST) compiler/tests/pipeline/test_pipeline_validation.py::test_no_arith_ops_after_lowering \
	         compiler/tests/pipeline/test_pipeline_validation.py::test_tile_sizes_within_bounds \
	         tests/test_forward_correctness.py::test_tiny_llama_compiles \
	         tests/test_forward_correctness.py::test_tiny_llama_config \
	         tests/test_forward_correctness.py::test_opt125m_compile_and_forward_cosine \
	         tests/test_forward_correctness.py::test_opt125m_forward_smoke \
	         -v --tb=short --timeout=30
test-weight-consistency: $(VENV)
	$(PYTEST) tests/test_weight_consistency.py -v --tb=short --timeout=120 -m "not slow"
verify-dylib:
	$(PYTHON) scripts/checks/verify_dylib_consistency.py outputs/compiled/opt_125m_fresh

verify-dylib-fresh:
	@for model in outputs/compiled/opt_125m_fresh outputs/compiled/tiny_llama; do \
		if [ -d $$model ]; then \
			echo "[$$model]"; \
			$(PYTHON) scripts/checks/check_dylib_freshness.py $$model || exit 1; \
		fi; \
	done
verify-consistency: $(VENV)
	$(PYTHON) scripts/checks/verify_dylib_consistency.py outputs/compiled/opt_125m_fresh
verify-diag:
	$(PYTHON) -c "import sys, os; sys.path.insert(0, os.getcwd()); from scripts.checks.verify_dylib_consistency import check_diag_tool_health; sys.exit(check_diag_tool_health('outputs/compiled/opt_125m_fresh'))"
verify-preflight: verify-consistency verify-diag

# KV Cache 测试集
test-kv-compiler:
	$(PYTEST) tests/test_kv_cache_compiler.py -x -v
test-kv-rust:
	cd runtime && cargo test --lib kv_cache
test-kv-python-e2e:
	$(PYTEST) tests/test_kv_cache_correctness.py -x -v --timeout=300
test-kv-all: test-kv-compiler test-kv-rust test-kv-python-e2e
	@echo "All KV cache tests PASSED"

# L2: 集成
test-integration: $(VENV)
	-$(PYTEST) tests/ -m integration -v --tb=short --timeout=300
test-rust-integ: $(VENV)
	mkdir -p outputs/logs/rust
	cd runtime && cargo test --bin serveforge > ../outputs/logs/rust/test_integ_$$(date +%Y%m%d_%H%M%S).log 2>&1
test-pipeline-smoke: $(VENV)
	$(PYTHON) tests/test_pipeline_smoke.py outputs/compiled/opt_125m_fresh --timeout 120
test-pipeline-validate: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" $(PYTHON) -m pytest compiler/tests/pipeline/test_pipeline_validation.py -v --tb=short --timeout=60
test-compile-full: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" $(PYTHON) -m pytest tests/test_compile_full.py -v --tb=short --timeout=300
test-consistency: $(VENV)
	@echo "=== 四路一致性测试 (HF/PY/CTYPES/RUST) ==="
	$(DYLIB_ENV) PYTHONPATH="$(PROJECT_ROOT)/llvm-project/build/tools/mlir/python_packages/mlir_core" \
	$(PYTHON) scripts/checks/test_consistency.py

test-dylib-quick: build-so
	rm -f outputs/compiled/tiny_llama/model.lowered.mlir
	PATH="$(LLVM_BUILD_BIN):$(PATH)" $(DYLIB_ENV) $(PYTHON) compiler/compile_dylib.py outputs/compiled/tiny_llama --model-name tiny_llama
	$(PYTHON) tests/test_forward_correctness.py -v --timeout=30

# L2 基线
test-baseline: $(VENV)
	-$(PYTEST) tests/ -m baseline -v --tb=long --timeout=300
test-fast: lint test-unit test-model test-patterns test-pipeline-quick test-forward-smoke test-dylib-quick
test-all: lint test-unit test-model test-patterns test-integration

# L3: 性能
profile: $(VENV)
	@echo "=== LLM-ServeForge 性能基线 ==="
	@date +"%Y-%m-%d %H:%M:%S"
	@$(PYTHON) scripts/perf_regression.py --record 2>&1 || echo "perf_regression 失败"
profile-check: $(VENV)
	@echo "=== 性能回归检查 ==="
	@$(PYTHON) scripts/perf_regression.py --threshold 5 2>&1 || \
		(echo "❌ 性能退化 > 5%, 请优化后重新提交" && exit 1)

# L4: 冒烟
smoke: $(VENV)
	$(PYTEST) tests/test_smoke.py -m smoke -v --tb=short

# ---- 诊断 ----
diagnose-bt:
	lldb -b -o "run" -o "bt all" -o "quit" -- $(PYTHON) scripts/diagnostics/diagnose_lowering.py
diagnose:
	@echo "=== MECE Diagnosis Template ==="
	@echo "A: Data — 权重复制/加载错误"
	@echo "  A1: 权重值一致性 → make verify-consistency"
	@echo "  A2: 权重绑定顺序 → verify_bindings()"
	@echo "B: Compute — FP 精度/融合差异"
	@echo "  B1: FX Export diff → python scripts/checks/check_weights.py"
	@echo "  B2: C++ lowering → make test-fixup"
	@echo "C: Runtime — 调用约定/签名"
	@echo "  C1: 权重加载 → make verify-consistency"
	@echo "  C2: sret 读取 → diagnose_pe_function()"
	@echo "D: Type/Shape — MLIR 类型/形状推断错误 [NEW]"
	@echo "  D1: 检查 lowered IR 中 tensor<f32> (0D 张量泄漏)"
	@echo "  D2: 验证每个 sf op 的输出类型与语义一致"
	@echo "  D3: 检查动态维度 ? 在不应出现的位置"
	@echo "=== Usage ==="
	@echo "  Run each check above in order. Stop at first failure."

# ---- 清理 ----
clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist *.egg-info outputs/compiled/ .profile_baseline.txt
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
clean-logs:
	rm -rf outputs/logs/pipeline/stages/ outputs/logs/test/ outputs/logs/e2e/ outputs/logs/bisect/ outputs/logs/rebuild/

# (removed: test-dylib-cos-full alias — use test-dylib-cos directly)
