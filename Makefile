.PHONY: lint lint-ruff lint-mypy test-unit test-integration test-fast test-all test-model test-patterns test-smoke profile smoke clean clean-logs diagnose-bt test-fixup test-ctypes-oracle test-pipeline-smoke test-rust test-rust-unit test-rust-integ test-pipeline-quick test-changed test-pipeline-timing test-pipeline-debug test-pipeline-validate test-vec test-lower test-baseline test-compile-full test-forward-smoke test-weight-consistency verify-dylib build-rust install-rust diagnose-fast diagnose-ci

# ---- 环境 ----
VENV := .venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest

$(VENV):
	uv venv --python 3.10 && uv sync

# ---- DYLD library paths ----
MLIR_LIBS_PATH := $(PWD)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs
TORCH_LIB_PATH := $(PWD)/$(VENV)/lib/python3.10/site-packages/torch/lib

# ---- L0: 静态检查 (<2s) ----
lint: lint-ruff lint-mypy

lint-ruff:
	$(RUFF) check hal/ compiler/ engine/ server/ tests/

lint-mypy:
	$(MYPY) hal/ compiler/ engine/ server/ --config-file pyproject.toml

# ---- L1: 单元测试 (<1s each, suite <10s) ----
test-unit: $(VENV)
	$(PYTEST) tests/ -m unit -v --tb=short --timeout=1

# ---- L1b: 模型产物质量检查 (<5s) ----
test-model: $(VENV)
	$(PYTHON) -m pytest tests/test_model_artifact.py -v --tb=short --timeout=10

# ---- L1c: 单个 lowering pattern 测试 (<10s) ----
test-patterns: $(VENV)
	$(PYTHON) -m pytest tests/test_lowering_patterns.py -v --tb=short --timeout=2

# ---- L1d: 组合快速验证 (lint + 全部 L1, <20s) ----
test-smoke: lint test-unit test-model test-patterns

# ---- L2: 集成测试 (<80s, 可选) ----
test-integration: $(VENV)
	-$(PYTEST) tests/ -m integration -v --tb=short --timeout=300

# ---- 基线回归 (cosine >0.999) ----
test-baseline: $(VENV)
	-$(PYTEST) tests/ -m baseline -v --tb=long --timeout=300

# ---- L1e: 向量化管线测试 (<30s) ----
test-vec: $(VENV)
	$(PYTHON) scripts/test_transform_vec.py
	$(PYTHON) -m pytest tests/test_pipeline_bugs.py -v --tb=short --timeout=60

# ---- L1g: Lowering 诊断 (每个 op 类型单独测试, <10s) ----
test-lower: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" \
	$(PYTHON) scripts/test_lowering_diag.py

# ---- 快速测试 (lint + L1 all, <20s) ----
test-fast: lint test-unit test-model test-patterns test-pipeline-quick test-forward-smoke

# ---- L1.5: 快速正确性检查 (<5s) ----
test-forward-smoke: $(VENV)
	mkdir -p logs/test
	@for model in compiled/opt_125m_fresh compiled/tiny_llama; do \
		if [ -d $$model ]; then \
			echo "[$$model]"; \
			LLM_SERVEFORGE_LOG=INFO $(PYTHON) scripts/check_forward_smoke.py $$model > logs/test/forward_smoke_$$(date +%Y%m%d_%H%M%S).log 2>&1 || exit 1; \
		fi; \
	done

# ---- 全量测试 (all tests, <2min) ----
test-all: lint test-unit test-model test-patterns test-integration

# ---- L3: 性能回归 (<5min) ----
profile: $(VENV)
	@echo "=== LLM-ServeForge 性能基线 ==="
	@date +"%Y-%m-%d %H:%M:%S"
	@$(PYTHON) scripts/perf_regression.py --record 2>&1 || echo "perf_regression 失败"
	@echo "提示: 功能里程碑后运行 make profile 记录关键指标"

# ---- L3b: 性能回归门禁 ----
profile-check: $(VENV)
	@echo "=== 性能回归检查 ==="
	@$(PYTHON) scripts/perf_regression.py --threshold 5 2>&1 || \
		(echo "❌ 性能退化 > 5%, 请优化后重新提交" && exit 1)

# ---- L4: 冒烟测试 (<2s) ----
smoke: $(VENV)
	$(PYTEST) tests/test_smoke.py -m smoke -v --tb=short

# ---- 诊断：lldb backtrace（C++ assertion 崩溃时用） ----
diagnose-bt:
	lldb -b -o "run" -o "bt all" -o "quit" -- $(PYTHON) scripts/diagnose_lowering.py

# ---- L1i: Pipeline 时序诊断 (每步单独计时+超时, <120s) ----
test-pipeline-timing: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" \
	$(PYTHON) scripts/pipeline_timing.py compiled/opt_125m_fresh

# ---- L1i: Pipeline 逐 pass 落盘 debug (中间文件保存+计时) ----
test-pipeline-debug: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" \
	$(PYTHON) scripts/pipeline_debug.py compiled/opt_125m_fresh --out /tmp/pipeline_debug

# ---- L1j: _fixup_unrealized_casts 单元测试 (16 patterns, <1s) ----
test-fixup: $(VENV)
	$(PYTEST) tests/test_fixup_casts.py -v --tb=short --timeout=10

# ---- L1m: ctypes oracle e2e 测试 (dylib vs Python executor, <30s) ----
test-ctypes-oracle: $(VENV)
	$(PYTEST) tests/test_ctypes_oracle.py -v --tb=short --timeout=30

# ---- L1k: Pipeline 冒烟测试 (全流程限时验证, <120s) ----
test-pipeline-smoke: $(VENV)
	$(PYTHON) scripts/test_pipeline_smoke.py compiled/opt_125m_fresh --timeout 120

# ---- L1l: Pipeline 验证 (IR 清洁度 + 分步计时 + 最终编译, <60s) ----
test-pipeline-validate: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" \
	$(PYTHON) -m pytest tests/test_pipeline_validation.py -v --tb=short --timeout=60

# ---- L1n: Pipeline 快速验证 (仅 IR 解析 + op 类型检查, <2s) ----
test-pipeline-quick: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" \
	$(PYTEST) tests/test_pipeline_validation.py::test_no_arith_ops_after_lowering \
	         tests/test_pipeline_validation.py::test_tile_sizes_within_bounds \
	         -v --tb=short --timeout=30

# ---- L1m: Rust 单元测试 (纯逻辑, ~5s) ----
test-rust-unit: $(VENV)
	mkdir -p logs/rust
	cd rust && cargo test --lib > ../logs/rust/test_unit_$$(date +%Y%m%d_%H%M%S).log 2>&1

# ---- L2b: Rust 集成测试 (含 .dylib 加载, ~30s) ----
test-rust-integ: $(VENV)
	mkdir -p logs/rust
	cd rust && cargo test --bin serveforge > ../logs/rust/test_integ_$$(date +%Y%m%d_%H%M%S).log 2>&1

# ---- L1m: Rust 测试 (全部 99+, <40s) ----
test-rust: test-rust-unit test-rust-integ

# ---- Rust 构建 & 安装到 venv (用于 Python FFI) ----
build-rust: $(VENV)
	cd rust && cargo build --release --features python-bindings 2>&1

install-rust: build-rust
	@mkdir -p logs/rust
	@echo "  ⚠️  检测环境冲突..." >&2
	@if [ -n "$$CONDA_PREFIX" ] && [ -n "$$VIRTUAL_ENV" ]; then \
		echo "  ⚠️  同时检测到 CONDA_PREFIX 和 VIRTUAL_ENV，尝试 unset CONDA_PREFIX..."; \
		CONDA_PREFIX="" maturin develop -r --manifest-path rust/Cargo.toml 2>&1 || \
		(echo "  ❌ maturin 失败 — 手动运行: cd rust && VIRTUAL_ENV=$$(pwd)/.venv PATH=$$(pwd)/.venv/bin:$$PATH CONDA_PREFIX= maturin develop -r" >&2 && exit 1); \
	else \
		maturin develop -r --manifest-path rust/Cargo.toml > logs/rust/install_$$(date +%Y%m%d_%H%M%S).log 2>&1; \
	fi
	@echo "  ✅ Rust 模块已安装到 venv"

# ---- 增量测试: 仅跑 git diff 相关测试 (<30s) ----
test-changed: $(VENV)
	@if [ -f scripts/run_related_tests.py ]; then \
		$(PYTHON) scripts/run_related_tests.py; \
	else \
		echo "⚠️  scripts/run_related_tests.py 不存在，回退到 test-fast"; \
		$(MAKE) test-fast; \
	fi

# ---- L1n: 权重一致性测试 (三路验证: GT/Python/Rust, <30s) ----
test-weight-consistency: $(VENV)
	$(PYTEST) tests/test_weight_consistency.py -v --tb=short --timeout=120 -m "not slow"

# ---- L1o: .dylib vs compute graph vs lowered IR 一致性检查 (<5s) ----
verify-dylib:
	$(PYTHON) scripts/verify_dylib_consistency.py compiled/opt_125m_fresh

# ---- L2a: 全模型编译测试 (逐步骤检测, <300s) ----
test-compile-full: $(VENV)
	DYLD_LIBRARY_PATH="$(MLIR_LIBS_PATH)" \
	$(PYTHON) -m pytest tests/test_compile_full.py -v --tb=short --timeout=300

# ---- SF dialect extension rebuild ----
.PHONY: rebuild-sf
rebuild-sf:
	bash scripts/build_sf_extension.sh

# ---- 工具 ----
clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist *.egg-info compiled/ .profile_baseline.txt
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-logs:
	rm -rf logs/pipeline/stages/ logs/test/ logs/e2e/ logs/bisect/ logs/rebuild/

# ---- L5: 全量清洁重建 (清空 + 编译 + 推理 + 精度对比) ----
.PHONY: rebuild rebuild-clean rebuild-mlir rebuild-dylib rebuild-test rebuild-cosine

# 全量重建: 清空 → 导出 model.mlir → 编译 .dylib → forward 测试 → cosine 对比
rebuild: rebuild-clean rebuild-mlir rebuild-dylib rebuild-test rebuild-cosine

# 清空编译产物 (保留其他模型)
rebuild-clean:
	rm -rf compiled/opt_125m_fresh

# 从 PyTorch 导出模型 → model.mlir
rebuild-mlir: $(VENV)
	$(PYTHON) scripts/compile.py opt-125m --output-dir ./compiled/opt_125m_fresh

# model.mlir → .dylib (C++ lowering + LLVM pipeline + llc + link)
rebuild-dylib: $(VENV)
	DYLD_LIBRARY_PATH="$(TORCH_LIB_PATH):$(MLIR_LIBS_PATH)" \
	$(PYTHON) scripts/compile_dylib.py compiled/opt_125m_fresh --model-name opt_125m

# Rust forward 测试
rebuild-test:
	cd rust && cargo test test_opt_125m_forward_runs -- --nocapture 2>&1 | tail -5

# Cosine 精度对比 (Rust vs Python vs HF)
rebuild-cosine: $(VENV)
	$(PYTHON) scripts/diagnose_issue45.py 2>&1 | grep -E "Cosine|DIAGNOSIS|argmax"

# 快速诊断循环 (使用已编译的 .dylib + 预计算的 logits)
diagnose-fast:
	DUMP_LAYERS=/tmp/ld_rs cargo run --manifest-path rust/Cargo.toml --bin forward_check 2>&1 | tail -3
	$(PYTHON) scripts/diagnose_issue45.py --fast 2>&1 | grep -E "Cosine|argmax|→"

# CI gate for cosine regression
diagnose-ci:
	$(PYTHON) scripts/diagnose_issue45.py --fast --ci --threshold 0.99 2>&1 | grep -E "CI|Cosine|argmax"
