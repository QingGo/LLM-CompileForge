.PHONY: lint lint-ruff lint-mypy test-unit test-integration test-fast test-all test-model test-patterns test-smoke profile smoke clean diagnose-bt test-fixup test-pipeline-smoke test-rust test-rust-unit test-rust-integ test-pipeline-quick test-changed

# ---- 环境 ----
VENV := .venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
PYTEST := $(VENV)/bin/pytest

$(VENV):
	uv venv --python 3.10 && uv sync

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
	DYLD_LIBRARY_PATH="$(PWD)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs" \
	$(PYTHON) scripts/test_lowering_diag.py

# ---- L1f: 管线性能回归 (<60s) ----
test-perf: $(VENV)
	$(PYTHON) scripts/perf_regression.py --save .perf_baseline.json

# ---- 快速测试 (lint + L1 all, <20s) ----
test-fast: lint test-unit test-model test-patterns test-pipeline-quick

# ---- 全量测试 (all tests, <2min) ----
test-all: lint test-unit test-model test-patterns test-integration

# ---- L3: 性能回归 (<5min) ----
profile: $(VENV)
	@echo "=== LLM-ServeForge 性能基线 ===" > .profile_baseline.txt
	@date >> .profile_baseline.txt
	@echo "模型加载耗时:" >> .profile_baseline.txt
	@$(PYTHON) -c "import time; t=time.perf_counter(); import torch; print(f'{time.perf_counter()-t:.2f}s')" >> .profile_baseline.txt 2>&1 || echo "torch 导入失败(非 GPU 环境)" >> .profile_baseline.txt
	@echo "---" >> .profile_baseline.txt
	@cat .profile_baseline.txt
	@echo "提示: 功能里程碑后运行 make profile 记录关键指标"

# ---- L4: 冒烟测试 (<2s) ----
smoke: $(VENV)
	$(PYTEST) tests/test_smoke.py -m smoke -v --tb=short

# ---- 诊断：lldb backtrace（C++ assertion 崩溃时用） ----
diagnose-bt:
	lldb -b -o "run" -o "bt all" -o "quit" -- $(PYTHON) scripts/diagnose_lowering.py

# ---- Benchmark (Phase 2.5 Sprint 1) ----
benchmark: $(VENV)
	$(PYTHON) scripts/benchmark.py --all --output benchmark_results.json

test-benchmark: $(VENV)
	$(PYTEST) tests/test_benchmark.py -m benchmark -v --tb=short --timeout=120

# ---- L1h: Pipeline 冒烟测试 (<10s) ----
test-pipeline: $(VENV)
	DYLD_LIBRARY_PATH="$(PWD)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs" \
	$(PYTHON) scripts/smoke_pipeline.py

# ---- L1i: Pipeline 时序诊断 (每步单独计时+超时, <120s) ----
test-pipeline-timing: $(VENV)
	DYLD_LIBRARY_PATH="$(PWD)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs" \
	$(PYTHON) scripts/pipeline_timing.py compiled/opt_125m_fresh

# ---- L1i: Pipeline 逐 pass 落盘 debug (中间文件保存+计时) ----
test-pipeline-debug: $(VENV)
	DYLD_LIBRARY_PATH="$(PWD)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs" \
	$(PYTHON) scripts/pipeline_debug.py compiled/opt_125m_fresh --out /tmp/pipeline_debug

# ---- L1j: _fixup_unrealized_casts 单元测试 (16 patterns, <1s) ----
test-fixup: $(VENV)
	$(PYTEST) tests/test_fixup_casts.py -v --tb=short --timeout=10

# ---- L1k: Pipeline 冒烟测试 (全流程限时验证, <120s) ----
test-pipeline-smoke: $(VENV)
	$(PYTHON) scripts/test_pipeline_smoke.py compiled/opt_125m_fresh --timeout 120

# ---- L1l: Pipeline 验证 (IR 清洁度 + 分步计时 + 最终编译, <60s) ----
test-pipeline-validate: $(VENV)
	DYLD_LIBRARY_PATH="$(PWD)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs" \
	$(PYTHON) -m pytest tests/test_pipeline_validation.py -v --tb=short --timeout=60

# ---- L1n: Pipeline 快速验证 (仅 IR 解析 + op 类型检查, <2s) ----
test-pipeline-quick: $(VENV)
	DYLD_LIBRARY_PATH="$(PWD)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs" \
	$(PYTEST) tests/test_pipeline_validation.py::test_no_arith_ops_after_lowering \
	         tests/test_pipeline_validation.py::test_tile_sizes_within_bounds \
	         -v --tb=short --timeout=30

# ---- L1m: Rust 单元测试 (纯逻辑, ~5s) ----
test-rust-unit: $(VENV)
	cd rust && cargo test --lib 2>&1

# ---- L2b: Rust 集成测试 (含 .dylib 加载, ~30s) ----
test-rust-integ: $(VENV)
	cd rust && cargo test --bin serveforge 2>&1

# ---- L1m: Rust 测试 (全部 99+, <40s) ----
test-rust: test-rust-unit test-rust-integ

# ---- 增量测试: 仅跑 git diff 相关测试 (<30s) ----
test-changed: $(VENV)
	@if [ -f scripts/run_related_tests.py ]; then \
		$(PYTHON) scripts/run_related_tests.py; \
	else \
		echo "⚠️  scripts/run_related_tests.py 不存在，回退到 test-fast"; \
		$(MAKE) test-fast; \
	fi

# ---- L2a: 全模型编译测试 (逐步骤检测, <300s) ----
test-compile-full: $(VENV)
	DYLD_LIBRARY_PATH="$(PWD)/llvm-project/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs" \
	$(PYTHON) -m pytest tests/test_compile_full.py -v --tb=short --timeout=300

# ---- SF dialect extension rebuild ----
.PHONY: rebuild-sf
rebuild-sf:
	bash scripts/build_sf_extension.sh

# ---- 工具 ----
clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist *.egg-info compiled/ .profile_baseline.txt
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
