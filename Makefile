.PHONY: lint lint-ruff lint-mypy test-unit test-integration test-fast test-all profile smoke clean

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

# ---- L2: 集成测试 (<80s, 可选) ----
test-integration: $(VENV)
	-$(PYTEST) tests/ -m integration -v --tb=short --timeout=300

# ---- 基线回归 (cosine >0.999) ----
test-baseline: $(VENV)
	-$(PYTEST) tests/ -m baseline -v --tb=long --timeout=300

# ---- 快速测试 (lint + unit, <15s) ----
test-fast: lint test-unit smoke

# ---- 全量测试 (all tests, <2min) ----
test-all: lint test-unit test-integration smoke

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

# ---- 工具 ----
clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache dist *.egg-info compiled/ .profile_baseline.txt
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
