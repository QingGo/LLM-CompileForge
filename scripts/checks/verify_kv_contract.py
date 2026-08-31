#!/usr/bin/env python3
"""KV-cache contract assertions derived from TRAPS.md K13-K23.

Each check is a greppable source-level invariant.  These do not replace the
hard cross-contract assertion executed at dylib load time
(runtime/src/cache/contract.rs); they guard the Python/Rust scheduler and
executor wiring that surrounds that assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

CHECKS: list[tuple[str, Path, list[str], str]] = [
    (
        "K13: LLMEngine wires scheduler use_kv_cache from executor capability",
        ROOT / "python_runtime/engine/llm_engine.py",
        [
            "use_kv_cache=getattr(self.executor, \"_uses_cache_manager\", False)",
        ],
        "line 110",
    ),
    (
        "K14: decode position is current_seq_len-1",
        ROOT / "runtime/src/engine/scheduler.rs",
        ["current_seq_len.saturating_sub(1)"],
        "decode branch",
    ),
    (
        "K15: decode grows block table before KV write/read",
        ROOT / "runtime/src/engine/scheduler.rs",
        ["block_manager.ensure_blocks(&req.request_id, target_tokens)"],
        "decode branch",
    ),
    (
        "K16: begin_step is called once per forward, not once per function",
        ROOT / "python_runtime/engine/mlir_executor.py",
        [
            "self._cache_mgr.begin_step(self._block_tables)",
        ],
        "_run_forward prologue",
    ),
    (
        "K17: executor filters intercepts by (func_index, output_index)",
        ROOT / "python_runtime/engine/mlir_executor.py",
        [
            "if len(idef.param_indices) >= 2:",
            "contract = (int(idef.param_indices[0]), int(idef.param_indices[1]))",
        ],
        "_handle_cache_intercept",
    ),
    (
        "K18: aggregation length is exact prefix, not block multiple",
        ROOT / "python_runtime/engine/mlir_executor.py",
        ["seq_len = int(positions.max().item()) + 1"],
        "KV read aggregation",
    ),
    (
        "K19: decode is defined as n_tokens == 1",
        ROOT / "python_runtime/engine/_inference_loop.py",
        ["is_decode=rd[\"n_tokens\"] == 1"],
        "batch request construction",
    ),
    (
        "K20: positions kwarg takes priority over arange fallback",
        ROOT / "python_runtime/engine/mlir_executor.py",
        [
            "positions_arg = kwargs.get(\"positions\", kwargs.get(\"position_ids\"))",
        ],
        "global input wiring",
    ),
    (
        "K21: intermediate prefill chunks are never sampled",
        ROOT / "runtime/src/engine/runner.rs",
        ["if req.state == RequestState::Prefill {\n                continue;"],
        "runner step",
    ),
    (
        "K22: cache-manager slab capacity is aligned to block pool",
        ROOT / "python_runtime/engine/llm_engine.py",
        [
            "if mgr is not None and getattr(mgr, \"_num_blocks\", 0) != cp.num_blocks:",
            "CacheManager(mgr._policy, num_blocks=cp.num_blocks)",
        ],
        "LLMEngine.__init__",
    ),
    (
        "K23: binding rebuild uses explicit venv maturin + PATH",
        ROOT / "Makefile",
        [
            "VIRTUAL_ENV=\"$(PROJECT_ROOT)/$(VENV)\"",
            "$(PROJECT_ROOT)/$(VENV)/bin/maturin",
        ],
        "build-rust target",
    ),
]


def main() -> None:
    failed = False
    for label, path, needles, where in CHECKS:
        if not path.exists():
            print(f"[FAIL] {label}: missing {path.relative_to(ROOT)}")
            failed = True
            continue
        text = path.read_text()
        missing = [n for n in needles if n not in text]
        if missing:
            print(f"[FAIL] {label} ({where})")
            for n in missing:
                print(f"       missing: {n!r}")
            failed = True
        else:
            print(f"[PASS] {label}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
