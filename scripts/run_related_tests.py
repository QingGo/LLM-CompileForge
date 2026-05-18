#!/usr/bin/env python3
"""Run tests related to files changed in the working tree.

Usage:
    python scripts/run_related_tests.py
    python scripts/run_related_tests.py --verbose

Uses ``git diff --name-only HEAD`` to find changed files, then
selects tests based on a file → test mapping.  Falls back to
``make test-fast`` if no specific tests are found.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── File → test mapping ───────────────────────────────────────────────
# Key: glob pattern matched against relative file paths
# Value: list of test commands (pytest args or make targets)

TEST_MAP: dict[str, list[str]] = {
    "compiler/mlir_dialect/fixups.py": [
        "tests/test_fixup_casts.py",
    ],
    "compiler/mlir_dialect/pipeline_stages.py": [
        "-k pipeline tests/test_pipeline_validation.py",
        "tests/test_pipeline_bugs.py",
    ],
    "compiler/mlir_dialect/compile_utils.py": [
        "tests/test_llvm_backend.py",
        "-k pipeline tests/test_pipeline_validation.py",
    ],
    "compiler/mlir_dialect/llvm_backend.py": [
        "tests/test_llvm_backend.py",
        "-k pipeline tests/test_pipeline_validation.py",
        "tests/test_fixup_casts.py",
    ],
    "compiler/mlir_dialect/lowering.py": [
        "tests/test_lowering_patterns.py",
        "-k pipeline tests/test_pipeline_validation.py",
    ],
    "compiler/mlir_dialect/sf.py": [
        "tests/test_mlir_dialect.py",
    ],
    "compiler/mlir_dialect/shape_inference.py": [
        "tests/test_mlir_dialect.py",
    ],
    "compiler/mlir_dialect/builder.py": [
        "tests/test_mlir_dialect.py",
    ],
    "compiler/fx_to_mlir.py": [
        "tests/test_fx_to_mlir.py",
    ],
    "compiler/pipeline.py": [
        "-k pipeline tests/test_pipeline_validation.py",
        "tests/test_pipeline_lowering.py",
    ],
    "compiler/mlir_artifact.py": [
        "tests/test_mlir_artifact.py",
        "tests/test_model_artifact.py",
    ],
    "compiler/serialize.py": [
        "tests/test_mlir_artifact.py",
    ],
    "hal/*.py": [
        "tests/test_hal.py",
        "tests/test_pytorch_backend.py",
    ],
    "engine/*.py": [
        "tests/test_llm_engine.py",
        "tests/test_mlir_executor.py",
    ],
    "rust/src/*.rs": [
        ("make", "test-rust-unit"),
    ],
    "rust/src/executor.rs": [
        ("make", "test-rust-unit"),
        ("make", "test-rust-integ"),
    ],
    "rust/src/weight_loader.rs": [
        ("make", "test-rust-unit"),
    ],
    "rust/src/hal_cpu.rs": [
        ("make", "test-rust-unit"),
    ],
    "rust/src/m1_tests.rs": [
        ("make", "test-rust-integ"),
    ],
}


def _match(pattern: str, filepath: str) -> bool:
    """Simple glob matching (supports ``*`` only)."""
    import fnmatch
    return fnmatch.fnmatch(filepath, pattern)


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Get changed files
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        capture_output=True, text=True,
        timeout=10,
    )
    if result.returncode != 0:
        # Try unstaged diff
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True, text=True,
            timeout=10,
        )
    changed = [f for f in result.stdout.splitlines() if f.strip()]

    if not changed:
        print("No changed files found. Running full test-fast.")
        os.execvp("make", ["make", "test-fast"])

    if verbose:
        print(f"Changed files ({len(changed)}):")
        for f in changed:
            print(f"  {f}")

    # Collect matching tests
    test_args: list[str] = []
    make_targets: list[str] = []

    for filepath in changed:
        for pattern, commands in TEST_MAP.items():
            if _match(pattern, filepath):
                for cmd in commands:
                    if isinstance(cmd, tuple) and cmd[0] == "make":
                        target = cmd[1]
                        if target not in make_targets:
                            make_targets.append(target)
                    else:
                        if cmd not in test_args:
                            test_args.append(cmd)

    if not test_args and not make_targets:
        print("No specific tests matched. Running make test-fast.")
        os.execvp("make", ["make", "test-fast"])

    # Run make targets first
    for target in make_targets:
        print(f"\n{'='*60}")
        print(f"Running: make {target}")
        print(f"{'='*60}")
        ret = subprocess.run(
            ["make", target],
            timeout=120,
        )
        if ret.returncode != 0:
            print(f"❌ make {target} failed")
            sys.exit(1)

    # Run pytest commands
    if test_args:
        pytest_bin = ".venv/bin/pytest"
        cmd = [pytest_bin, "-v", "--tb=short", *test_args]
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*60}")
        ret = subprocess.run(cmd, timeout=120)
        if ret.returncode != 0:
            print(f"❌ pytest failed")
            sys.exit(1)

    print("\n✅ All related tests passed")


if __name__ == "__main__":
    main()
