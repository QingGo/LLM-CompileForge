#!/usr/bin/env python3
"""Pre-built interestingness tests for reduce_mlir.py.

Each script is a standalone executable that can be used with
``--interestingness "python scripts/diagnostics/check_pass.py ..."``
or standalone.

Usage:
    # Check if a pass crashes/hangs on an MLIR file
    python scripts/diagnostics/check_pass.py convert-vector-to-llvm input.mlir

    # Check if a pass fails with a specific error pattern
    python scripts/diagnostics/check_pass.py sf-lower-to-linalg input.mlir \\
        --match "operand type mismatch"

    # With timeout
    python scripts/diagnostics/check_pass.py convert-vector-to-llvm input.mlir \\
        --timeout 30

Exit codes:
    0 = interesting (bug still present: pass crashed/hung/matched error)
    1 = not interesting (pass completed successfully or different error)
    2 = malformed input (MLIR can't be parsed)
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path


def _setup_mlir_context():
    """Set up MLIR context with all dialects registered."""
    import mlir.ir as ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    ctx.load_all_available_dialects()
    try:
        from mlir._mlir_libs import _mlirRegisterEverything

        reg = ir.DialectRegistry()
        _mlirRegisterEverything.register_dialects(reg)
        ctx.append_dialect_registry(reg)
    except Exception:
        pass
    return ctx


def check_pass(
    mlir_path: str,
    pass_pipeline: str,
    match_error: str | None = None,
    timeout: float = 30.0,
) -> int:
    """Run a pass pipeline on an MLIR file and check if the bug persists.

    Returns exit code:
        0 = pass crashed/hung/matched expected error (bug present)
        1 = pass completed normally (bug gone)
        2 = MLIR can't be parsed (invalid reduction)
    """
    import mlir.ir as ir
    import mlir.passmanager as pm

    ctx = _setup_mlir_context()

    try:
        with ir.Location.unknown(ctx):
            mod = ir.Module.parse(Path(mlir_path).read_text(), ctx)
    except Exception:
        return 2

    try:
        with ir.Location.unknown(ctx):
            p = pm.PassManager.parse(
                f"builtin.module({pass_pipeline})", ctx
            )
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(p.run, mod.operation)
                try:
                    future.result(timeout=timeout)
                except FutureTimeout:
                    return 0
    except Exception as exc:
        exc_text = str(exc)
        if match_error:
            if re.search(match_error, exc_text, re.IGNORECASE):
                return 0
            return 1
        return 0

    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-built MLIR pass interestingness test for reduce_mlir.py"
    )
    parser.add_argument(
        "pass_pipeline",
        help="MLIR pass pipeline to run (e.g., 'convert-vector-to-llvm')",
    )
    parser.add_argument(
        "mlir_file",
        help="Path to the MLIR file to test",
    )
    parser.add_argument(
        "--match", "-m",
        help="Only treat as interesting if the error matches this regex pattern",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=30.0,
        help="Timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert: exit 0 when pass SUCCEEDS (for checking bug fix)",
    )

    args = parser.parse_args()

    try:
        code = check_pass(
            args.mlir_file,
            args.pass_pipeline,
            args.match,
            args.timeout,
        )
    except Exception:
        traceback.print_exc(file=sys.stderr)
        code = 1

    if args.invert:
        code = 0 if code != 0 else 1

    return code


if __name__ == "__main__":
    sys.exit(main())
