"""EmitRust backend — generate hal_ops_cpu.rs from hal_ir.json.

Reads a compiled model's ``hal_ir.json``, collects all unique op types,
and emits a self-contained Rust source file with naive scalar CPU
implementations for each op type.

Usage::

    from compiler.mlir_dialect.hal_ir.emit_rust import emit_rust

    path = emit_rust("compiled/opt_125m_kv/hal_ir.json")
    print(f"Generated: {path}")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from compiler.mlir_dialect.hal_ir.op_implementations import (
    BLAS_EXTERN,
    HEADER,
    OP_SHAPE_META,
    STUB_CACHE,
)
from compiler.mlir_dialect.hal_ir.op_implementations import (
    OP_IMPLS as _OP_IMPLS,
)

# Re-export for downstream consumers (e.g. compile_hal.py)
OP_IMPLS = _OP_IMPLS

_log = logging.getLogger(__name__)


# ── Main generation function ────────────────────────────────────────────


def emit_rust(
    hal_ir_path: str | Path,
    output_path: str | Path | None = None,
) -> str:
    """Generate ``hal_ops_cpu.rs`` from ``hal_ir.json``.

    Reads the HAL IR JSON, collects all unique op types used in the model,
    and emits a self-contained Rust source file with CPU implementations
    for each op type.

    Args:
        hal_ir_path: Path to ``hal_ir.json``.
        output_path: Where to write the Rust file. If ``None``, writes to
            the same directory as ``hal_ir.json`` with name ``hal_ops_cpu.rs``.

    Returns:
        Absolute path to the written Rust file.
    """
    hal_ir_path = Path(hal_ir_path)
    if not hal_ir_path.exists():
        raise FileNotFoundError(f"hal_ir.json not found: {hal_ir_path}")

    data = json.loads(hal_ir_path.read_text())
    model_name: str = data.get("model_name", "unknown")
    functions: list[dict[str, Any]] = data.get("functions", [])

    # Collect unique op types in use
    op_types_in_use: set[str] = set()
    for fn in functions:
        for op in fn.get("ops", []):
            op_types_in_use.add(op["op"])

    _log.info(
        "EmitRust: model=%s functions=%d unique_ops=%s",
        model_name,
        len(functions),
        sorted(op_types_in_use),
    )

    # ── Assemble Rust source ──────────────────────────────────────────

    parts: list[str] = []

    # Header
    parts.append(HEADER.format(model_name=model_name))

    # Struct
    parts.append(OP_SHAPE_META)

    # BLAS extern
    if "matmul" in op_types_in_use:
        parts.append(BLAS_EXTERN)

    # Stubs (always emitted — harmless if unused)
    parts.append(STUB_CACHE)

    # Per-op implementations
    for op_type in sorted(op_types_in_use):
        if op_type in ("cache_read", "cache_write"):
            continue  # handled by STUB_CACHE above
        impl = OP_IMPLS.get(op_type)
        if impl is None:
            _log.warning("EmitRust: no implementation for op type '%s' — skipping", op_type)
            continue
        parts.append(impl)

    rust_source = "\n".join(parts)

    # ── Write output ──────────────────────────────────────────────────

    if output_path is None:
        output_path = hal_ir_path.parent / "hal_ops_cpu.rs"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rust_source)

    _log.info("EmitRust: wrote %s (%d bytes)", output_path, len(rust_source))
    return str(output_path.resolve())


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m compiler.mlir_dialect.hal_ir.emit_rust <hal_ir.json> [output.rs]")
        sys.exit(1)

    hal_ir_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    path = emit_rust(hal_ir_path, output_path)
    print(f"Generated: {path}")
