"""EmitRust backend: orchestrate Rust code generation from HAL IR.

This module provides the ``emit_rust()`` orchestrator that reads HAL IR JSON
and dispatches per-op templates from ``OP_IMPLS`` to generate a complete
``hal_ops_cpu.rs`` file.
"""

from __future__ import annotations

import json
import logging
import os

__all__ = ["emit_rust"]

_log = logging.getLogger(__name__)


def emit_rust(
    hal_ir_path: str,
    output_path: str,
) -> str:
    """Generate hal_ops_cpu.rs from HAL IR JSON.

    Args:
        hal_ir_path: Path to the HAL IR JSON file.
        output_path: Desired output path for the generated Rust source.

    Returns:
        The output_path (absolute, normalized).
    """
    _log.warning(
        "emit_rust is a stub. HAL IR at %s, output would go to %s",
        hal_ir_path,
        output_path,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(hal_ir_path, encoding="utf-8") as fh:
        hal_ir: object = json.load(fh)

    # hal_ir is expected to be a dict with an "ops" list.
    assert isinstance(hal_ir, dict)
    ops: object = hal_ir.get("ops", [])
    assert isinstance(ops, list)

    op_types = sorted({
        op.get("op", "unknown")
        for op in ops
        if isinstance(op, dict)
    })

    header = "// Auto-generated stub by emit_rust — replace with real codegen.\n\n"
    body_lines: list[str] = [header]
    body_lines.append("// Op types found in HAL IR:\n")
    for ot in op_types:
        body_lines.append(f"//   {ot}\n")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.writelines(body_lines)

    _log.info("Stub written to %s (%d ops)", output_path, len(ops))
    return os.path.abspath(output_path)
