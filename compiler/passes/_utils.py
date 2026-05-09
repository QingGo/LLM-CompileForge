"""Shared utilities for compiler passes."""

from __future__ import annotations

from compiler.ir import IrOp


def find_consumer(
    output_name: str,
    op_name: str,
    producer_map: dict[str, IrOp],
) -> IrOp | None:
    """Find an IrOp that consumes *output_name* and has the given *op_name*."""
    for op in producer_map.values():
        if output_name in op.inputs and op.name == op_name:
            return op
    return None


def find_consumer_in_list(
    ops: list[IrOp],
    start: int,
    value: str,
    op_name: str,
) -> IrOp | None:
    """Find the first op after *start* that consumes *value* and has name *op_name*."""
    for i in range(start, len(ops)):
        op = ops[i]
        if value in op.inputs and op.name == op_name:
            return op
    return None


def find_producer(
    output_name: str,
    op_name: str,
    producer_map: dict[str, IrOp],
) -> IrOp | None:
    """Find the op that produces *output_name* and has name *op_name*."""
    if output_name in producer_map:
        op = producer_map[output_name]
        if op.name == op_name:
            return op
    return None
