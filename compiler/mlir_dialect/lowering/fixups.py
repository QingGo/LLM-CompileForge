# ruff: noqa: E501 — long lines in MLIR fixup f-strings

"""Fixups for MLIR LLVM dialect IR — text-level and MLIR-bindings-based.

These functions perform transformations on LLVM dialect IR text,
replacing unrealized_conversion_cast chains with equivalent LLVM ops,
stripping version-incompatible flags, fixing scalar→tensor arith.constant
type mismatches, and lowering vector-typed arith ops that the pass
pipeline cannot handle.

Functions that require MLIR Python bindings fall back gracefully when
bindings are not available, preserving the string→string contract.
"""

from __future__ import annotations

import logging
import re
import typing as _typing

if _typing.TYPE_CHECKING:
    import mlir.ir as ir


def _strided_to_struct(memref_type: str) -> str:
    """Build the equivalent ``!llvm.struct<...>`` for a memref type."""
    m = re.match(r'memref<([^>]*?)(?:,|>)', memref_type)
    if not m:
        return '!llvm.struct<(ptr, ptr, i64)>'
    dims = m.group(1)
    rank = dims.count('x')
    sizes = f'array<{rank} x i64>' if rank else ''
    strides = f'array<{rank} x i64>' if rank else ''
    return (f'!llvm.struct<(ptr, ptr, i64, {sizes}, {strides})>'
            if rank else '!llvm.struct<(ptr, ptr, i64)>')


def _llvm_struct_rank(struct_type: str) -> int:
    """Extract the memref rank from an ``!llvm.struct<...>`` type string."""
    m = re.search(r'array<(\d+)\s*x\s*i64>', struct_type)
    return int(m.group(1)) if m else 0


def _fixup_unrealized_casts_pass(module: ir.Module) -> None:
    """Replace ``unrealized_conversion_cast`` with proper LLVM ops.

    Operates on the MLIR module in-place using official MLIR Python bindings.

    1. Entry cast:  ``!llvm.ptr, !llvm.ptr, i64, ... → memref<strided>``
    2. Exit cast:   ``memref<strided> → !llvm.struct<...>``
    3. Direct cast: ``!llvm.ptr, !llvm.ptr, i64, ... → !llvm.struct<...>``

    Each is replaced with ``llvm.mlir.undef`` + ``llvm.insertvalue`` chain.
    """
    import mlir.ir as ir

    ctx = module.operation.context
    with ir.Location.unknown(ctx):
        _fixup_unrealized_casts_pass_body(module)

def _fixup_unrealized_casts_pass_body(module: ir.Module) -> None:
    import mlir.ir as ir

    # Phase 1: classify all unrealized_conversion_cast ops in the module.
    # Use walk() to find casts nested inside llvm.func bodies (after
    # convert-func-to-llvm, casts are no longer at the top-level block).
    # Group by parent block to handle SSA-name reuse across functions.
    entry_casts: list[tuple[ir.Operation, ir.Block, ir.OpView]] = []
    exit_casts: list[tuple[ir.Operation, ir.Block, ir.OpView]] = []
    direct_casts: list[tuple[ir.Operation, ir.Block, ir.OpView]] = []
    all_casts: list[ir.Operation] = []

    def _classify_cast(op: ir.Operation) -> ir.WalkResult:
        if op.operation.name != "builtin.unrealized_conversion_cast":
            return ir.WalkResult.ADVANCE
        all_casts.append(op)
        operands = list(op.operands)
        result_type = str(op.result.type)

        # Entry: result is a memref type
        if result_type.startswith("memref<"):
            entry_casts.append((op, op.operation.parent, None))
        # Exit: single operand, source is memref, result is llvm struct
        elif (len(operands) == 1
              and str(operands[0].type).startswith("memref<")
              and result_type.startswith("!llvm.struct<")):
            exit_casts.append((op, op.operation.parent, None))
        # Direct: result is llvm struct
        elif result_type.startswith("!llvm.struct<"):
            direct_casts.append((op, op.operation.parent, None))
        return ir.WalkResult.ADVANCE

    module.operation.walk(_classify_cast)

    if not entry_casts and not exit_casts and not direct_casts:
        return

    _log = logging.getLogger("compiler.fixups")

    def _build_struct_chain(
        result_name: str,
        struct_type: ir.Type,
        operands: list[ir.Value],
        rank: int,
        block: ir.Block,
        ip: ir.InsertionPoint,
    ) -> ir.Value:
        """Build ``undef`` + ``insertvalue`` chain in *block* at *ip*.

        Returns the final ``!llvm.struct<…>`` value.
        """
        ctx = struct_type.context
        # 1. undef
        undef = ir.Operation.create(
            "llvm.mlir.undef",
            results=[struct_type],
            operands=[],
            ip=ip,
        )
        curr = undef.result

        # 2. insertvalue for each operand
        n_args = len(operands)
        for vi in range(n_args):
            if vi < 3:
                pos_attr = ir.DenseI64ArrayAttr.get([vi], context=ctx)
            else:
                arr_idx = vi - 3
                if arr_idx < rank:
                    pos_attr = ir.DenseI64ArrayAttr.get([3, arr_idx], context=ctx)
                else:
                    pos_attr = ir.DenseI64ArrayAttr.get([4, arr_idx - rank], context=ctx)
            inserted = ir.Operation.create(
                "llvm.insertvalue",
                results=[struct_type],
                operands=[curr, operands[vi]],
                attributes={"position": pos_attr},
                ip=ip,
            )
            curr = inserted.result
        return curr

    changes = 0

    # Phase 2: handle direct struct casts (simple case — no memref type involved)
    for cast_op, block, _ in direct_casts:
        operands = list(cast_op.operands)
        struct_type = cast_op.result.type
        rank = _struct_rank_from_type(struct_type)
        ip = ir.InsertionPoint(cast_op)
        curr = _build_struct_chain(
            f"fixup_{cast_op.result}",
            struct_type, operands, rank, block, ip,
        )
        cast_op.result.replace_all_uses_with(curr)
        cast_op.operation.erase()
        changes += 1

    # Phase 3: handle entry casts (build struct from bare ptr/size values)
    for cast_op, block, _ in entry_casts:
        operands = list(cast_op.operands)
        memref_type_str = str(cast_op.result.type)
        struct_type_str = _strided_to_struct(memref_type_str)
        struct_type = ir.Type.parse(struct_type_str, cast_op.operation.context)
        rank = _llvm_struct_rank(struct_type_str)
        ip = ir.InsertionPoint(cast_op)
        curr = _build_struct_chain(
            f"fixup_{cast_op.result}",
            struct_type, operands, rank, block, ip,
        )
        cast_op.result.replace_all_uses_with(curr)
        cast_op.operation.erase()
        changes += 1

    # Phase 4: handle exit casts (memref → struct — find matching entry)
    # An exit cast takes a memref-typed value (from an entry cast we just fixed)
    # and converts it to an llvm struct.  Since the entry cast now produces
    # an llvm struct, the exit cast is redundant — just forward uses.
    for cast_op, _block, _ in exit_casts:
        src = cast_op.operands[0]
        cast_op.result.replace_all_uses_with(src)
        cast_op.operation.erase()
        changes += 1

    total = len(entry_casts) + len(direct_casts) + len(exit_casts)
    _log.warning("MLIR pass removed %d / %d unrealized_conversion_cast ops",
                 changes, total)


def _struct_rank_from_type(struct_type: ir.Type) -> int:
    """Extract memref rank from an ``!llvm.struct<...>`` type by parsing
    its string representation."""
    s = str(struct_type)
    return _llvm_struct_rank(s)


def _fixup_arith_tensor_constants_mlir(mlir_text: str) -> str:
    """Phase 2: MLIR-bindings-based verification and fix of arith.constant ops.

    Parses the (already regex-fixed) IR, walks all ops in all regions, and
    converts any remaining ``arith.constant`` ops with scalar value + tensor
    result type to use ``DenseElementsAttr``.  Falls back to the input text
    if MLIR bindings are unavailable or parsing fails.
    """
    try:
        import mlir.ir as ir
    except ImportError:
        return mlir_text

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        try:
            module = ir.Module.parse(mlir_text, ctx)
        except Exception:
            # Even after regex fix, the IR might be invalid for other reasons
            return mlir_text

        _fix_count = _walk_and_fix_tensor_constants(module)

        if _fix_count:
            _log = logging.getLogger("compiler.fixups")
            _log.warning(
                "Fixed %d arith.constant ops with scalar→tensor type mismatch (MLIR)", _fix_count
            )

        # Always re-serialize via str(module) to ensure clean custom-format output.
        # This catches cases where the regex produced valid-but-ugly generic format.
        return str(module)


def _walk_and_fix_tensor_constants(module: ir.Module) -> int:
    """Walk all regions/blocks in ``module`` and fix scalar→tensor arith.constant ops.

    Uses the MLIR Python API (``ir.Operation.walk``) to visit every operation
    and convert ``FloatAttr``/``IntegerAttr`` values on ``arith.constant`` ops
    with tensor result types to ``DenseElementsAttr`` (splat).

    Returns the number of ops fixed.
    """
    import mlir.ir as ir  # noqa: F811 — late import inside function

    _fix_count = 0
    _candidates: list[ir.Operation] = []

    # Phase 2a: Collect candidates (avoid modifying IR while walking).
    # The walk callback must return WalkResult.ADVANCE to continue.
    _walk_result = ir.WalkResult

    def _collector(op: ir.Operation) -> ir.WalkResult:
        if str(op.operation.name) != "arith.constant":
            return _walk_result.ADVANCE

        _result_type = op.operation.result.type
        if not isinstance(_result_type, ir.ShapedType):
            return _walk_result.ADVANCE  # scalar result type — valid as-is

        _value_attr = op.operation.attributes.get("value")
        if _value_attr is None:
            return _walk_result.ADVANCE

        if isinstance(_value_attr, ir.DenseElementsAttr):
            return _walk_result.ADVANCE  # already correct

        if isinstance(_value_attr, (ir.FloatAttr, ir.IntegerAttr)):
            _candidates.append(op.operation)

        return _walk_result.ADVANCE

    module.operation.walk(_collector)

    # Phase 2b: Fix each candidate
    for _op in _candidates:
        _result_type = _op.result.type
        _value_attr = _op.attributes["value"]

        if isinstance(_value_attr, ir.DenseElementsAttr):
            continue  # already correct

        if not isinstance(_value_attr, (ir.FloatAttr, ir.IntegerAttr)):
            continue  # unsupported scalar attribute type

        # Build splat dense attribute
        try:
            _new_attr = ir.DenseElementsAttr.get_splat(_result_type, _value_attr)
            _op.attributes["value"] = _new_attr
            _fix_count += 1
        except Exception:
            _log = logging.getLogger("compiler.fixups")
            _log.debug("Failed to fix arith.constant at %s", _op.location, exc_info=True)

    return _fix_count

