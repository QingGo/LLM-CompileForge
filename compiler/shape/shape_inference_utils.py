"""Type utilities and broadcasting helpers for shape inference.

Extracted from shape_inference.py to isolate the foundational helpers.
Used by both the MLIR-level and pure-Python shape inference modules.
"""

from __future__ import annotations

from typing import Any

import mlir.ir as ir

# ── Type helpers ─────────────────────────────────────────────


def _elt_type_str(tp: ir.Type) -> str:
    """Extract element type string from MLIR tensor type."""
    if isinstance(tp, (ir.RankedTensorType, ir.UnrankedTensorType)):
        et = str(tp.element_type)
        for s in ["f32", "f64", "f16", "bf16", "i32", "i64", "i8", "i1", "ui8"]:
            if s in et:
                return s
    return "f32"


def _ranked_shape(tp: ir.Type) -> tuple[int | None, ...] | None:
    """Extract shape tuple from MLIR tensor type. None for unranked."""
    if isinstance(tp, ir.RankedTensorType):
        return tuple(int(d) if d >= 0 else None for d in tp.shape)
    return None


_ELT_TYPE_MAP: dict[str, ir.Type] | None = None


def _get_elt_type_map() -> dict[str, ir.Type]:
    """Return shared str→ir.Type mapping, lazily initialized."""
    global _ELT_TYPE_MAP
    if _ELT_TYPE_MAP is None:
        _ELT_TYPE_MAP = {
            "f32": ir.F32Type.get(),
            "f64": ir.F64Type.get(),
            "f16": ir.F16Type.get(),
            "bf16": ir.BF16Type.get(),
            "i8": ir.IntegerType.get_signless(8),
            "i32": ir.IntegerType.get_signless(32),
            "i64": ir.IntegerType.get_signless(64),
            "bool": ir.IntegerType.get_signless(1),
        }
    return _ELT_TYPE_MAP


def _elt_from_str(s: str) -> ir.Type:
    return _get_elt_type_map().get(s, ir.F32Type.get())


def _make_ranked_type(shape: tuple[int | None, ...], elt: str) -> ir.RankedTensorType:
    return ir.RankedTensorType.get(
        [-1 if d is None else d for d in shape],
        _elt_from_str(elt),
    )


def _infer_ir_via_pure(
    pure_fn: Any,
    input_types: list[ir.Type],
    **kwargs: Any,
) -> list[ir.Type]:
    """Unify infer_* / _infer_*_pure: extract ir.Type → tuples → call _pure → re-wrap.

    Eliminates the duplicate shape computation logic between ``infer_*`` and
    ``_infer_*_pure`` function pairs.  The _pure functions operate on plain
    Python tuples ``(shape, element_type_str)`` which are simpler to test.
    """
    if not input_types:
        return []
    shapes = [_ranked_shape(t) for t in input_types]
    elts = [_elt_type_str(t) for t in input_types]
    result = pure_fn(shapes, elts, **kwargs)
    return [_make_ranked_type(s, e) for s, e in result]


# ── Broadcasting helpers ─────────────────────────────────────


def _broadcast_shapes(*shapes: tuple[int | None, ...]) -> tuple[int | None, ...]:
    """NumPy/MLIR-style broadcasting: align right, size-1 broadcasts to any.

    Each dim must be 1, equal, or one can be None (dynamic, which is compatible
    with anything).  Returns the broadcasted shape.
    """
    if not shapes:
        return ()
    max_rank = max(len(s) for s in shapes)
    result: list[int | None] = []
    for i in range(max_rank):
        dims = []
        for s in shapes:
            idx = len(s) - max_rank + i
            dims.append(s[idx] if idx >= 0 else 1)
        non_one = [d for d in dims if d is not None and d != 1]
        any_dynamic = any(d is None for d in dims)
        if not non_one:
            result.append(None if any_dynamic else dims[0])
        elif len(non_one) == 1:
            result.append(non_one[0])
        else:
            if len(set(non_one)) > 1:
                raise ValueError(f"Incompatible dims for broadcast: {dims}")
            result.append(non_one[0])
    return tuple(result)


def _broadcast_types(*types: ir.Type) -> tuple[int | None, ...]:
    """Broadcast shapes from MLIR tensor types, returning a shape tuple."""
    shapes: list[tuple[int | None, ...]] = []
    for t in types:
        s = _ranked_shape(t)
        if s is None:
            return (None,) * max(len(shapes) + 1, 1)
        shapes.append(s)
    return _broadcast_shapes(*shapes)
