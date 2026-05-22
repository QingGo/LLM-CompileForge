from __future__ import annotations

import logging
import warnings
from typing import Any

import torch
import torch.fx

from compiler.mlir_dialect._op_defs import _ATEN_TO_HAL
from compiler.mlir_dialect.shape_inference import infer_output_shape

_log = logging.getLogger(__name__)


def _symint_to_int(val: Any) -> int | None:
    if isinstance(val, torch.SymInt):
        if hasattr(val, "node") and val.node is not None:
            hint = getattr(val.node, "hint", None)
            if hint is not None:
                result = int(hint)
                # MLIR kDynamic sentinel (INT64_MAX) means dynamic dimension
                if result == 9223372036854775807:
                    return None
                return result
        return None
    if isinstance(val, int):
        return None if val == 9223372036854775807 else val
    if isinstance(val, str):
        return None
    try:
        result = int(val)
        return None if result == 9223372036854775807 else result
    except (TypeError, ValueError):
        return None


def _symint_for_view(val: Any) -> int:
    concrete = _symint_to_int(val)
    if concrete is not None:
        return concrete
    return -1


def _resolve_shape_tuple(raw_shape: Any) -> tuple[int | None, ...]:
    result: list[int | None] = []
    for d in raw_shape:
        result.append(_symint_to_int(d))
    return tuple(result)


def _parse_mlir_type_to_shape(type_str: str) -> tuple[tuple[int | None, ...], str]:
    """Parse MLIR type string like 'tensor<1x64xf32>' → ((1,64), 'f32')."""
    if not type_str.startswith("tensor<"):
        return ((1,), "f32")
    inner = type_str[len("tensor<"):-1]  # remove tensor<...>
    parts = inner.split("x")
    elt = parts[-1]
    shape: list[int | None] = []
    for p in parts[:-1]:
        shape.append(None if p == "?" else int(p))
    return (tuple(shape), elt)


def _resolve_op_types(
    hal_op: str,
    input_names: list[str],
    ssa_map: dict[str, str],
    shape_map: dict[str, tuple[tuple[int | None, ...], str]],
    weights: dict[str, torch.Tensor],
    kwargs: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Compute input/output MLIR type strings for an sf operation."""

    input_shapes: list[tuple[int | None, ...]] = []
    input_elts: list[str] = []

    for inp_name in input_names:
        # Resolve operand to original node name for shape lookup.
        # Exact match first, then longest-prefix as fallback to avoid
        # false resolver matches (e.g. '%reshape_5' matching 'reshape'
        # before 'reshape_5').
        resolved: str | None = None
        for node_name, ssa_name in ssa_map.items():
            if ssa_name == inp_name or node_name == inp_name:
                resolved = node_name
                break
        if resolved is None:
            candidates = sorted(
                (n for n in ssa_map if inp_name.startswith(f"%{n}")),
                key=len, reverse=True,
            )
            resolved = candidates[0] if candidates else inp_name

        if resolved in shape_map:
            s, e = shape_map[resolved]
        elif inp_name in weights:
            t = weights[inp_name]
            s = tuple(t.shape) if len(t.shape) > 0 else (1,)
            e = _fake_to_shape_tuple(t)[1]
        elif inp_name.startswith("%") and inp_name[1:] in weights:
            t = weights[inp_name[1:]]
            s = tuple(t.shape) if len(t.shape) > 0 else (1,)
            e = _fake_to_shape_tuple(t)[1]
        else:
            warnings.warn(
                f"Shape not found for operand {inp_name!r} in op {hal_op!r}, "
                f"using fallback shape (2, 64)",
                stacklevel=2,
            )
            s = (2, 64)
            e = "f32"
        input_shapes.append(s)
        input_elts.append(e)

    # Coerce non-float element types to f32 BEFORE inference, so shape inference
    # receives the correct element type (e.g. scalar i64 → f32 for float ops).
    float_ops = {"add", "mul", "sub", "div", "max", "le", "logical_and",
                   "linear", "matmul", "layer_norm", "rms_norm",
                   "relu", "gelu", "silu", "sigmoid", "exp", "neg", "tanh",
                   "identity", "sum", "mean", "softmax",
                   "transpose", "slice", "ones_like", "cumsum"}
    if hal_op in float_ops:
        for i in range(len(input_elts)):
            if input_elts[i] not in ("f32", "f16", "bf16", "f64"):
                input_elts[i] = "f32"
                if i < len(input_names) and input_names[i] in weights:
                    weights[input_names[i]] = weights[input_names[i]].float()

    try:
        out = infer_output_shape(hal_op, input_shapes, input_elts, **kwargs)
    except (ValueError, TypeError, NotImplementedError) as e:
        _log.warning(
            "shape inference fallback for op=%s shapes=%s elts=%s: %s",
            hal_op, input_shapes, input_elts, e,
        )
        if input_elts:
            out = [(input_shapes[0], input_elts[0])]
        else:
            out = [((1,), "f32")]

    in_type_strs = [_shape_to_mlir_type(s, e) for s, e in zip(input_shapes, input_elts, strict=False)]
    out_type_strs = [_shape_to_mlir_type(s, e) for s, e in out]
    return in_type_strs, out_type_strs


def _map_aten_op(target: Any) -> str | None:
    if isinstance(target, str):
        target_str = target
    elif hasattr(target, "name"):
        target_str = str(target)
    elif hasattr(target, "__name__"):
        target_str = target.__name__
    else:
        target_str = str(target)
    target_str = target_str.replace("::", ".")
    if target_str in _ATEN_TO_HAL:
        return _ATEN_TO_HAL[target_str]
    if "." in target_str:
        parts = target_str.rsplit(".", 1)
        overload_candidates = {"default", "int", "float", "str", "bool", "complex",
                               "Scalar", "ScalarList", "Tensor", "dimname", "layout",
                               "device", "memory_format", "generator", "dim", "start",
                               "other", "dtype", "dtype_layout", "values", "copy"}
        if len(parts) == 2 and parts[1] in overload_candidates:
            base = parts[0]
            if base in _ATEN_TO_HAL:
                return _ATEN_TO_HAL[base]
    return None


def _extract_node_kwargs(node: torch.fx.Node) -> dict[str, Any]:
    return dict(node.kwargs)


def _dtype_to_mlir(dtype: str) -> str:
    mapping = {
        "float32": "f32", "float16": "f16", "bfloat16": "bf16",
        "float64": "f64", "int32": "i32", "int64": "i64",
        "int8": "i8", "uint8": "ui8", "bool": "i1",
    }
    return mapping.get(dtype, "f32")


def _tensor_type_str(dtype: str, shape: tuple[int | None, ...]) -> str:
    dims = "x".join(str(d) if d is not None else "?" for d in shape)
    elt = _dtype_to_mlir(dtype)
    return f"tensor<{dims}x{elt}>" if dims else f"tensor<{elt}>"


def _type_from_fake(fake: torch.Tensor) -> str:
    shape = _resolve_shape_tuple(fake.shape)
    dtype = str(fake.dtype).replace("torch.", "")
    return _tensor_type_str(dtype, shape)


def _fake_to_shape_tuple(fake: torch.Tensor) -> tuple[tuple[int | None, ...], str]:
    """Extract (shape, elt_str) from a fake tensor for shape inference."""
    shape = _resolve_shape_tuple(fake.shape)
    elt = str(fake.dtype).replace("torch.", "")
    return shape, elt


def _shape_to_mlir_type(shape: tuple, elt: str) -> str:
    """Convert (shape, element_type) to MLIR type string like tensor<1x64xf32>."""
    def _dim_str(d: Any) -> str:
        if d is None:
            return "?"
        try:
            return str(int(d)) if int(d) > 0 else "?"
        except (TypeError, ValueError):
            return "?"
    dims = "x".join(_dim_str(d) for d in shape)
    elt_map = {
        "float32": "f32", "float16": "f16", "bfloat16": "bf16",
        "float64": "f64", "int32": "i32", "int64": "i64",
        "int8": "i8", "uint8": "ui8", "bool": "i1",
    }
    mlir_elt = elt_map.get(elt, "f32")
    return f"tensor<{dims}x{mlir_elt}>" if dims else f"tensor<{mlir_elt}>"
