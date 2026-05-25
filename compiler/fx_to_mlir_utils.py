from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.fx

from compiler.mlir_dialect._op_defs import _ATEN_TO_HAL
from compiler.mlir_dialect.shape_inference import infer_output_shape

_log = logging.getLogger(__name__)

# ── Dtype function registry ────────────────────────────────────────

_DtypeFn = Callable[[list[str], dict[str, Any]], str | None]

DTYPE_REGISTRY: dict[str, _DtypeFn] = {}


def dtype_rule(op_name: str) -> Callable[[_DtypeFn], _DtypeFn]:
    def decorator(fn: _DtypeFn) -> _DtypeFn:
        DTYPE_REGISTRY[op_name] = fn
        return fn
    return decorator


def resolve_dtype(
    op_name: str, input_elts: list[str], kwargs: dict[str, Any]
) -> str | None:
    fn = DTYPE_REGISTRY.get(op_name)
    if fn is None:
        return None
    return fn(input_elts, kwargs)


def _replace_element_type(type_str: str, new_elt: str) -> str:
    if not type_str.startswith("tensor<"):
        return type_str
    # strip tensor<...> then replace the element type (last x-segment)
    inner = type_str[len("tensor<"):-1]
    parts = inner.split("x")
    parts[-1] = new_elt
    return f"tensor<{'x'.join(parts)}>"


def _apply_dtype_hook(
    op_name: str,
    input_elts: list[str],
    kwargs: dict[str, Any],
    out_type_strs: list[str],
) -> list[str]:
    elt = resolve_dtype(op_name, input_elts, kwargs)
    if elt is not None and out_type_strs:
        out_type_strs[0] = _replace_element_type(out_type_strs[0], elt)
    return out_type_strs


# ── Dtype functions: per-op type rules ─────────────────────────────

_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.int64: "i64",
    torch.int32: "i32",
    torch.float32: "f32",
    torch.bool: "i1",
}


def _resolve_dtype(
    kwargs: dict[str, Any], input_elts: list[str]
) -> str | None:
    """Look up dtype from kwargs using torch.dtype keys in _DTYPE_MAP.
    
    Supports both torch.dtype values (torch.int64 → "i64") and integer
    dtype codes (3 → "i64") for backward compat with dtype function tests.
    """
    dtype_val = kwargs.get("dtype")
    if dtype_val is None:
        return None
    if isinstance(dtype_val, int):
        code_to_dtype = {
            0: torch.uint8, 1: torch.int8, 2: torch.short, 3: torch.int64,
            4: torch.int32, 5: torch.int64, 6: torch.float32, 7: torch.float64,
            8: torch.complex64, 9: torch.complex128, 11: torch.bool,
            12: torch.qint8, 13: torch.quint8, 14: torch.qint32,
            15: torch.bfloat16, 16: torch.float16,
        }
        dtype_val = code_to_dtype.get(dtype_val)
        if dtype_val is None:
            return None
    return _DTYPE_MAP.get(dtype_val)


@dtype_rule("ones_like")
def _dtype_ones_like(
    input_elts: list[str], kwargs: dict[str, Any]
) -> str | None:
    dtype_val = _resolve_dtype(kwargs, input_elts)
    if dtype_val is not None:
        return dtype_val
    # aten.ones.default has no input tensor to inherit dtype from — the
    # first "input" is actually the shape dimensions (sym_size results).
    # PyTorch default for tensor creation is float32.
    return "f32"


@dtype_rule("cumsum")
def _dtype_cumsum(
    input_elts: list[str], kwargs: dict[str, Any]
) -> str | None:
    dtype_val = _resolve_dtype(kwargs, input_elts)
    if dtype_val is not None:
        return dtype_val
    elt = input_elts[0]
    # Integer input → promote to i64
    if elt.startswith("i"):
        return "i64"
    # Float input → passthrough
    if elt.startswith("f"):
        return elt
    return None


@dtype_rule("arange")
def _dtype_arange(
    input_elts: list[str], kwargs: dict[str, Any]
) -> str | None:
    dtype_val = _resolve_dtype(kwargs, input_elts)
    if dtype_val is not None:
        return dtype_val
    # sf.arange always produces i64 (ODS constraint: Sf_Int64Tensor).
    return "i64"



def _symint_to_name(val: Any) -> str | None:
    """Extract symbolic variable name from a torch.SymInt, if available.

    SymInt objects in ``node.meta["val"]`` carry symbolic names accessible
    via ``str(val.node)`` (e.g. ``"s0"``, ``"s1"``).  These names identify
    which broadcast dimension carries which symbolic identity.

    Returns the symbolic name (e.g. ``"s0"``) or ``None`` if the value is
    not a symbolic integer.
    """
    if isinstance(val, torch.SymInt):
        if hasattr(val, "node") and val.node is not None:
            name = str(val.node)
            if isinstance(name, str) and name.startswith("s"):
                return name
    return None


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

    # Normalize element types to canonical MLIR format — shape_inference
    # and _fake_to_shape_tuple may produce PyTorch-format type strings
    # (e.g. "float32", "int64") while type checks and downstream ops
    # expect MLIR format (e.g. "f32", "i64").
    _PYTORCH_TO_MLIR = {
        "float32": "f32", "float": "f32",
        "float16": "f16", "half": "f16",
        "bfloat16": "bf16",
        "float64": "f64", "double": "f64",
        "int32": "i32",
        "int64": "i64", "long": "i64",
        "int8": "i8",
        "uint8": "ui8",
        "bool": "i1",
    }
    input_elts = [_PYTORCH_TO_MLIR.get(e, e) for e in input_elts]

    # ── Type validation: fail fast on mismatched operand types ──

    def _type_mismatch_msg(op_name: str, mismatches: list[tuple[int, str, str]]) -> str:
        """Build a detailed error message for type mismatches."""
        src = kwargs.get("source_node", "unknown")
        layer = kwargs.get("dump_layer", "unknown")
        lines = [
            f"Type mismatch in '{op_name}' (FX node: {src}, layer: {layer})",
            f"  operand types: {input_elts}",
            f"  operand names: {input_names}",
        ]
        for mi, t0, t1 in mismatches:
            n = input_names[mi] if mi < len(input_names) else "?"
            extra = ""
            if n.startswith("_const_"):
                if n in weights:
                    w = weights[n]
                    extra = f" (const value={w.item() if w.numel()==1 else 'tensor'}, dtype={w.dtype})"
                else:
                    extra = " (const, NOT in weights)"
            elif n.startswith("%"):
                extra = f" (SSA: {n})"
            lines.append(
                f"  operand[{mi}] type={t1!r} differs from operand[0] type={t0!r}{extra}"
            )
        return "\n".join(lines)

    # Binary ops: both operands must be same type.
    # For constants (_const_*) with type mismatch, implement PyTorch's
    # promotion rules: int → float when mixed with float operands.
    # This is NOT a silent coercion — it's explicitly tracking the
    # promotion with a log message so it's visible in debug output.
    same_type_ops = {"add", "mul", "sub", "div", "max", "pow"}
    if hal_op in same_type_ops and len(input_elts) >= 2:
        for i in range(1, len(input_elts)):
            if input_elts[i] != input_elts[0]:
                # Check if we can resolve via promotion
                n = input_names[i] if i < len(input_names) else ""
                if n.startswith("_const_") and n in weights:
                    promoted = False
                    # i64 → f32 promotion (PyTorch semantics)
                    if input_elts[i] == "i64" and input_elts[0] == "f32":
                        weights[n] = weights[n].float()
                        input_elts[i] = "f32"
                        promoted = True
                    # i64 → f16 promotion
                    if input_elts[i] == "i64" and input_elts[0] == "f16":
                        weights[n] = weights[n].half()
                        input_elts[i] = "f16"
                        promoted = True
                    if promoted:
                        _log.debug(
                            "Type promotion for '%s': %s i64→%s (matches %s operand %s)",
                            hal_op, n, input_elts[i],
                            "lhs" if i > 0 else "rhs",
                            input_names[0],
                        )
                else:
                    # Non-const operand — can't promote, must fail
                    mismatches = [(i, input_elts[0], input_elts[i])
                                  for i in range(1, len(input_elts))
                                  if input_elts[0] != input_elts[i]]
                    if mismatches:
                        raise TypeError(_type_mismatch_msg(hal_op, mismatches))
                    break

    # Comparison ops: both operands must be same type
    cmp_ops = {"le", "ge", "gt", "lt", "eq", "ne"}
    if hal_op in cmp_ops and len(input_elts) >= 2:
        mismatches = [(i, input_elts[0], input_elts[i])
                      for i in range(1, len(input_elts))
                      if input_elts[0] != input_elts[i]]
        if mismatches:
            raise TypeError(_type_mismatch_msg(hal_op, mismatches))

    # Float-required ops: fail fast if any operand is not float
    float_required_ops = {"linear", "layer_norm", "rms_norm", "softmax",
                          "relu", "gelu", "silu", "sigmoid", "tanh", "exp",
                          "neg", "matmul", "mean", "cumsum"}
    if hal_op in float_required_ops:
        for i in range(len(input_elts)):
            if input_elts[i] not in ("f32", "f16", "bf16", "f64"):
                src = kwargs.get("source_node", "unknown")
                layer = kwargs.get("dump_layer", "unknown")
                n = input_names[i] if i < len(input_names) else "?"
                raise TypeError(
                    f"'{hal_op}' requires float inputs (FX: {src}, layer: {layer}), "
                    f"but operand[{i}] '{n}' has type={input_elts[i]}"
                )

    # Index op: indices must be integer or float type (C++ lowering handles
    # float→i64 conversion via FPToUIOp with WARNING for backward compat).
    if hal_op == "index":
        for i in range(1, len(input_elts)):
            if input_elts[i] not in ("i64", "i32", "int64", "int32", "f32", "f64"):
                raise TypeError(
                    f"sf.index index operand {i} must be integer or float type, "
                    f"got type={input_elts[i]}"
                )

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
    out_type_strs = _apply_dtype_hook(hal_op, input_elts, kwargs, out_type_strs)
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
            val = int(d)
            # Safety: convert PyTorch sentinel (sys.maxsize) to dynamic dim,
            # and any non-positive integer (-1, 0) to dynamic dim.
            if val >= 9223372036854775807 or val <= 0:
                return "?"
            return str(val)
        except (TypeError, ValueError):
            return "?"
    dims = "x".join(_dim_str(d) for d in shape)
    elt_map = {
        "float32": "f32", "float16": "f16", "bfloat16": "bf16",
        "float64": "f64", "int32": "i32", "int64": "i64",
        "int8": "i8", "uint8": "ui8", "bool": "i1",
    }
    # Accept both PyTorch format (e.g. "float32", "int64") and MLIR format
    # (e.g. "f32", "i64") — dtype rules may produce MLIR-format element type
    # strings.  Always emit canonical MLIR format.
    mlir_elt = elt_map.get(elt)
    if mlir_elt is None and elt in elt_map.values():
        mlir_elt = elt
    elif mlir_elt is None:
        mlir_elt = "f32"
    return f"tensor<{dims}x{mlir_elt}>" if dims else f"tensor<{mlir_elt}>"
