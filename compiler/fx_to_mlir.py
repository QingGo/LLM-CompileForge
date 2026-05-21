"""FX Graph → MlirModule conversion.

Replaces the old two-step pipeline (fx_to_ir → IrModule → mlir_emitter → model.mlir)
with a single step that produces an MlirModule directly.  The MlirModule is the
canonical representation consumed by MlirExecutor and serialized to model.mlir.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import torch
import torch.fx
from torch.export import ExportedProgram

from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp
from compiler.mlir_dialect._op_defs import (
    _ATEN_TO_HAL,
    _LIST_ARG_ATTR,
    _SCALAR_INT_POSITIONS,
    _SCALAR_KWARG_NAMES,
)
from compiler.mlir_dialect.shape_inference import infer_output_shape

_log = logging.getLogger(__name__)


# ── Utility helpers ─────────────────────────────────────────

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
    import warnings

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
            s = tuple(t.shape)
            e = _fake_to_shape_tuple(t)[1]
        elif inp_name.startswith("%") and inp_name[1:] in weights:
            t = weights[inp_name[1:]]
            s = tuple(t.shape)
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


# ── Dimension-position attribute names ─────────────────────
# These encode axis/dim indices that must be adjusted when tensor
# rank changes at a function boundary during per-function splitting.

_DIM_ATTR_NAMES: set[str] = {
    "dim", "dim0", "dim1", "dimensions", "axis",
}

_DIM_LIST_ATTR_NAMES: set[str] = {
    "dims",
}


def _adjust_op_attributes(op: MlirOp, input_rank_map: dict[str, int]) -> MlirOp:
    """Clamp dim/axis attributes that are out-of-bounds for the op's
    declared input ranks at a function boundary.

    When the full CFG is split into per-function chunks, some ops
    retain ``dim``-like attributes from the original higher-rank
    context.  This clamps such values into the valid ``[0, rank-1]``
    range so the op can execute in isolation.
    """
    if op.op_name in ("weight", "constant", "_func_boundary"):
        return op

    new_attrs = dict(op.attributes)

    for _i, operand in enumerate(op.operands):
        key = operand.lstrip("%")
        rank = input_rank_map.get(key)
        if rank is None or rank <= 0:
            continue

        for attr_name in _DIM_ATTR_NAMES & set(new_attrs.keys()):
            val = new_attrs[attr_name]
            if isinstance(val, int) and val >= rank:
                new_attrs[attr_name] = rank - 1

        for attr_name in _DIM_LIST_ATTR_NAMES & set(new_attrs.keys()):
            vals = new_attrs[attr_name]
            if isinstance(vals, (list, tuple)):
                new_attrs[attr_name] = tuple(
                    (rank - 1) if isinstance(v, int) and v >= rank else v
                    for v in vals
                )

    return MlirOp(
        name=op.name, dialect=op.dialect, op_name=op.op_name,
        operands=list(op.operands), results=list(op.results),
        attributes=new_attrs,
        input_types=list(op.input_types), output_types=list(op.output_types),
    )


def _split_into_functions(
    mlir_ops: list[MlirOp],
    func_inputs: list[tuple[str, str]],
    weights: dict[str, torch.Tensor],
    ops_per_func: int = 500,
    boundaries: list[int] | None = None,
) -> tuple[list[MlirOp], int]:
    """Insert sentinel ops marking function boundaries.

    If *boundaries* is provided (non-empty), insert sentinels at those
    indices (layer-aware splitting). Otherwise, insert a sentinel every
    *ops_per_func* ops (op-count splitting, existing behaviour).

    Returns (augmented_ops, num_functions).
    """
    if boundaries:
        # Layer-based: insert sentinel before each boundary op
        result: list[MlirOp] = []
        boundary_set = set(boundaries)
        for i, op in enumerate(mlir_ops):
            if i in boundary_set:
                result.append(MlirOp(
                    name="_sentinel", dialect="_sentinel", op_name="_func_boundary",
                    operands=[], results=[], attributes={},
                    input_types=[], output_types=[],
                ))
            result.append(op)
        return result, len(boundaries) + 1

    # Op-count splitting (existing behaviour)
    if len(mlir_ops) <= ops_per_func:
        return mlir_ops, 1

    result: list[MlirOp] = []
    for i, op in enumerate(mlir_ops):
        if i > 0 and i % ops_per_func == 0:
            result.append(MlirOp(
                name="_sentinel", dialect="_sentinel", op_name="_func_boundary",
                operands=[], results=[], attributes={},
                input_types=[], output_types=[],
            ))
        result.append(op)
    return result, (len(mlir_ops) + ops_per_func - 1) // ops_per_func


# ── Layer detection helpers ─────────────────────────────


def _detect_layer(nn_module_stack: dict, prev_layer: str) -> str:
    """Extract layer identifier from ``nn_module_stack`` dict.

    Returns one of:
      - ``"embed_prefix"`` — decoder/embedding prefix (before any layer)
      - ``"layer_{N}"``    — transformer layer ``N``
      - ``"output"``       — output region (final norm + lm_head)
    """
    if not nn_module_stack:
        return prev_layer

    # Look for decoder layer numbers in stack keys
    layer_num: int | None = None
    for key in nn_module_stack:
        m = re.search(r'layers\.(\d+)', str(key))
        if m:
            num = int(m.group(1))
            if layer_num is None or num > layer_num:
                layer_num = num

    if layer_num is not None:
        return f"layer_{layer_num}"

    # Check for output-module indicators
    key_str_lower = " ".join(str(k).lower() for k in nn_module_stack)
    if any(kw in key_str_lower for kw in ['lm_head', 'final_norm', 'final_layer_norm']):
        return "output"

    # If we've seen layers before and now no layer key → output region
    if prev_layer and prev_layer.startswith("layer_"):
        return "output"

    return "embed_prefix"


def _log_split_plan(mlir_ops: list, boundaries: list[int]) -> None:
    """Log layer-based split plan to stderr."""
    segments: list[tuple[str, int]] = []
    prev = 0
    for b in sorted(boundaries):
        if b <= prev:
            continue
        layer_name = mlir_ops[prev].attributes.get("dump_layer", "?")
        segments.append((layer_name, b - prev))
        prev = b
    if prev < len(mlir_ops):
        layer_name = mlir_ops[prev].attributes.get("dump_layer", "?")
        segments.append((layer_name, len(mlir_ops) - prev))

    print(
        f"[fx_to_mlir] Layer-based split: {len(segments)} functions detected",
        file=sys.stderr,
    )
    for i, (name, count) in enumerate(segments):
        print(f"[fx_to_mlir]   func_{i}: {name} — {count} ops", file=sys.stderr)


def _make_multi_functions(
    mlir_ops: list[MlirOp],
    global_inputs: list[tuple[str, str]],
    global_outputs: list[tuple[str, str]],
    weights: dict[str, torch.Tensor],
    param_names: set[str],
    const_names: set[str],
    base_name: str,
) -> list[MlirFunction]:
    """Split mlir_ops at sentinel boundaries into separate MlirFunction objects."""

    # Remove sentinel ops and partition into blocks
    blocks: list[list[MlirOp]] = []
    current: list[MlirOp] = []
    for op in mlir_ops:
        if op.op_name == "_func_boundary":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(op)
    if current:
        blocks.append(current)

    if len(blocks) <= 1:
        return [MlirFunction(
            name=base_name, inputs=global_inputs, outputs=global_outputs,
            ops=mlir_ops, weights=weights,
            param_weight_names=param_names, const_weight_names=const_names,
        )]

    # Build type map: SSA name → MLIR type string (normalize % prefix)
    type_map: dict[str, str] = {}
    for op in mlir_ops:
        for idx, r in enumerate(op.results):
            key = r.lstrip("%")
            if idx < len(op.output_types):
                type_map[key] = op.output_types[idx]
            else:
                type_map[key] = "tensor<f32>"

    # Track which function produces each result (normalize % prefix)
    producer: dict[str, int] = {}
    for fi, block in enumerate(blocks):
        for op in block:
            for r in op.results:
                producer[r.lstrip("%")] = fi

    # Track weight references per function (normalize % prefix)
    weight_refs_per_func: list[set[str]] = [set() for _ in blocks]
    for fi, block in enumerate(blocks):
        for op in block:
            for operand in op.operands:
                key = operand.lstrip("%")
                if key in weights:
                    weight_refs_per_func[fi].add(key)

    funcs: list[MlirFunction] = []

    for fi, block in enumerate(blocks):
        # External inputs needed by this block (normalize %)
        needed: set[str] = set()
        for op in block:
            for operand in op.operands:
                key = operand.lstrip("%")
                if key in producer and producer[key] != fi:
                    needed.add(key)

        # Function inputs with proper types
        f_inputs: list[tuple[str, str]]
        if fi == 0:
            f_inputs = list(global_inputs)
        else:
            f_inputs = []
            # Build producer order map: SSA name → (func_idx, op_sequence_number)
            # op_sequence_number is the index of the producing op within its function
            # block, used to preserve the original computation order across functions.
            prod_order: dict[str, tuple[int, int]] = {}
            for fii, block_i in enumerate(blocks):
                for oi, op in enumerate(block_i):
                    for r in op.results:
                        prod_order[r.lstrip("%")] = (fii, oi)

            # Sort inputs to match ciface calling convention:
            #   Non-weight from func[0] first (sorted by func[0] output index),
            #   then cross-function values (from func[fi-1] etc.),
            #   then weights from func[0] last.
            # Order inputs to match ciface calling convention:
            # 1. Non-weight values from func[0] (scalar, hidden_state, mask)
            # 2. Cross-function values (hidden state from func[fi-1])
            # 3. Weight values from func[0]
            # 4. sym_size scalars from func[0]
            # Order inputs to match func[1]'s proven-correct convention:
            # [scalar, hidden_state, mask, weights..., sym_size...]
            nw_0 = [(v, prod_order.get(v, (999, 999))) for v in needed
                    if prod_order.get(v, (999,))[0] == 0 and v not in weights]
            xfunc = [(v, prod_order.get(v, (999, 999))) for v in needed
                     if prod_order.get(v, (999,))[0] != 0 and v not in weights]
            w_0 = [(v, (999, 999)) for v in needed if v in weights]
            # First entry: earliest non-weight from func[0] (the scalar)
            nw_sorted = sorted(nw_0, key=lambda x: x[1][1])
            if nw_sorted:
                f_inputs.append((f"%{nw_sorted[0][0]}", type_map.get(nw_sorted[0][0], "tensor<f32>")))
                nw_sorted = nw_sorted[1:]
            # Then cross-function values (hidden state from func[fi-1])
            for val, _ in sorted(xfunc, key=lambda x: x[1]):
                f_inputs.append((f"%{val}", type_map.get(val, "tensor<f32>")))
            # Then remaining non-weight from func[0] (mask, etc.)
            for val, _ in nw_sorted:
                if not type_map.get(val, "").startswith("tensor<1x"):
                    f_inputs.append((f"%{val}", type_map.get(val, "tensor<f32>")))
            # Then weights
            for val, _ in sorted(w_0, key=lambda x: x[0]):
                f_inputs.append((f"%{val}", type_map.get(val, "tensor<f32>")))
            # Finally sym_size scalars
            for val, _ in nw_sorted:
                if type_map.get(val, "").startswith("tensor<1x"):
                    f_inputs.append((f"%{val}", type_map.get(val, "tensor<f32>")))

        # Values produced here that are consumed elsewhere (normalize %)
        produced_here: set[str] = set()
        for op in block:
            for r in op.results:
                produced_here.add(r.lstrip("%"))

        exported: set[str] = set()
        for val in produced_here:
            for fi2 in range(fi + 1, len(blocks)):
                for op2 in blocks[fi2]:
                    for operand in op2.operands:
                        if operand.lstrip("%") == val:
                            exported.add(val)
                            break

        # Include global outputs
        for name, _ in global_outputs:
            if name.lstrip("%") in produced_here:
                exported.add(name.lstrip("%"))

        f_outputs = [(f"%{v}", type_map.get(v, "tensor<f32>")) for v in sorted(exported)]
        if not f_outputs:
            f_outputs = [(f"%{list(produced_here)[0]}", type_map.get(list(produced_here)[0], "tensor<f32>"))]

        # Compute rank of each function input from its type string
        input_rank_map: dict[str, int] = {}
        for name, tp in f_inputs:
            rank = 1 if "x" not in tp and "tensor<" in tp else tp.count("x")
            input_rank_map[name.lstrip("%")] = max(rank, 1) if "tensor<" in tp else 0

        adjusted_block = [_adjust_op_attributes(op, input_rank_map) for op in block]

        funcs.append(MlirFunction(
            name=f"{base_name}_{fi}",
            inputs=f_inputs,
            outputs=f_outputs,
            ops=adjusted_block,
            weights={k: v for k, v in weights.items() if k in weight_refs_per_func[fi]},
            param_weight_names=param_names & weight_refs_per_func[fi],
            const_weight_names=const_names & weight_refs_per_func[fi],
        ))

    return funcs


def fx_graph_to_mlir(
    program: ExportedProgram,
    function_name: str = "main",
    split_strategy: str = "layer",
) -> MlirModule:
    gm = program.graph_module
    graph = gm.graph
    state_dict = program.state_dict
    sig = program.graph_signature

    # ── Phase 1: function inputs ──────────────────────────
    func_inputs: list[tuple[str, str]] = []
    for inp_name in sig.user_inputs:
        node = next((n for n in graph.nodes if n.name == inp_name), None)
        if node is not None and "val" in node.meta:
            tp = _type_from_fake(node.meta["val"])
        else:
            tp = "tensor<f32>"
        func_inputs.append((f"%{inp_name}", tp))

    # ── Phase 2: weights ──────────────────────────────────
    weight_name_map: dict[str, str] = {}
    if hasattr(sig, "input_specs"):
        for spec in sig.input_specs:
            if spec.kind.value in (2, 3, 4):
                if spec.target:
                    weight_name_map[spec.arg.name] = spec.target.replace(".", "_")

    weights: dict[str, torch.Tensor] = {}
    hf_key_map: dict[str, str] = {}  # clean name → original HF safetensors key
    param_names: set[str] = set()
    const_names: set[str] = set()
    for name, tensor in state_dict.items():
        clean = name.replace(".", "_")
        weights[clean] = tensor
        hf_key_map[clean] = name  # original HF key
        param_names.add(clean)
    if hasattr(program, "constants"):
        for name, tensor in program.constants.items():
            clean = name.replace(".", "_")
            if clean not in weights:
                weights[clean] = tensor
            const_names.add(clean)

    # ── Phase 3: walk operations ──────────────────────────
    mlir_ops: list[MlirOp] = []
    func_outputs: list[tuple[str, str]] = []
    name_counter = 0
    ssa_map: dict[str, str] = {}
    tuple_outputs: dict[str, list[str]] = {}
    # Shape tracking: SSA name → (shape_tuple, element_type_str)
    shape_map: dict[str, tuple[tuple[int | None, ...], str]] = {}
    # Layer tracking for split_strategy="layer"
    current_layer: str = "embed_prefix"
    layer_boundaries: list[int] = []
    node_layer_map: dict[str, str] = {}

    for node in graph.nodes:
        if node.op == "placeholder":
            ssa_map[node.name] = f"%{weight_name_map.get(node.name, node.name)}"
            if "val" in node.meta:
                shape_map[node.name] = _fake_to_shape_tuple(node.meta["val"])
            continue

        if node.op == "get_attr":
            attr_name = str(node.target).replace(".", "_")
            ssa_map[node.name] = f"%{attr_name}"
            if "val" in node.meta:
                shape_map[node.name] = _fake_to_shape_tuple(node.meta["val"])
            continue

        if node.op == "call_function":
            hal_op = _map_aten_op(node.target)
            if hal_op is None:
                continue

            # Layer detection (split_strategy="layer")
            nn_stack = node.meta.get("nn_module_stack", {})
            if nn_stack:
                new_layer = _detect_layer(nn_stack, current_layer)
                if new_layer != current_layer:
                    layer_boundaries.append(len(mlir_ops))
                    current_layer = new_layer
            node_layer_map[node.name] = current_layer

            if hal_op == "_skip_wrap":
                ssa_map[node.name] = func_inputs[0][0] if func_inputs else node.name
                continue

            skwargs = _SCALAR_KWARG_NAMES.get(hal_op, {})
            sipos = _SCALAR_INT_POSITIONS.get(hal_op, [])
            laa = _LIST_ARG_ATTR.get(hal_op, "__skip__")

            if hal_op == "getitem":
                _do_getitem(node, ssa_map, weights, tuple_outputs, mlir_ops)
                continue

            if hal_op == "split":
                _do_split(node, ssa_map, tuple_outputs, mlir_ops)
                continue

            if hal_op == "chunk":
                _do_chunk(node, ssa_map, tuple_outputs, mlir_ops)
                continue

            extra_kwargs: dict[str, Any] = {}
            input_names, name_counter = _collect_input_args(
                node, hal_op, ssa_map, name_counter,
                weights, skwargs, sipos, laa, extra_kwargs,
            )
            kwargs = _extract_node_kwargs(node)
            kwargs.update(extra_kwargs)
            _apply_post_kwargs(hal_op, node, input_names, kwargs)

            output_name = f"%{node.name}" if node.name else f"%_out_{name_counter}"
            name_counter += 1
            ssa_map[node.name] = output_name
            kwargs["source_node"] = node.name
            kwargs["dump_layer"] = current_layer

            # Compute output types via shape inference
            input_types, output_types = _resolve_op_types(
                hal_op, input_names, ssa_map, shape_map, weights, kwargs,
            )
            # Record output shape for downstream ops
            if output_types:
                shape_map[node.name] = _parse_mlir_type_to_shape(output_types[0])

            mlir_ops.append(MlirOp(
                name=f"sf.{hal_op}", dialect="sf", op_name=hal_op,
                operands=input_names, results=[output_name],
                attributes=kwargs,
                input_types=input_types, output_types=output_types,
            ))
            continue

        if node.op == "output":
            for arg in node.args[0] if node.args else []:
                if isinstance(arg, torch.fx.Node):
                    out_name = ssa_map.get(arg.name, arg.name)
                    if "val" in arg.meta:
                        tp = _type_from_fake(arg.meta["val"])
                    else:
                        tp = "tensor<f32>"
                    func_outputs.append((out_name, tp))
            continue

    # Backfill dump_layer for handler-generated ops (getitem, split, chunk)
    for op in mlir_ops:
        if "dump_layer" not in op.attributes:
            src = op.attributes.get("source_node", "")
            if src and src in node_layer_map:
                op.attributes["dump_layer"] = node_layer_map[src]

    # ── Phase 4: prepend weight constants ─────────────────
    wops: list[MlirOp] = []
    for wname, tensor in weights.items():
        s = tuple(tensor.shape) if len(tensor.shape) > 0 else (1,)
        elt = _fake_to_shape_tuple(tensor)[1]
        tp_str = _shape_to_mlir_type(s, elt)
        wops.append(MlirOp(
            name="sf.weight", dialect="sf", op_name="weight",
            operands=[wname], results=[f"%{wname}"],
            attributes={"name": wname},
            input_types=[], output_types=[tp_str],
        ))
    mlir_ops = wops + mlir_ops

    # Adjust layer boundaries to account for prepended weight ops
    if layer_boundaries and split_strategy == "layer":
        weight_offset = len(wops)
        adjusted_boundaries = [b + weight_offset for b in layer_boundaries]
    else:
        adjusted_boundaries = []

    # Constants: everything NOT from state_dict (synthesised scalars etc.)
    all_weight_names = set(weights.keys())
    const_names = (const_names or set()) | (all_weight_names - param_names)

    # ── Phase 5: split into per-function chunks (bufferization scaling) ──
    if adjusted_boundaries:
        _log_split_plan(mlir_ops, adjusted_boundaries)
        mlir_ops, func_count = _split_into_functions(
            mlir_ops, func_inputs, weights, boundaries=adjusted_boundaries,
        )
    elif split_strategy == "layer":
        # No layer boundaries detected — skip split (single function)
        func_count = 1
        _log.warning(
            "split_strategy='layer' but no layer boundaries detected; "
            "model may not be a transformer or nn_module_stack is empty"
        )
    else:
        # Fallback to op-count-based splitting
        ops_per_func = 500
        if len(mlir_ops) > ops_per_func:
            mlir_ops, func_count = _split_into_functions(
                mlir_ops, func_inputs, weights, ops_per_func,
            )
        else:
            func_count = 1

    # ── Phase 6: assemble ─────────────────────────────────
    meta: dict[str, Any] = {"source": "torch.export", "artifact_format": "mlir"}
    if hf_key_map:
        meta["hf_key_map"] = hf_key_map

    if func_count == 1:
        return MlirModule(
            functions=[MlirFunction(
                name=function_name, inputs=func_inputs,
                outputs=func_outputs, ops=mlir_ops, weights=weights,
                param_weight_names=param_names, const_weight_names=const_names,
            )],
            metadata=meta,
        )
    else:
        # Multi-function: split by function boundaries computed by _split_into_functions
        functions: list[MlirFunction] = _make_multi_functions(
            mlir_ops, func_inputs, func_outputs, weights,
            param_names, const_names, function_name,
        )
        meta["num_functions"] = func_count
        return MlirModule(
            functions=functions,
            metadata=meta,
        )


# ── handler functions ─────────────────────────────────────


def _do_getitem(
    node: torch.fx.Node,
    ssa_map: dict[str, str],
    weights: dict[str, torch.Tensor],
    tuple_outputs: dict[str, list[str]],
    mlir_ops: list[MlirOp],
) -> None:
    source_node = node.args[0] if node.args else None
    idx = node.args[1] if len(node.args) > 1 else 0
    if isinstance(source_node, torch.fx.Node) and source_node.name in tuple_outputs:
        outputs = tuple_outputs[source_node.name]
        if isinstance(idx, int) and 0 <= idx < len(outputs):
            ssa_map[node.name] = outputs[idx]
            return
    if isinstance(source_node, torch.fx.Node):
        tensor_ssa = ssa_map.get(source_node.name, source_node.name)
        dim_val = _symint_to_int(idx) if isinstance(idx, torch.SymInt) else idx
        output_name = f"%{node.name}" if node.name else f"%_out_{len(mlir_ops)}"
        ssa_map[node.name] = output_name
        mlir_ops.append(MlirOp(
            name="sf.sym_size", dialect="sf", op_name="sym_size",
            operands=[tensor_ssa], results=[output_name],
            attributes={"dim": dim_val, "source_node": node.name},
        ))


def _do_split(
    node: torch.fx.Node,
    ssa_map: dict[str, str],
    tuple_outputs: dict[str, list[str]],
    mlir_ops: list[MlirOp],
) -> None:
    tensor_node = node.args[0] if node.args else None
    split_sizes_raw = node.args[1] if len(node.args) > 1 else []
    dim = node.args[2] if len(node.args) > 2 else 0
    if not isinstance(tensor_node, torch.fx.Node) or not split_sizes_raw:
        return
    tensor_ssa = ssa_map.get(tensor_node.name, tensor_node.name)
    if isinstance(dim, torch.SymInt):
        dim = _symint_to_int(dim) or 0
    sizes: list[int] = []
    for s in split_sizes_raw:  # type: ignore[union-attr]
        concrete = _symint_to_int(s) if isinstance(s, torch.SymInt) else s
        if isinstance(concrete, int) and concrete is not None:
            sizes.append(concrete)
        else:
            return
    if not sizes:
        return
    outputs: list[str] = []
    offset = 0
    for i, size in enumerate(sizes):
        out_name = f"%{node.name}__split_{i}"
        mlir_ops.append(MlirOp(
            name="sf.slice", dialect="sf", op_name="slice",
            operands=[tensor_ssa], results=[out_name],
            attributes={"dim": dim, "start": offset, "end": offset + size, "source_node": node.name},
        ))
        outputs.append(out_name)
        offset += size
    tuple_outputs[node.name] = outputs
    ssa_map[node.name] = outputs[0] if outputs else ssa_map.get(node.name, node.name)


def _do_chunk(
    node: torch.fx.Node,
    ssa_map: dict[str, str],
    tuple_outputs: dict[str, list[str]],
    mlir_ops: list[MlirOp],
) -> None:
    tensor_node = node.args[0] if node.args else None
    chunks = node.args[1] if len(node.args) > 1 else 2
    dim = node.args[2] if len(node.args) > 2 else 0
    if not isinstance(tensor_node, torch.fx.Node) or "val" not in tensor_node.meta:
        return
    tensor_ssa = ssa_map.get(tensor_node.name, tensor_node.name)
    if isinstance(chunks, torch.SymInt):
        chunks = _symint_to_int(chunks) or 2
    if isinstance(dim, torch.SymInt):
        dim = _symint_to_int(dim) or 0
    fake = tensor_node.meta["val"]
    shape = _resolve_shape_tuple(fake.shape)
    total_val = shape[dim] if isinstance(dim, int) and dim < len(shape) else None
    if total_val is None:
        return
    total: int = total_val
    n_chunks = int(chunks)  # type: ignore[arg-type]
    outputs: list[str] = []
    offset = 0
    for i in range(n_chunks):
        size = total // n_chunks + (1 if i < (total % n_chunks) else 0)
        out_name = f"%{node.name}__chunk_{i}"
        mlir_ops.append(MlirOp(
            name="sf.slice", dialect="sf", op_name="slice",
            operands=[tensor_ssa], results=[out_name],
            attributes={"dim": dim, "start": offset, "end": offset + size, "source_node": node.name},
        ))
        outputs.append(out_name)
        offset += size
    tuple_outputs[node.name] = outputs
    ssa_map[node.name] = outputs[0] if outputs else ssa_map.get(node.name, node.name)


def _collect_input_args(
    node: torch.fx.Node,
    hal_op: str,
    ssa_map: dict[str, str],
    name_counter: int,
    weights: dict[str, torch.Tensor],
    scalar_kwargs_map: dict[int, str],
    scalar_int_positions: list[int],
    list_arg_attr: str | None,
    extra_kwargs: dict[str, Any],
) -> tuple[list[str], int]:
    input_names: list[str] = []
    skip_positions: set[int] = set(scalar_int_positions)

    for i, arg in enumerate(node.args):
        if isinstance(arg, torch.fx.Node):
            input_names.append(ssa_map.get(arg.name, arg.name))
        elif isinstance(arg, bool):
            if i in skip_positions:
                kwarg_name = scalar_kwargs_map.get(i)
                if kwarg_name:
                    extra_kwargs.setdefault(kwarg_name, arg)
        elif isinstance(arg, (int, float, torch.SymInt)) and not isinstance(arg, bool):
            if i in skip_positions:
                kwarg_name = scalar_kwargs_map.get(i)
                if kwarg_name:
                    extra_kwargs.setdefault(
                        kwarg_name, _symint_to_int(arg) if isinstance(arg, torch.SymInt) else arg
                    )
                continue
            const_name = f"_const_{name_counter}"
            name_counter += 1
            scalar_val = _symint_to_int(arg) if isinstance(arg, torch.SymInt) else arg
            if scalar_val is None:
                scalar_val = 1
            # Use f32 for float scalars, i64 for int scalars
            if isinstance(scalar_val, float):
                weights[const_name] = torch.tensor(scalar_val, dtype=torch.float32)
            else:
                weights[const_name] = torch.tensor(scalar_val)
            input_names.append(const_name)
        elif isinstance(arg, str):
            kwarg_name = scalar_kwargs_map.get(i)
            if kwarg_name:
                extra_kwargs.setdefault(kwarg_name, arg)
            else:
                const_name = f"_const_{name_counter}"
                name_counter += 1
                weights[const_name] = torch.tensor(0)
                input_names.append(const_name)
        elif isinstance(arg, (torch.dtype, torch.memory_format, torch.layout)):
            kwarg_name = scalar_kwargs_map.get(i)
            if kwarg_name:
                extra_kwargs.setdefault(kwarg_name, str(arg))
        elif isinstance(arg, (list, tuple)):
            if list_arg_attr == "__skip__":
                continue
            if list_arg_attr == "__conv1d__":
                kwarg_name = scalar_kwargs_map.get(i)
                if kwarg_name and kwarg_name not in extra_kwargs:
                    extra_kwargs[kwarg_name] = list(arg)
            elif list_arg_attr is None:
                # This path (expand/arange) uses the list items as both operands
                # and as the shape attribute for C++ lowering.
                # Only SSA refs become operands; static ints stay in shape attr.
                resolved_shape: list[Any] = []
                for item in arg:
                    if isinstance(item, torch.fx.Node):
                        ssa_name = ssa_map.get(item.name, item.name)
                        input_names.append(ssa_name)
                        resolved_shape.append(ssa_name)
                    else:
                        # Static value (int, -1, etc.): store in shape attr only,
                        # NOT as operand — C++ lowering reads from shape attr.
                        resolved_shape.append(item)
                if resolved_shape:
                    extra_kwargs.setdefault("shape", tuple(resolved_shape))
            elif list_arg_attr not in extra_kwargs:
                resolved: list[str | int] = []
                use_view = hal_op == "view"
                for s in arg:
                    if isinstance(s, torch.fx.Node):
                        ssa_name = ssa_map.get(s.name, s.name)
                        resolved.append(ssa_name)
                        input_names.append(ssa_name)
                    elif use_view:
                        resolved.append(_symint_for_view(s))
                    elif isinstance(s, (int, torch.SymInt)) and not isinstance(s, bool):
                        resolved.append(_symint_to_int(s) or 1 if isinstance(s, torch.SymInt) else s)
                    else:
                        resolved.append(s)
                extra_kwargs[list_arg_attr] = tuple(resolved)

    return input_names, name_counter


def _apply_post_kwargs(
    hal_op: str,
    node: torch.fx.Node,
    input_names: list[str],
    kwargs: dict[str, Any],
) -> None:
    if hal_op == "ones_like" and not input_names:
        kwargs.setdefault("shape", (1, 1))
    if hal_op == "full_like" and not input_names:
        kwargs.setdefault("shape", (1,))
        int_args = [a for a in node.args if isinstance(a, int) and not isinstance(a, bool)]
        if len(int_args) >= 2:
            kwargs["dim0"] = int_args[0]
            kwargs["dim1"] = int_args[1]
    if hal_op == "softmax" and "dim" not in kwargs:
        int_args = [a for a in node.args if isinstance(a, (int, torch.SymInt)) and not isinstance(a, bool)]  # type: ignore[misc]
        if int_args:
            kwargs["dim"] = _symint_to_int(int_args[0]) or int_args[0]
    if hal_op == "unsqueeze" and "dim" not in kwargs:
        int_args = [a for a in node.args if isinstance(a, (int, torch.SymInt)) and not isinstance(a, bool)]  # type: ignore[misc]
        if int_args:
            kwargs["dim"] = _symint_to_int(int_args[0]) or int_args[0]
