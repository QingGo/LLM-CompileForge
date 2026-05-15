"""FX Graph → MlirModule conversion.

Replaces the old two-step pipeline (fx_to_ir → IrModule → mlir_emitter → model.mlir)
with a single step that produces an MlirModule directly.  The MlirModule is the
canonical representation consumed by MlirExecutor and serialized to model.mlir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx
from torch.export import ExportedProgram

from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp
from compiler.mlir_dialect.shape_inference import infer_output_shape

# ── Unified operator definition ─────────────────────────────
# _OpDef is the single source of truth for FX→MLIR op conversion.
# Adding a new op requires *only* adding one entry here — all four
# lookup tables are auto-derived by _build_tables() below.

_SUPPRESS_LIST = "_SKIP_"


@dataclass
class _OpDef:
    hal_name: str
    aten_names: tuple[str, ...]
    list_arg_attr: str | None = _SUPPRESS_LIST
    scalar_kwargs: dict[int, str] = field(default_factory=dict)
    scalar_skip: tuple[int, ...] = ()


_OP_DEFS: list[_OpDef] = [
    _OpDef("add", ("aten.add.Tensor", "aten.add.Scalar", "aten.add", "aten.add_.Tensor", "add")),
    _OpDef("mul", ("aten.mul.Tensor", "aten.mul.Scalar", "aten.mul", "aten.mul_.Tensor", "aten.mul_.Scalar")),
    _OpDef("sub", ("aten.sub", "aten.sub.Tensor", "aten.rsub", "aten.rsub.Scalar")),
    _OpDef("neg", ("neg", "aten.neg.default", "aten.neg")),
    _OpDef("pow", ("pow", "aten.pow.Tensor_Scalar", "aten.pow")),
    _OpDef("max", ("aten.max", "aten.max.other")),
    _OpDef("relu", ("aten.relu", "aten.relu.default")),
    _OpDef("gelu", ("aten.gelu", "aten.gelu.default")),
    _OpDef("silu", ("aten.silu", "aten.silu.default")),
    _OpDef("sigmoid", ("aten.sigmoid.default", "aten.sigmoid")),
    _OpDef("softplus", ("aten.softplus.default",)),
    _OpDef("exp", ("aten.exp.default",)),
    _OpDef("layer_norm", ("aten.layer_norm", "aten.layer_norm.default",
                           "aten.native_layer_norm", "aten.native_layer_norm.default"),
            list_arg_attr="normalized_shape"),
    _OpDef("rms_norm", ("aten.rms_norm", "aten.rms_norm.default"),
            list_arg_attr="normalized_shape"),
    _OpDef("softmax", ("aten._softmax", "aten._softmax.default", "aten.softmax.int"),
            scalar_kwargs={1: "dim"}),
    _OpDef("matmul", ("aten.matmul", "aten.matmul.default", "aten.mm", "aten.mm.default", "aten.bmm")),
    _OpDef("linear", ("aten.linear", "aten.linear.default")),
    _OpDef("view", ("aten.view", "aten.view.default", "aten.reshape", "aten.reshape.default"),
            list_arg_attr="shape"),
    _OpDef("unsqueeze", ("aten.unsqueeze", "aten.unsqueeze.default"),
            scalar_kwargs={1: "dim"}),
    _OpDef("expand", ("aten.expand", "aten.expand.default"),
            list_arg_attr=None),
    _OpDef("permute", ("aten.permute", "aten.permute.default"),
            list_arg_attr="dims"),
    _OpDef("transpose", ("aten.transpose", "aten.transpose.int"),
            scalar_kwargs={1: "dim0", 2: "dim1"}),
    _OpDef("slice", ("aten.slice.Tensor", "aten.slice_copy.Tensor"),
            scalar_kwargs={1: "dim", 2: "start", 3: "end", 4: "step"}),
    _OpDef("select", ("aten.select.int", "aten.select"),
            scalar_kwargs={1: "dim", 2: "index"}),
    _OpDef("cat", ("aten.cat", "aten.cat.default"),
            list_arg_attr=None, scalar_kwargs={1: "dim"}),
    _OpDef("split", ("aten.split_with_sizes.default",),
            list_arg_attr="split_sizes", scalar_kwargs={2: "dim"}),
    _OpDef("chunk", ("aten.chunk.default",),
            scalar_kwargs={1: "chunks", 2: "dim"}),
    _OpDef("scaled_dot_product_attention",
            ("aten.scaled_dot_product_attention", "aten.scaled_dot_product_attention.default")),
    _OpDef("gt", ("gt", "aten.gt.Tensor", "aten.gt")),
    _OpDef("lt", ("aten.lt", "aten.lt.Tensor")),
    _OpDef("eq", ("aten.eq.Tensor",)),
    _OpDef("ne", ("aten.ne.Scalar", "aten.ne.Tensor")),
    _OpDef("le", ("aten.le.Tensor",)),
    _OpDef("logical_and", ("aten.__and__.Tensor",)),
    _OpDef("cos", ("aten.cos.default", "aten.cos")),
    _OpDef("sin", ("aten.sin.default", "aten.sin")),
    _OpDef("rsqrt", ("rsqrt", "aten.rsqrt.default", "aten.rsqrt")),
    _OpDef("mean", ("mean", "aten.mean.dim", "aten.mean")),
    _OpDef("triu", ("triu", "aten.triu.default", "aten.triu"),
            scalar_kwargs={1: "diagonal"}),
    _OpDef("tril", ("aten.tril.default", "aten.tril"),
            scalar_kwargs={1: "diagonal"}),
    _OpDef("cumsum", ("aten.cumsum", "aten.cumsum.default"),
            scalar_kwargs={1: "dim"}),
    _OpDef("sum", ("aten.sum.dim_IntList",),
            list_arg_attr="dim", scalar_kwargs={1: "dim", 2: "keepdim"}),
    _OpDef("diff", ("aten.diff.default",),
            scalar_kwargs={1: "n", 2: "dim"}),
    _OpDef("arange", ("aten.arange.start", "aten.arange", "aten.arange.default")),
    _OpDef("ones_like", ("aten.ones", "aten.ones.default"),
            list_arg_attr="shape"),
    _OpDef("full_like", ("aten.full", "aten.full.default"),
            list_arg_attr="shape", scalar_kwargs={1: "fill_value"}),
    _OpDef("zeros", ("aten.zeros.default",),
            list_arg_attr="shape"),
    _OpDef("zeros_like", ("aten.zeros_like.default",)),
    _OpDef("new_ones", ("aten.new_ones.default",)),
    _OpDef("eye", ("aten.eye.default",),
            scalar_kwargs={0: "n", 1: "m"}),
    _OpDef("embedding", ("aten.embedding", "aten.embedding.default"),
            scalar_skip=(2,)),
    _OpDef("masked_fill", ("aten.masked_fill", "aten.masked_fill.Scalar",
                            "aten.masked_fill_", "aten.masked_fill_.Scalar")),
    _OpDef("conv1d", ("aten.conv1d.default",),
            list_arg_attr="__conv1d__",
            scalar_kwargs={2: "bias", 3: "stride", 4: "padding", 5: "dilation", 6: "groups"}),
    _OpDef("pad", ("aten.pad.default",),
            list_arg_attr="pad", scalar_skip=(1,)),
    _OpDef("index", ("aten.index.Tensor",),
            list_arg_attr=None),
    _OpDef("sym_size", ("sym_size", "aten.sym_size.int", "aten.sym_size"),
            scalar_kwargs={1: "dim"}),
    _OpDef("copy_", ("aten.copy_.default",)),
    _OpDef("type_as", ("aten.type_as", "aten.type_as.default")),
    _OpDef("identity", (
        "_assert_tensor_metadata",
        "aten._assert_tensor_metadata.default", "aten._assert_tensor_metadata",
        "aten.to", "aten.to.dtype", "aten.to.dtype_layout",
        "aten.contiguous", "aten.contiguous.default",
        "aten.clone", "aten.clone.default",
        "aten.dropout", "aten.dropout.default",
        "aten.detach", "aten.detach.default", "aten.detach_", "aten.detach_.default",
        "aten.alias", "aten.alias.default",
        "aten.lift_fresh_copy", "aten.lift_fresh_copy.default",
    ), scalar_kwargs={1: "dtype"}),
    _OpDef("getitem", ("getitem",)),
    _OpDef("_skip_wrap", ("wrap_with_set_grad_enabled",)),
    # ── RWKV ops ─────────────────────────────────
    _OpDef("div", ("aten.div.Tensor", "aten.div.default")),
    _OpDef("tanh", ("aten.tanh.default",)),
    _OpDef("sqrt", ("aten.sqrt.default",)),
    _OpDef("clamp_min", ("aten.clamp_min.default",)),
    _OpDef("einsum", ("aten.einsum.default",),
            scalar_kwargs={0: "equation"}, list_arg_attr=None),
    _OpDef("stack", ("aten.stack.default",)),
    _OpDef("linalg_norm", ("aten.linalg_vector_norm.default",)),
    _OpDef("var", ("aten.var.dim",)),
    _OpDef("view_as", ("aten.view_as.default",)),
    _OpDef("expand_as", ("aten.expand_as.default",)),
]

_ATEN_TO_HAL: dict[str, str] = {}
_LIST_ARG_ATTR: dict[str, str | None] = {}
_SCALAR_KWARG_NAMES: dict[str, dict[int, str]] = {}
_SCALAR_INT_POSITIONS: dict[str, list[int]] = {}


def _build_tables() -> None:
    for od in _OP_DEFS:
        hal = od.hal_name
        for aten_name in od.aten_names:
            if aten_name in _ATEN_TO_HAL and _ATEN_TO_HAL[aten_name] != hal:
                raise AssertionError(
                    f"aten '{aten_name}' maps to both "
                    f"'{_ATEN_TO_HAL[aten_name]}' and '{hal}'"
                )
            _ATEN_TO_HAL[aten_name] = hal
        if od.list_arg_attr != _SUPPRESS_LIST:
            _LIST_ARG_ATTR.setdefault(hal, od.list_arg_attr)
        if od.scalar_kwargs:
            _SCALAR_KWARG_NAMES.setdefault(hal, od.scalar_kwargs)
        positions = set(od.scalar_kwargs.keys()) | set(od.scalar_skip)
        if positions:
            existing = set(_SCALAR_INT_POSITIONS.get(hal, []))
            _SCALAR_INT_POSITIONS[hal] = sorted(existing | positions)


_build_tables()


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
    except Exception:
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
    ops_per_func: int,
) -> tuple[list[MlirOp], int]:
    """Insert sentinel ops marking function boundaries every ops_per_func ops.

    Returns (augmented_ops, num_functions).
    """
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
            for val in sorted(needed, key=lambda v: (producer.get(v, 0), v)):
                tp = type_map.get(val, "tensor<f32>")
                f_inputs.append((f"%{val}", tp))

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

    # Constants: everything NOT from state_dict (synthesised scalars etc.)
    all_weight_names = set(weights.keys())
    const_names = (const_names or set()) | (all_weight_names - param_names)

    # ── Phase 5: split into per-function chunks (bufferization scaling) ──
    ops_per_func = 500
    if len(mlir_ops) > ops_per_func:
        mlir_ops, func_count = _split_into_functions(mlir_ops, func_inputs, weights, ops_per_func)
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
