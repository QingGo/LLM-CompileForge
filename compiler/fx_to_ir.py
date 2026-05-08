"""FX Graph → Custom IR conversion.

Converts a torch.export ExportedProgram's FX graph into our custom
IrModule representation (compiler/ir.py). The IR can then be fed to
the pass pipeline and ultimately to the engine executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx
from torch.export import ExportedProgram

from compiler.ir import IrFunction, IrModule, IrOp, IrType

# ── Unified operator definition ─────────────────────────────
# _OpDef is the single source of truth for FX→IR op conversion.
# It replaces the three separate tables (_ATEN_TO_HAL, _LIST_ARG_ATTR,
# _SCALAR_KWARG_NAMES) that previously required coordinated updates.
#
# To add a new op: add one _OpDef entry to _OP_DEFS below.  The four
# lookup tables (_ATEN_TO_HAL, _LIST_ARG_ATTR, _SCALAR_KWARG_NAMES,
# _SCALAR_INT_POSITIONS) are auto-derived at import time.

_SUPPRESS_LIST = "_SKIP_"  # sentinel: no list-arg handling for this op


@dataclass
class _OpDef:
    """Unified definition of a HAL operator for FX graph conversion.

    Attributes:
        hal_name: HAL operator name (e.g. ``"add"``, ``"matmul"``).
        aten_names: All aten op name strings that map to this HAL op.
        list_arg_attr: How to handle a list/tuple positional arg:
            * ``"_SKIP_"`` (default) — no list-arg handling.
            * ``None`` — flatten list elements into individual input refs.
            * ``"__conv1d__"`` — multi-list-arg dispatch via ``scalar_kwargs``.
            * any other str — store list value as this kwarg attribute name.
        scalar_kwargs: Mapping from positional index to kwarg name for
            scalar args that should be promoted to IrOp attributes.
        scalar_skip: Extra positional indices to skip (consumed but not
            stored as kwargs).  e.g. ``embedding`` has ``scalar_skip=(2,)``
            for ``padding_idx``.
    """
    hal_name: str
    aten_names: tuple[str, ...]
    list_arg_attr: str | None = _SUPPRESS_LIST
    scalar_kwargs: dict[int, str] = field(default_factory=dict)
    scalar_skip: tuple[int, ...] = ()


# ── Master operator registry ────────────────────────────────
# Grouped by logical category for readability.  Adding a new op
# requires *only* adding one entry here — all four lookup tables
# are auto-derived by _build_tables() below.

_OP_DEFS: list[_OpDef] = [
    # ── Arithmetic ──
    _OpDef("add", ("aten.add.Tensor", "aten.add.Scalar", "aten.add", "aten.add_.Tensor", "add")),
    _OpDef("mul", ("aten.mul.Tensor", "aten.mul.Scalar", "aten.mul", "aten.mul_.Tensor", "aten.mul_.Scalar")),
    _OpDef("sub", ("aten.sub", "aten.sub.Tensor", "aten.rsub", "aten.rsub.Scalar")),
    _OpDef("neg", ("neg", "aten.neg.default", "aten.neg")),
    _OpDef("pow", ("pow", "aten.pow.Tensor_Scalar", "aten.pow")),
    _OpDef("max", ("aten.max", "aten.max.other")),
    # ── Activation ──
    _OpDef("relu", ("aten.relu", "aten.relu.default")),
    _OpDef("gelu", ("aten.gelu", "aten.gelu.default")),
    _OpDef("silu", ("aten.silu", "aten.silu.default")),
    _OpDef("sigmoid", ("aten.sigmoid.default", "aten.sigmoid")),
    _OpDef("softplus", ("aten.softplus.default",)),
    _OpDef("exp", ("aten.exp.default",)),
    # ── Normalization ──
    _OpDef("layer_norm", ("aten.layer_norm", "aten.layer_norm.default",
                           "aten.native_layer_norm", "aten.native_layer_norm.default"),
            list_arg_attr="normalized_shape"),
    _OpDef("rms_norm", ("aten.rms_norm", "aten.rms_norm.default"),
            list_arg_attr="normalized_shape"),
    _OpDef("softmax", ("aten._softmax", "aten._softmax.default", "aten.softmax.int"),
            scalar_kwargs={1: "dim"}),
    # ── Linear algebra ──
    _OpDef("matmul", ("aten.matmul", "aten.matmul.default", "aten.mm", "aten.mm.default", "aten.bmm")),
    _OpDef("linear", ("aten.linear", "aten.linear.default")),
    # ── Shape & indexing ──
    _OpDef("view", ("aten.view", "aten.view.default", "aten.reshape", "aten.reshape.default"),
            list_arg_attr="shape"),
    _OpDef("unsqueeze", ("aten.unsqueeze", "aten.unsqueeze.default"),
            scalar_kwargs={1: "dim"}),
    _OpDef("expand", ("aten.expand", "aten.expand.default"),
            list_arg_attr=None),
    _OpDef("permute", ("aten.permute", "aten.permute.default")),
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
    # ── Attention ──
    _OpDef("scaled_dot_product_attention",
            ("aten.scaled_dot_product_attention", "aten.scaled_dot_product_attention.default")),
    # ── Comparison ──
    _OpDef("gt", ("gt", "aten.gt.Tensor", "aten.gt")),
    _OpDef("lt", ("aten.lt", "aten.lt.Tensor")),
    _OpDef("eq", ("aten.eq.Tensor",)),
    _OpDef("ne", ("aten.ne.Scalar", "aten.ne.Tensor")),
    _OpDef("le", ("aten.le.Tensor",)),
    _OpDef("logical_and", ("aten.__and__.Tensor",)),
    # ── Math ──
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
    # ── Constant creation ──
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
    # ── Embedding ──
    _OpDef("embedding", ("aten.embedding", "aten.embedding.default"),
            scalar_skip=(2,)),
    # ── Misc ──
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
    # ── Identity / passthrough ──
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
    # ── Special (compile-time resolved) ──
    _OpDef("getitem", ("getitem",)),
    _OpDef("_skip_wrap", ("wrap_with_set_grad_enabled",)),
]

# ── Auto-derived lookup tables ──────────────────────────────

_ATEN_TO_HAL: dict[str, str] = {}
_LIST_ARG_ATTR: dict[str, str | None] = {}
_SCALAR_KWARG_NAMES: dict[str, dict[int, str]] = {}
_SCALAR_INT_POSITIONS: dict[str, list[int]] = {}


def _build_tables() -> None:
    """Populate all four lookup tables from _OP_DEFS.

    Called at module import time.  After this, the four module-level
    dicts behave identically to the hand-maintained versions they replace.
    """
    for od in _OP_DEFS:
        hal = od.hal_name
        # _ATEN_TO_HAL: reverse map from aten name → HAL name
        for aten_name in od.aten_names:
            if aten_name in _ATEN_TO_HAL and _ATEN_TO_HAL[aten_name] != hal:
                raise AssertionError(
                    f"aten '{aten_name}' maps to both "
                    f"'{_ATEN_TO_HAL[aten_name]}' and '{hal}'"
                )
            _ATEN_TO_HAL[aten_name] = hal
        # _LIST_ARG_ATTR
        if od.list_arg_attr != _SUPPRESS_LIST:
            _LIST_ARG_ATTR.setdefault(hal, od.list_arg_attr)
        # _SCALAR_KWARG_NAMES
        if od.scalar_kwargs:
            _SCALAR_KWARG_NAMES.setdefault(hal, od.scalar_kwargs)
        # _SCALAR_INT_POSITIONS: kwarg positions + extra skip positions
        positions = set(od.scalar_kwargs.keys()) | set(od.scalar_skip)
        if positions:
            existing = set(_SCALAR_INT_POSITIONS.get(hal, []))
            _SCALAR_INT_POSITIONS[hal] = sorted(existing | positions)


_build_tables()


def _symint_to_int(val: Any) -> int | None:
    """Convert a SymInt to concrete int if possible, else return None."""
    if isinstance(val, torch.SymInt):
        if hasattr(val, "node") and val.node is not None:
            hint = getattr(val.node, "hint", None)
            if hint is not None:
                return int(hint)
        return None
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _symint_for_view(val: Any) -> int:
    """Convert view shape element: concrete int or -1 for dynamic dim."""
    concrete = _symint_to_int(val)
    if concrete is not None:
        return concrete
    return -1


def _resolve_shape_tuple(raw_shape: Any) -> tuple[int | None, ...]:
    """Convert raw shape (possibly with SymInt) to tuple of int or None."""
    result: list[int | None] = []
    for d in raw_shape:
        result.append(_symint_to_int(d))
    return tuple(result)


def _map_aten_op(target: Any) -> str | None:
    """Map an aten operator to its HAL op name."""
    if isinstance(target, str):
        target_str = target
    elif hasattr(target, "name"):
        # OpOverload.name() returns 'aten::diff' (short), str() returns 'aten.diff.default' (full).
        # Prefer str() for overload resolution, fall back to name() for compatibility.
        target_str = str(target)
    elif hasattr(target, "__name__"):
        target_str = target.__name__  # built-in functions (operator.add, etc.)
    else:
        target_str = str(target)
    # Normalize: OpOverload.name() returns 'aten::view' — convert '::' → '.'
    target_str = target_str.replace("::", ".")
    # Try exact match first
    if target_str in _ATEN_TO_HAL:
        return _ATEN_TO_HAL[target_str]
    # Try matching by stripping overload suffix: 'aten.softmax.int' → 'aten.softmax'
    if "." in target_str:
        parts = target_str.rsplit(".", 1)
        # Only strip if the last part looks like an overload (e.g. 'int', 'default', 'Tensor')
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
    """Extract keyword arguments from an FX node.

    For call_function nodes, args follow the function signature;
    kwargs are stored in node.kwargs when using named parameters.
    """
    kwargs: dict[str, Any] = dict(node.kwargs)
    # Merge any extra attributes that look like op parameters
    for attr in ("dim", "eps", "is_causal", "dropout_p", "normalized_shape"):
        if attr in kwargs:
            continue
    return kwargs


def fx_graph_to_ir(
    program: ExportedProgram,
    function_name: str = "main",
) -> IrModule:
    """Convert an ExportedProgram's FX graph to an IrModule.

    The conversion process:
      1. Walk FX graph nodes in topological order.
      2. Placeholder nodes → function inputs.
      3. get_attr nodes → weight references.
      4. call_function nodes → IrOp entries mapped via _ATEN_TO_HAL.
      5. output nodes → function outputs.

    Args:
        program: The ExportedProgram from torch.export.
        function_name: Name for the resulting IrFunction.

    Returns:
        IrModule containing the converted graph and weights.
    """
    gm = program.graph_module
    graph = gm.graph
    state_dict = program.state_dict

    # ── Phase 1: collect placeholder → function inputs ──────
    sig = program.graph_signature
    func_inputs: list[tuple[str, IrType]] = []
    placeholder_to_name: dict[str, str] = {}

    # Map from user input names to node types
    for inp_name in sig.user_inputs:
        node = None
        for n in graph.nodes:
            if n.name == inp_name:
                node = n
                break
        if node is not None and "val" in node.meta:
            fake = node.meta["val"]
            shape = _resolve_shape_tuple(fake.shape)
            dtype = str(fake.dtype).replace("torch.", "")
            func_inputs.append((inp_name, IrType(dtype=dtype, shape=shape)))
        else:
            func_inputs.append((inp_name, IrType(dtype="float32")))

    # Map parameter/buffer names — build weight_name_map from input_specs
    weight_name_map: dict[str, str] = {}
    if hasattr(sig, "input_specs"):
        for spec in sig.input_specs:
            # Include PARAMETER (value 2), BUFFER, and CONSTANT_TENSOR spec kinds
            if spec.kind.value in (2, 3, 4):
                placeholder_name = spec.arg.name
                target_path = spec.target
                if target_path:
                    clean_name = target_path.replace(".", "_")
                    weight_name_map[placeholder_name] = clean_name
    else:
        # Fallback: old API with inputs_to_parameters
        for param_name in getattr(sig, "inputs_to_parameters", {}):
            placeholder_to_name[param_name] = param_name

    # ── Phase 2: collect weights ────────────────────────────
    weights: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        clean_name = name.replace(".", "_")
        weights[clean_name] = tensor
    # Also include exported program constants (e.g., lifted tensors)
    if hasattr(program, "constants"):
        for name, tensor in program.constants.items():
            clean_name = name.replace(".", "_")
            if clean_name not in weights:
                weights[clean_name] = tensor

    # ── Phase 3: walk operations ────────────────────────────
    ir_ops: list[IrOp] = []
    func_outputs: list[tuple[str, IrType]] = []
    name_counter = 0

    # Track SSA value → producer node for dataflow edges
    ssa_map: dict[str, str] = {}  # SSA name → producing node name

    # Track tuple-producing nodes — maps node name → [ssa_output_0, ssa_output_1, ...]
    _tuple_outputs: dict[str, list[str]] = {}

    for node in graph.nodes:
        if node.op == "placeholder":
            if node.name in weight_name_map:
                # Weight placeholder — map to weight reference
                ssa_map[node.name] = weight_name_map[node.name]
            else:
                ssa_map[node.name] = node.name
            continue

        if node.op == "get_attr":
            # Weight/constant access — reference by clean name
            attr_name = str(node.target).replace(".", "_")
            ssa_map[node.name] = attr_name
            continue

        if node.op == "call_function":
            hal_op = _map_aten_op(node.target)
            if hal_op is None:
                continue
            if hal_op == "_skip_wrap":
                # wrap_with_set_grad_enabled: skip, redirect output to input_ids
                ssa_map[node.name] = func_inputs[0][0] if func_inputs else node.name
                continue

            # ── getitem: resolve tuple indexing at compile time ──
            if hal_op == "getitem":
                source_node = node.args[0] if node.args else None
                idx = node.args[1] if len(node.args) > 1 else 0
                if isinstance(source_node, torch.fx.Node) and source_node.name in _tuple_outputs:
                    outputs = _tuple_outputs[source_node.name]
                    if isinstance(idx, int) and 0 <= idx < len(outputs):
                        ssa_map[node.name] = outputs[idx]
                        continue
                # Fallback: getitem on a non-tuple source — treat as sym_size (shape dim extraction)
                if isinstance(source_node, torch.fx.Node):
                    tensor_ssa = ssa_map.get(source_node.name, source_node.name)
                    dim_val = _symint_to_int(idx) if isinstance(idx, torch.SymInt) else idx
                    output_name = node.name or f"_out_{name_counter}"
                    name_counter += 1
                    ssa_map[node.name] = output_name
                    ir_ops.append(IrOp(name="sym_size", inputs=[tensor_ssa],
                                       outputs=[output_name],
                                       attributes={"dim": dim_val, "source_node": node.name}))
                continue

            # ── split_with_sizes: expand to N slice ops ──
            if hal_op == "split":
                tensor_node = node.args[0] if node.args else None
                split_sizes_raw = node.args[1] if len(node.args) > 1 else []
                dim = node.args[2] if len(node.args) > 2 else 0
                if isinstance(tensor_node, torch.fx.Node) and split_sizes_raw:
                    tensor_ssa = ssa_map.get(tensor_node.name, tensor_node.name)
                    if isinstance(dim, torch.SymInt):
                        dim = _symint_to_int(dim) or 0
                    # Resolve split sizes to concrete ints
                    sizes: list[int] = []
                    for s in split_sizes_raw:
                        concrete = _symint_to_int(s) if isinstance(s, torch.SymInt) else s
                        if isinstance(concrete, int) and concrete is not None:
                            sizes.append(concrete)
                        else:
                            sizes = []
                            break
                    if sizes:
                        outputs = []
                        offset = 0
                        for i, size in enumerate(sizes):
                            out_name = f"{node.name}__split_{i}"
                            slice_op = IrOp(
                                name="slice", inputs=[tensor_ssa],
                                outputs=[out_name],
                                attributes={
                                    "dim": dim, "start": offset,
                                    "end": offset + size, "source_node": node.name,
                                },
                            )
                            ir_ops.append(slice_op)
                            outputs.append(out_name)
                            offset += size
                        _tuple_outputs[node.name] = outputs
                        ssa_map[node.name] = outputs[0] if outputs else node.name
                        continue

            # ── chunk: expand to N slice ops ──
            if hal_op == "chunk":
                tensor_node = node.args[0] if node.args else None
                chunks = node.args[1] if len(node.args) > 1 else 2
                dim = node.args[2] if len(node.args) > 2 else 0
                if isinstance(tensor_node, torch.fx.Node) and "val" in tensor_node.meta:
                    tensor_ssa = ssa_map.get(tensor_node.name, tensor_node.name)
                    if isinstance(chunks, torch.SymInt):
                        chunks = _symint_to_int(chunks) or 2
                    if isinstance(dim, torch.SymInt):
                        dim = _symint_to_int(dim) or 0
                    fake = tensor_node.meta["val"]
                    shape = _resolve_shape_tuple(fake.shape)
                    total_val = shape[dim]
                    if isinstance(dim, int) and dim < len(shape) and total_val is not None:
                        total: int = total_val
                        outputs = []
                        offset = 0
                        for i in range(int(chunks)):
                            size = total // int(chunks) + (1 if i < (total % int(chunks)) else 0)
                            out_name = f"{node.name}__chunk_{i}"
                            slice_op = IrOp(
                                name="slice", inputs=[tensor_ssa],
                                outputs=[out_name],
                                attributes={
                                    "dim": dim, "start": offset,
                                    "end": offset + size, "source_node": node.name,
                                },
                            )
                            ir_ops.append(slice_op)
                            outputs.append(out_name)
                            offset += size
                        _tuple_outputs[node.name] = outputs
                        ssa_map[node.name] = outputs[0] if outputs else node.name
                        continue

            # Collect input SSA names and extract non-tensor kwargs
            input_names: list[str] = []
            extra_kwargs: dict[str, Any] = {}
            skip_positions: list[int] = _SCALAR_INT_POSITIONS.get(hal_op, [])
            scalar_kwargs: dict[int, str] = _SCALAR_KWARG_NAMES.get(hal_op, {})

            for i, arg in enumerate(node.args):
                if isinstance(arg, torch.fx.Node):
                    input_names.append(ssa_map.get(arg.name, arg.name))
                elif isinstance(arg, bool):
                    # Boolean positional args: treat as kwarg if in skip_positions
                    if i in skip_positions:
                        kwarg_name = scalar_kwargs.get(i)
                        if kwarg_name:
                            extra_kwargs.setdefault(kwarg_name, arg)
                        continue
                elif isinstance(arg, (int, float, torch.SymInt)) and not isinstance(arg, bool):
                    if i in skip_positions:
                        kwarg_name = scalar_kwargs.get(i)
                        if kwarg_name:
                            if isinstance(arg, torch.SymInt):
                                extra_kwargs.setdefault(kwarg_name, _symint_to_int(arg))
                            else:
                                extra_kwargs.setdefault(kwarg_name, arg)
                        continue
                    const_name = f"_const_{name_counter}"
                    name_counter += 1
                    if isinstance(arg, torch.SymInt):
                        scalar_val: Any = _symint_to_int(arg)
                        if scalar_val is None:
                            scalar_val = 1
                    else:
                        scalar_val = arg
                    weights[const_name] = torch.tensor(scalar_val)
                    input_names.append(const_name)
                elif isinstance(arg, (torch.dtype, torch.memory_format, torch.layout)):
                    kwarg_name = scalar_kwargs.get(i)
                    if kwarg_name:
                        extra_kwargs.setdefault(kwarg_name, str(arg))
                    continue
                elif isinstance(arg, (list, tuple)):
                    list_attr = _LIST_ARG_ATTR.get(hal_op, "__skip__")
                    if list_attr == "__skip__":
                        continue

                    if list_attr == "__conv1d__":
                        # conv1d: multiple list args dispatched via scalar_kwargs
                        kwarg_name = scalar_kwargs.get(i)
                        if kwarg_name and kwarg_name not in extra_kwargs:
                            extra_kwargs[kwarg_name] = list(arg)

                    elif list_attr is None:
                        # Flatten into individual inputs (cat, expand, index)
                        for item in arg:
                            if isinstance(item, torch.fx.Node):
                                input_names.append(ssa_map.get(item.name, item.name))
                            else:
                                const_name = f"_const_{name_counter}"
                                name_counter += 1
                                weights[const_name] = torch.tensor(item)
                                input_names.append(const_name)

                    elif list_attr not in extra_kwargs:
                        # SSA-aware shape/normalized_shape attribute extraction
                        resolved: list[str | int] = []
                        _use_view = hal_op == "view"
                        for s in arg:
                            if isinstance(s, torch.fx.Node):
                                ssa_name = ssa_map.get(s.name, s.name)
                                resolved.append(ssa_name)
                                input_names.append(ssa_name)
                            elif _use_view:
                                resolved.append(_symint_for_view(s))
                            elif isinstance(s, (int, torch.SymInt)) and not isinstance(s, bool):
                                resolved.append(_symint_to_int(s) or 1 if isinstance(s, torch.SymInt) else s)
                            else:
                                resolved.append(s)
                        extra_kwargs[list_attr] = tuple(resolved)

            kwargs = _extract_node_kwargs(node)
            kwargs.update(extra_kwargs)

            # Extract positional int args for ops that need them
            if hal_op == "ones_like" and not input_names:
                # Create ones from shape attribute
                extra_kwargs.setdefault("shape", (1, 1))
            if hal_op == "full_like" and not input_names:
                extra_kwargs.setdefault("shape", (1,))
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

            output_name = node.name or f"_out_{name_counter}"
            name_counter += 1
            ssa_map[node.name] = output_name

            kwargs["source_node"] = node.name
            ir_ops.append(IrOp(
                name=hal_op, inputs=input_names, outputs=[output_name],
                attributes=kwargs, in_place=(hal_op == "copy_"),
            ))
            continue

        if node.op == "output":
            for arg in node.args[0] if node.args else []:
                if isinstance(arg, torch.fx.Node):
                    out_name = ssa_map.get(arg.name, arg.name)
                    out_dtype = "float32"
                    out_shape: tuple[int | None, ...] = ()
                    if "val" in arg.meta:
                        fake = arg.meta["val"]
                        out_shape = _resolve_shape_tuple(fake.shape)
                        out_dtype = str(fake.dtype).replace("torch.", "")
                    func_outputs.append((out_name, IrType(dtype=out_dtype, shape=out_shape)))
            continue

    # ── Phase 4: assemble ───────────────────────────────────
    function = IrFunction(
        name=function_name,
        inputs=func_inputs,
        outputs=func_outputs,
        ops=ir_ops,
        weights=weights,
    )
    return IrModule(functions=[function], metadata={"source": "torch.export"})
