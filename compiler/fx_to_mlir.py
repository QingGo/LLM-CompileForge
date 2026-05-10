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
                return int(hint)
        return None
    if isinstance(val, int):
        return val
    try:
        return int(val)
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
    for name, tensor in state_dict.items():
        weights[name.replace(".", "_")] = tensor
    if hasattr(program, "constants"):
        for name, tensor in program.constants.items():
            clean = name.replace(".", "_")
            weights.setdefault(clean, tensor)

    # ── Phase 3: walk operations ──────────────────────────
    mlir_ops: list[MlirOp] = []
    func_outputs: list[tuple[str, str]] = []
    name_counter = 0
    ssa_map: dict[str, str] = {}
    tuple_outputs: dict[str, list[str]] = {}

    for node in graph.nodes:
        if node.op == "placeholder":
            ssa_map[node.name] = f"%{weight_name_map.get(node.name, node.name)}"
            continue

        if node.op == "get_attr":
            attr_name = str(node.target).replace(".", "_")
            ssa_map[node.name] = f"%{attr_name}"  # SSA reference to weight constant result
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

            mlir_ops.append(MlirOp(
                name=f"sf.{hal_op}", dialect="sf", op_name=hal_op,
                operands=input_names, results=[output_name],
                attributes=kwargs,
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
    for wname in weights:
        wops.append(MlirOp(
            name="sf.weight", dialect="sf", op_name="weight",
            operands=[wname], results=[f"%{wname}"],
            attributes={"name": wname},
        ))
    mlir_ops = wops + mlir_ops

    # ── Phase 5: assemble ─────────────────────────────────
    return MlirModule(
        functions=[MlirFunction(
            name=function_name, inputs=func_inputs,
            outputs=func_outputs, ops=mlir_ops, weights=weights,
        )],
        metadata={"source": "torch.export", "artifact_format": "mlir"},
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
            weights[const_name] = torch.tensor(scalar_val)
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
                for item in arg:
                    if isinstance(item, torch.fx.Node):
                        input_names.append(ssa_map.get(item.name, item.name))
                    else:
                        const_name = f"_const_{name_counter}"
                        name_counter += 1
                        weights[const_name] = torch.tensor(item)
                        input_names.append(const_name)
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
