"""FX Graph → MlirModule conversion.

Replaces the old two-step pipeline (fx_to_ir → IrModule → mlir_emitter → model.mlir)
with a single step that produces an MlirModule directly.  The MlirModule is the
canonical representation consumed by MlirExecutor and serialized to model.mlir.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.fx
from torch.export import ExportedProgram

from compiler.fx_to_ir import (
    _LIST_ARG_ATTR,
    _SCALAR_INT_POSITIONS,
    _SCALAR_KWARG_NAMES,
    _extract_node_kwargs,
    _map_aten_op,
    _resolve_shape_tuple,
    _symint_for_view,
    _symint_to_int,
)
from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp


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
            ssa_map[node.name] = weight_name_map.get(node.name, f"%{node.name}")
            continue

        if node.op == "get_attr":
            ssa_map[node.name] = str(node.target).replace(".", "_")
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
