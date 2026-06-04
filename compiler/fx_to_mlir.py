"""FX Graph → MlirModule conversion.

Replaces the old two-step pipeline (fx_to_ir → IrModule → mlir_emitter → model.mlir)
with a single step that produces an MlirModule directly.  The MlirModule is the
canonical representation consumed by MlirExecutor and serialized to model.mlir.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.fx
from torch.export import ExportedProgram

from compiler.fx_to_mlir_split import (
    _detect_layer,
    _log_split_plan,
    _make_multi_functions,
    _split_into_functions,
)
from compiler.fx_to_mlir_utils import (
    _extract_node_kwargs,
    _fake_to_shape_tuple,
    _map_aten_op,
    _parse_mlir_type_to_shape,
    _resolve_op_types,
    _resolve_shape_tuple,
    _shape_to_mlir_type,
    _symint_for_view,
    _symint_to_int,
    _symint_to_name,
    _type_from_fake,
)
from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp  # type: ignore[attr-defined]
from compiler.mlir_dialect.sf._op_defs import (
    _LIST_ARG_ATTR,
    _SCALAR_INT_POSITIONS,
    _SCALAR_KWARG_NAMES,
)

_log = logging.getLogger(__name__)


def fx_graph_to_mlir(
    program: ExportedProgram,
    function_name: str = "main",
    split_strategy: str = "layer",
    cache_policy: Any = None,
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
    func_outputs: list[tuple[str, str, bool]] = []
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
                # Even unmapped ops have val metadata — capture for type context
                if "val" in node.meta:
                    shape_map[node.name] = _fake_to_shape_tuple(node.meta["val"])
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

            # Generate semantic SSA name based on op type
            output_name = _semantic_ssa_name(hal_op, node, name_counter, kwargs, ssa_map)
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

            # Preserve symbolic dimension names for sf.index ops.
            # These names (e.g. "s0", "s1") flow into sf.dim_names attribute
            # and are used by the C++ lowering as a debug guardrail against
            # sourcing broadcast dims from the data tensor (Bug 2 regression).
            if hal_op == "index" and "val" in node.meta:
                fake_val = node.meta["val"]
                dim_names: list[str] = []
                if hasattr(fake_val, "shape"):
                    for d in fake_val.shape:
                        sym_name = _symint_to_name(d)
                        dim_names.append(sym_name if sym_name else "?")
                if dim_names:
                    kwargs["dim_names"] = dim_names

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
                    func_outputs.append((out_name, tp, False))

    # Post-process: ensure all output names are valid MLIR SSA names.
    # Some FX nodes may be skipped (unmapped ops) so ssa_map fallback
    # returns the FX node name instead of an MLIR SSA name.  Derive
    # correct names from the op results that produce each output.
    if func_outputs:
        _fixup_output_names(func_outputs, mlir_ops)

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

    # ── SD-PA boundary detection (KV cache split) ────────────
    # When CachePolicy has scaled_dot_product_attention intercepts, add
    # SD-PA op indices to the boundary set so each transformer layer is
    # split into two functions at the SD-PA boundary:
    #   main_Xa (QKV proj) → [K/V consumed_internally] → main_Xb (Attn+FFN)
    # Works with or without layer boundaries (SDPA ops may exist even in
    # models without detectable layer structure).
    if cache_policy is not None and hasattr(cache_policy, "intercepts"):
        has_sdpa_intercept = any(
            getattr(i, "op_name", None) == "scaled_dot_product_attention"
            for i in cache_policy.intercepts
        )
        if has_sdpa_intercept:
            sdpa_indices = [i for i, op in enumerate(mlir_ops)
                            if op.op_name == "scaled_dot_product_attention"]
            if sdpa_indices:
                adjusted_boundaries = sorted(set(adjusted_boundaries) | set(sdpa_indices))
                _log.info(
                    "[fx_to_mlir] SD-PA split: %d SD-PA ops found, "
                    "total boundaries = %d",
                    len(sdpa_indices), len(adjusted_boundaries),
                )

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


def _semantic_ssa_name(
    hal_op: str,
    node: torch.fx.Node,
    name_counter: int,
    kwargs: dict[str, Any],
    ssa_map: dict[str, str],
) -> str:
    """Generate a descriptive SSA name based on op type and context."""
    if hal_op == "sym_size":
        dim = kwargs.get("dim", "?")
        # Use the source tensor's SSA name (e.g., input_ids_dim_0)
        src_name = "tensor"
        if node.args and isinstance(node.args[0], torch.fx.Node):
            src_name = ssa_map.get(node.args[0].name, node.args[0].name).lstrip("%")
        return f"%{src_name}_dim_{dim}"
    if hal_op == "arange":
        # Use the input's SSA name for context (e.g., input_ids_dim_1 → arange_dim1_0)
        if node.args and isinstance(node.args[0], torch.fx.Node):
            inp_ssa = ssa_map.get(node.args[0].name, node.args[0].name)
            ctx = inp_ssa.lstrip("%").replace("%", "")
        else:
            ctx = str(name_counter)
        return f"%arange_{ctx}_{name_counter}"
    if hal_op == "le":
        return "%causal_mask"
    if hal_op == "expand":
        return "%attn_mask"
    if hal_op == "logical_and":
        return f"%mask_and_{name_counter}"
    if hal_op == "view":
        return f"%reshape_{name_counter}"
    # Default: keep original FX node name
    return f"%{node.name}" if node.name else f"%_out_{name_counter}"


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
                weights[const_name] = torch.tensor(scalar_val, dtype=torch.int64)
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
                        # Static value: convert sentinel (sys.maxsize) to -1
                        # for shape inference in C++ lowering.
                        val = _symint_to_int(item) if isinstance(item, torch.SymInt) else item
                        if isinstance(val, int) and val == 9223372036854775807:
                            val = -1
                        resolved_shape.append(val)
                if resolved_shape:
                    extra_kwargs.setdefault("shape", tuple(resolved_shape))
            elif list_arg_attr not in extra_kwargs:
                resolved: list[str | int] = []
                use_view = hal_op in ("view", "ones_like", "full_like", "new_zeros")
                is_view = hal_op == "view"
                dyn_pos = 0  # sentinel counter for dynamic dim positions
                for s in arg:
                    if isinstance(s, torch.fx.Node):
                        ssa_name = ssa_map.get(s.name, s.name)
                        # Dynamic dim: add as operand and encode position
                        # via sentinel -(dyn_pos+2) so the view handler knows
                        # which -1 was originally a dynamic vs static dim.
                        if use_view and is_view:
                            # sentinel: -2 for 1st dynamic, -3 for 2nd, etc.
                            resolved.append(-(dyn_pos + 2))
                            dyn_pos += 1
                        elif use_view:
                            resolved.append(-1)
                        else:
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


def _fixup_output_names(
    func_outputs: list[tuple[str, str, bool]],
    mlir_ops: list[MlirOp],
) -> None:
    """Ensure every output name is a valid MLIR SSA name.

    When an output references an FX node that was skipped during conversion
    (unmapped ops), ``ssa_map`` fallback returns the FX node name — not a
    valid MLIR SSA name (e.g. ``%42``).  This function derives correct names
    from the op results that produce each output, matching by type in
    declaration order.

    In-place: modifies ``func_outputs`` entries that lack valid SSA names.
    """
    type_to_results: dict[str, list[str]] = {}
    for op in mlir_ops:
        for j, r in enumerate(op.results):
            if j < len(op.output_types):
                type_to_results.setdefault(op.output_types[j], []).append(r)

    type_positions: dict[str, int] = {}
    for i, (name, tp, consumed) in enumerate(func_outputs):
        if name and name.startswith("%"):
            clean = name.lstrip("%")
            if any(r == name or r.lstrip("%") == clean for op in mlir_ops for r in op.results):
                continue
        results = type_to_results.get(tp, [])
        pos = type_positions.get(tp, 0)
        if pos < len(results):
            func_outputs[i] = ("%" + results[pos], tp, consumed)
            type_positions[tp] = pos + 1
