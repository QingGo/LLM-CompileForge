"""FX Graph → Custom IR conversion.

Converts a torch.export ExportedProgram's FX graph into our custom
IrModule representation (compiler/ir.py). The IR can then be fed to
the pass pipeline and ultimately to the engine executor.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.fx
from torch.export import ExportedProgram

from compiler.ir import IrFunction, IrModule, IrOp, IrType

# ── Aten → HAL operator mapping table ───────────────────────

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


_ATEN_TO_HAL: dict[str, str] = {
    "aten.add.Tensor": "add",
    "aten.add.Scalar": "add",
    "aten.add": "add",
    "aten.add_.Tensor": "add",
    "aten.mul.Tensor": "mul",
    "aten.mul.Scalar": "mul",
    "aten.mul": "mul",
    "aten.mul_.Tensor": "mul",
    "aten.mul_.Scalar": "mul",
    # Built-in Python ops that appear in FX graphs
    "add": "add",
    "getitem": "sym_size",
    "neg": "neg",
    "rsqrt": "rsqrt",
    "mean": "mean",
    "pow": "pow",
    "gt": "identity",
    "triu": "triu",
    "sym_size": "sym_size",
    "_assert_tensor_metadata": "identity",
    # Additional aten overloads
    "aten.neg.default": "neg",
    "aten.neg": "neg",
    "aten.rsqrt.default": "rsqrt",
    "aten.rsqrt": "rsqrt",
    "aten.pow.Tensor_Scalar": "pow",
    "aten.pow": "pow",
    "aten.mean.dim": "mean",
    "aten.mean": "mean",
    "aten.gt.Tensor": "identity",
    "aten.gt": "identity",
    "aten.triu.default": "triu",
    "aten.triu": "triu",
    "aten.sym_size.int": "sym_size",
    "aten.sym_size": "sym_size",
    "aten._assert_tensor_metadata.default": "identity",
    "aten._assert_tensor_metadata": "identity",
    # Context manager wrappers — skipped, output mapped to input_ids for sym_size
    "wrap_with_set_grad_enabled": "_skip_wrap",
    "aten.arange.start": "arange",
    "aten.matmul": "matmul",
    "aten.matmul.default": "matmul",
    "aten.linear": "linear",
    "aten.linear.default": "linear",
    "aten.mm": "matmul",
    "aten.mm.default": "matmul",
    "aten.bmm": "matmul",
    "aten.gelu": "gelu",
    "aten.gelu.default": "gelu",
    "aten.silu": "silu",
    "aten.silu.default": "silu",
    "aten._softmax": "softmax",
    "aten._softmax.default": "softmax",
    "aten.softmax.int": "softmax",
    "aten.layer_norm": "layer_norm",
    "aten.layer_norm.default": "layer_norm",
    "aten.native_layer_norm": "layer_norm",
    "aten.native_layer_norm.default": "layer_norm",
    "aten.rms_norm": "rms_norm",
    "aten.rms_norm.default": "rms_norm",
    "aten.permute": "permute",
    "aten.permute.default": "permute",
    "aten.transpose": "transpose",
    "aten.transpose.int": "transpose",
    "aten.scaled_dot_product_attention": "scaled_dot_product_attention",
    "aten.scaled_dot_product_attention.default": "scaled_dot_product_attention",
    "aten.cat": "cat",
    "aten.cat.default": "cat",
    "aten.slice.Tensor": "slice",
    "aten.slice_copy.Tensor": "slice",
    "aten.view": "view",
    "aten.view.default": "view",
    "aten.reshape": "view",
    "aten.reshape.default": "view",
    # Activation functions
    "aten.relu": "relu",
    "aten.relu.default": "relu",
    "aten.sub": "sub",
    "aten.sub.Tensor": "sub",
    "aten.rsub": "sub",
    "aten.rsub.Scalar": "sub",
    # Element-wise ops
    "aten.max": "max",
    "aten.max.other": "max",
    # Type conversion (passthrough)
    "aten.to": "identity",
    "aten.to.dtype": "identity",
    "aten.to.dtype_layout": "identity",
    # Misc
    "aten.expand": "identity",
    "aten.expand.default": "identity",
    # Constant creation — computed at runtime from template tensor
    "aten.ones": "ones_like",
    "aten.ones.default": "ones_like",
    "aten.full": "full_like",
    "aten.full.default": "full_like",
    "aten.arange": "arange",
    "aten.arange.default": "arange",
    "aten.cumsum": "identity",
    "aten.cumsum.default": "identity",
    "aten.masked_fill": "identity",
    "aten.masked_fill.Scalar": "identity",
    "aten.masked_fill_": "identity",
    "aten.masked_fill_.Scalar": "identity",
    "aten.lt": "identity",
    "aten.lt.Tensor": "identity",
    # Embedding / unsqueeze
    "aten.embedding": "embedding",
    "aten.embedding.default": "embedding",
    "aten.unsqueeze": "unsqueeze",
    "aten.unsqueeze.default": "unsqueeze",
    # Pass-through / identity ops (no-op during inference)
    "aten.contiguous": "identity",
    "aten.contiguous.default": "identity",
    "aten.clone": "identity",
    "aten.clone.default": "identity",
    "aten.dropout": "identity",
    "aten.dropout.default": "identity",
    "aten.detach": "identity",
    "aten.detach.default": "identity",
    "aten.detach_": "identity",
    "aten.detach_.default": "identity",
    "aten.alias": "identity",
    "aten.alias.default": "identity",
    "aten.type_as": "identity",
    "aten.type_as.default": "identity",
    "aten.lift_fresh_copy": "identity",
    "aten.lift_fresh_copy.default": "identity",
}


def _map_aten_op(target: Any) -> str | None:
    """Map an aten operator to its HAL op name."""
    if isinstance(target, str):
        target_str = target
    elif hasattr(target, "name"):
        target_str = target.name()  # pyright: ignore[reportUnknownMemberType]
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

            # Collect input SSA names and extract non-tensor kwargs
            input_names: list[str] = []
            extra_kwargs: dict[str, Any] = {}
            # Skip scalars that map to known positional attributes
            _scalar_int_positions: dict[str, list[int]] = {
                "softmax": [1],   # dim = args[1]
                "transpose": [1, 2],  # dim0, dim1
                "unsqueeze": [1],  # dim
                "embedding": [2],  # padding_idx
                "full_like": [1],  # fill_value = args[1]
                "triu": [1],  # diagonal = args[1]
                "sym_size": [1],  # dim = args[1]
            }
            skip_positions: list[int] = _scalar_int_positions.get(hal_op, [])
            # Map position → kwarg name for scalar attributes
            _scalar_kwarg_names: dict[str, dict[int, str]] = {
                "full_like": {1: "fill_value"},
                "triu": {1: "diagonal"},
                "sym_size": {1: "dim"},
                "softmax": {1: "dim"},
                "unsqueeze": {1: "dim"},
                "transpose": {1: "dim0", 2: "dim1"},
            }
            scalar_kwargs: dict[int, str] = _scalar_kwarg_names.get(hal_op, {})

            for i, arg in enumerate(node.args):
                if isinstance(arg, torch.fx.Node):
                    input_names.append(ssa_map.get(arg.name, arg.name))
                elif isinstance(arg, (int, float)) and not isinstance(arg, bool):
                    if i in skip_positions:
                        # Extract scalar as a kwarg
                        kwarg_name = scalar_kwargs.get(i)
                        if kwarg_name:
                            extra_kwargs.setdefault(kwarg_name, arg)
                        continue
                    const_name = f"_const_{name_counter}"
                    name_counter += 1
                    weights[const_name] = torch.tensor(arg)
                    input_names.append(const_name)
                elif isinstance(arg, (list, tuple)):
                    if hal_op == "view" and "shape" not in extra_kwargs:
                        resolved: list[str | int] = []
                        for s in arg:
                            if isinstance(s, torch.fx.Node):
                                ssa_name = ssa_map.get(s.name, s.name)
                                resolved.append(ssa_name)
                                input_names.append(ssa_name)
                            else:
                                resolved.append(_symint_for_view(s))
                        extra_kwargs["shape"] = tuple(resolved)
                    elif hal_op == "layer_norm" and "normalized_shape" not in extra_kwargs:
                        extra_kwargs["normalized_shape"] = tuple(arg)
                    elif hal_op in ("ones_like", "full_like") and "shape" not in extra_kwargs:
                        shape_resolved: list[str | int] = []
                        for s in arg:
                            if isinstance(s, torch.fx.Node):
                                ssa_name = ssa_map.get(s.name, s.name)
                                shape_resolved.append(ssa_name)
                                input_names.append(ssa_name)
                            else:
                                shape_resolved.append(_symint_to_int(s) or 1)
                        extra_kwargs["shape"] = tuple(shape_resolved)

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

            ir_ops.append(IrOp(name=hal_op, inputs=input_names, outputs=[output_name], attributes=kwargs))
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
