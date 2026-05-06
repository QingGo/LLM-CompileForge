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

_ATEN_TO_HAL: dict[str, str] = {
    "aten.add.Tensor": "add",
    "aten.add.Scalar": "add",
    "aten.add": "add",
    "aten.mul.Tensor": "mul",
    "aten.mul.Scalar": "mul",
    "aten.mul": "mul",
    "aten.matmul": "matmul",
    "aten.matmul.default": "matmul",
    "aten.linear": "matmul",
    "aten.linear.default": "matmul",
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
                               "device", "memory_format", "generator"}
        if len(parts) == 2 and parts[1] in overload_candidates:
            base = parts[0]
            if base in _ATEN_TO_HAL:
                return _ATEN_TO_HAL[base]
            # Try matching keys that start with 'base.'
            for aten_key, hal_name in _ATEN_TO_HAL.items():
                if aten_key == base or aten_key.startswith(base + "."):
                    return hal_name
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
            shape: tuple[int | None, ...] = tuple(fake.shape)
            dtype = str(fake.dtype).replace("torch.", "")
            func_inputs.append((inp_name, IrType(dtype=dtype, shape=shape)))
        else:
            func_inputs.append((inp_name, IrType(dtype="float32")))

    # Map parameter/buffer names — build weight_name_map from input_specs
    weight_name_map: dict[str, str] = {}
    if hasattr(sig, "input_specs"):
        for spec in sig.input_specs:
            if spec.kind.value in (2,):  # InputKind.PARAMETER
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

            # Collect input SSA names and extract non-tensor kwargs
            input_names: list[str] = []
            extra_kwargs: dict[str, Any] = {}
            # Skip scalars that map to known positional attributes
            _scalar_int_positions: dict[str, list[int]] = {
                "softmax": [1],   # dim = args[1]
                "transpose": [1, 2],  # dim0, dim1
                "unsqueeze": [1],  # dim
                "embedding": [2],  # padding_idx
            }
            skip_positions: list[int] = _scalar_int_positions.get(hal_op, [])

            for i, arg in enumerate(node.args):
                if isinstance(arg, torch.fx.Node):
                    input_names.append(ssa_map.get(arg.name, arg.name))
                elif isinstance(arg, (int, float)) and not isinstance(arg, bool):
                    if i in skip_positions:
                        continue  # This scalar is a positional attribute, not an input
                    const_name = f"_const_{name_counter}"
                    name_counter += 1
                    weights[const_name] = torch.tensor(arg)
                    input_names.append(const_name)
                elif isinstance(arg, (list, tuple)):
                    if hal_op == "view" and "shape" not in extra_kwargs:
                        extra_kwargs["shape"] = tuple(arg)
                    elif hal_op == "layer_norm" and "normalized_shape" not in extra_kwargs:
                        extra_kwargs["normalized_shape"] = tuple(arg)
                    elif hal_op in ("ones_like", "full_like") and "shape" not in extra_kwargs:
                        extra_kwargs["shape"] = tuple(arg)

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
                int_args = [a for a in node.args if isinstance(a, int) and not isinstance(a, bool)]
                if int_args:
                    kwargs["dim"] = int_args[0]
            if hal_op == "unsqueeze" and "dim" not in kwargs:
                int_args = [a for a in node.args if isinstance(a, int) and not isinstance(a, bool)]
                if int_args:
                    kwargs["dim"] = int_args[0]

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
                        out_shape = tuple(fake.shape)
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
