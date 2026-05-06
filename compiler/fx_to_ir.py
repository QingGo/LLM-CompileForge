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
}


def _map_aten_op(target: str | torch._ops.OpOverload) -> str | None:
    """Map an aten operator to its HAL op name."""
    target_str: str = target if isinstance(target, str) else target.name()  # type: ignore[no-untyped-call]
    # Try exact match first
    if target_str in _ATEN_TO_HAL:
        return _ATEN_TO_HAL[target_str]
    # Try prefix match (strip overload suffix)
    parts = target_str.rsplit(".", 1)
    base = parts[0] if len(parts) == 2 else target_str
    for aten_key, hal_name in _ATEN_TO_HAL.items():
        if aten_key.startswith(base):
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
        node = graph.find_node(inp_name)
        if node is not None and "val" in node.meta:
            fake = node.meta["val"]
            shape: tuple[int | None, ...] = tuple(fake.shape)
            dtype = str(fake.dtype).replace("torch.", "")
            func_inputs.append((inp_name, IrType(dtype=dtype, shape=shape)))
        else:
            func_inputs.append((inp_name, IrType(dtype="float32")))

    # Map parameter/buffer names
    for param_name in sig.inputs_to_parameters:
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
                # Unsupported op — skip with a placeholder
                continue

            # Collect input SSA names
            input_names: list[str] = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    input_names.append(ssa_map.get(arg.name, arg.name))
                elif isinstance(arg, (int, float)):
                    # Inline constant as weight
                    const_name = f"_const_{name_counter}"
                    name_counter += 1
                    weights[const_name] = torch.tensor(arg)
                    input_names.append(const_name)

            output_name = node.name or f"_out_{name_counter}"
            name_counter += 1
            ssa_map[node.name] = output_name

            kwargs = _extract_node_kwargs(node)
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
