"""MLIR text emitter — generates real MLIR from IrModule.

Maps the post-optimization IrModule ops to standard MLIR dialects
(linalg, arith, chlo, math) and a custom `serveforge` (sf) dialect
for operations without clean MLIR equivalents.

The emitted MLIR is valid text that can be parsed by `mlir-opt`,
`mlir-translate`, and other MLIR ecosystem tools.

Reference: design doc §4.2 — FX → MLIR conversion with operator mapping table.
"""

from __future__ import annotations

from typing import Any

from compiler.ir import IrFunction, IrModule, IrOp

# ── Op → MLIR dialect mapping ───────────────────────────────
#
# Key design decision (aligned with §4.2.4):
#   - Standard arithmetic / linalg ops → arith / linalg / chlo
#   - Shape / layout / custom → sf. custom dialect
#   - Fused ops → sf. custom dialect

_OP_MLIR_MAP: dict[str, tuple[str, str]] = {
    "matmul": ("linalg", "matmul"),
    "linear": ("linalg", "matmul"),
    "softmax": ("linalg", "softmax"),
    "permute": ("linalg", "transpose"),
    "transpose": ("linalg", "transpose"),
    "add": ("arith", "addf"),
    "sub": ("arith", "subf"),
    "mul": ("arith", "mulf"),
    "div": ("arith", "divf"),
    "neg": ("arith", "negf"),
    "max": ("arith", "maximumf"),
    "gt": ("arith", "cmpf.ogt"),
    "lt": ("arith", "cmpf.olt"),
    "rsqrt": ("math", "rsqrt"),
    "pow": ("math", "powf"),
    "gelu": ("chlo", "gelu"),
    "silu": ("chlo", "silu"),
    "relu": ("chlo", "relu"),
    "rms_norm": ("sf", "rms_norm"),
    "layer_norm": ("sf", "layer_norm"),
    "scaled_dot_product_attention": ("sf", "sdpa"),
    "cat": ("sf", "concatenate"),
    "slice": ("sf", "slice"),
    "view": ("sf", "reshape"),
    "embedding": ("sf", "embedding"),
    "unsqueeze": ("sf", "unsqueeze"),
    "triu": ("sf", "triu"),
    "cumsum": ("sf", "cumsum"),
    "masked_fill": ("sf", "masked_fill"),
    "ones_like": ("sf", "ones_like"),
    "full_like": ("sf", "full_like"),
    "arange": ("sf", "arange"),
    "expand": ("sf", "expand"),
    "identity": ("sf", "identity"),
    "sym_size": ("sf", "sym_size"),
    "constant": ("sf", "constant"),
    "mean": ("sf", "reduce_mean"),
    "fused_rms_norm_matmul": ("sf", "fused_rms_norm_matmul"),
    "fused_silu_mul": ("sf", "fused_silu_mul"),
}


def _map_op_to_mlir(op_name: str) -> tuple[str, str]:
    return _OP_MLIR_MAP.get(op_name, ("sf", op_name))


def _dtype_to_mlir(dtype: str) -> str:
    mapping: dict[str, str] = {
        "float32": "f32",
        "float16": "f16",
        "bfloat16": "bf16",
        "float64": "f64",
        "int64": "i64",
        "int32": "i32",
        "int16": "i16",
        "int8": "i8",
        "bool": "i1",
    }
    return mapping.get(dtype, "f32")


def _tensor_type_str(dtype: str, shape: tuple[int | None, ...]) -> str:
    """Build an MLIR tensor type string, e.g. 'tensor<?x4xf32>'."""
    if not shape:
        return "tensor<f32>"
    dims = "x".join(str(d) if d is not None else "?" for d in shape)
    return f"tensor<{dims}x{_dtype_to_mlir(dtype)}>"


def _attr_to_mlir(value: Any) -> str:
    """Convert a Python value to MLIR attribute syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:e}"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (tuple, list)):
        inner = ", ".join(_attr_to_mlir(v) for v in value)
        return f"[{inner}]"
    if value is None:
        return "none"
    return str(value)


class MLIREmitter:
    """Emits MLIR text from an IrModule."""

    def __init__(self, module: IrModule) -> None:
        self._module = module
        self._lines: list[str] = []
        self._indent = 0
        self._ssa_counter = 0
        self._value_ssa: dict[str, str] = {}

    def emit(self) -> str:
        """Generate the full MLIR text."""
        self._lines = []
        self._append("module {")
        self._indent = 2

        for func in self._module.functions:
            self._emit_function(func)

        self._indent = 0
        self._append("}")
        return "\n".join(self._lines) + "\n"

    # ── Function emission ─────────────────────────────────

    def _emit_function(self, func: IrFunction) -> None:
        self._value_ssa = {}
        self._ssa_counter = 0

        # Build argument SSA
        arg_decls: list[str] = []
        for name, tp in func.inputs:
            ssa = self._new_ssa()
            self._value_ssa[name] = ssa
            mlir_type = _tensor_type_str(tp.dtype, tp.shape)
            arg_decls.append(f"{ssa}: {mlir_type}")

        args_str = ", ".join(arg_decls)

        # Return type
        if func.outputs:
            out_types = [_tensor_type_str(tp.dtype, tp.shape) for _, tp in func.outputs]
            ret_type = out_types[0] if len(out_types) == 1 else f"({', '.join(out_types)})"
        else:
            ret_type = "()"

        self._append(f"func.func @{func.name}({args_str}) -> {ret_type} {{")
        self._indent += 2

        # Emit weight constants
        self._emit_weights(func)

        # Emit ops
        for op in func.ops:
            self._emit_op(op, func)

        # Build return values
        ret_parts: list[str] = []
        for out_name, _ in func.outputs:
            ret_parts.append(self._value_ssa.get(out_name, "undefined"))
        self._append(f"func.return {', '.join(ret_parts)} : {ret_type}")

        self._indent -= 2
        self._append("}")

    def _emit_weights(self, func: IrFunction) -> None:
        """Emit weight tensors as sf.weight constants."""
        for wname, tensor in func.weights.items():
            ssa = self._new_ssa()
            self._value_ssa[wname] = ssa
            dtype_str = _dtype_to_mlir(str(tensor.dtype).replace("torch.", ""))
            shape_str = "x".join(str(d) for d in tensor.shape)
            self._append(f'{ssa} = "sf.weight"() <{{name = "{wname}"}}> : () -> tensor<{shape_str}x{dtype_str}>')

    # ── Op emission ───────────────────────────────────────

    def _emit_op(self, op: IrOp, func: IrFunction) -> None:
        dialect, mlir_op = _map_op_to_mlir(op.name)
        qualified = f"{dialect}.{mlir_op}"

        # Collect SSA input references
        ssa_inputs: list[str] = []
        for inp_name in op.inputs:
            ssa = self._value_ssa.get(inp_name)
            if ssa is not None:
                ssa_inputs.append(ssa)
            else:
                ssa_inputs.append(inp_name)

        # Assign SSA to outputs
        ssa_outputs: list[str] = []
        for out_name in op.outputs:
            ssa = self._new_ssa()
            ssa_outputs.append(ssa)
            self._value_ssa[out_name] = ssa

        # Build the MLIR operation
        inputs_str = ", ".join(ssa_inputs)
        outputs_str = ", ".join(ssa_outputs)

        # Attributes
        attr_parts = []
        for k, v in sorted(op.attributes.items()):
            attr_parts.append(f"{k} = {_attr_to_mlir(v)}")
        attr_str = f" {{{', '.join(attr_parts)}}}" if attr_parts else ""

        self._append(f'{outputs_str} = "{qualified}"({inputs_str}){attr_str} : () -> ()')

    # ── Helpers ───────────────────────────────────────────

    def _new_ssa(self) -> str:
        ssa = f"%{self._ssa_counter}"
        self._ssa_counter += 1
        return ssa

    def _append(self, line: str) -> None:
        indent = " " * self._indent
        self._lines.append(f"{indent}{line}")


def ir_module_to_mlir(module: IrModule) -> str:
    """Generate MLIR text from an IrModule.

    Usage:
        mlir_text = ir_module_to_mlir(module)
    """
    return MLIREmitter(module).emit()
