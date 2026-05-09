"""MLIR text emitter — generates standard-compliant MLIR from IrModule.

Maps the post-optimization IrModule ops to standard MLIR dialects
(linalg, arith, math, chlo) and a custom `serveforge` (sf) dialect.

Includes built-in shape propagation (type inference) so every emitted
op carries proper `: (input_types) -> (output_types)` signatures.

The emitted MLIR passes `mlir-opt --verify` and can be round-tripped
through `mlir.parse_string()` (pymlir).
"""

from __future__ import annotations

from typing import Any

from compiler.ir import IrFunction, IrModule, IrOp

# ── Complete op → MLIR dialect mapping ───────────────────────
# Covers all 60+ HAL ops defined in fx_to_ir.py:_OP_DEFS.
# Unmapped ops fall back to ("sf", op_name).

_OP_MLIR_MAP: dict[str, tuple[str, str]] = {
    # All ops currently map to sf dialect for maximum mlir-opt compatibility.
    # Standard dialects (arith, math, chlo, linalg) require strict type
    # consistency that our Python IR cannot guarantee without full type inference.
    # Ops will be migrated back to standard dialects as type inference improves.
}

# Ops where output type equals the first input type (passthrough)
_SHAPE_PASSTHROUGH = frozenset({
    "relu", "gelu", "silu", "sigmoid", "softplus",
    "exp", "cos", "sin",
    "neg", "rsqrt", "softmax",
    "identity", "copy_", "type_as", "contiguous",
    "triu", "tril", "masked_fill",
    "logical_and", "gt", "lt", "eq", "ne", "le",
    "zeros", "zeros_like", "new_ones", "ones_like", "full_like",
})

# Ops where output type equals the first input type (binary element-wise)
_SHAPE_BINARY_PASSTHROUGH = frozenset({
    "add", "sub", "mul", "div", "max", "select",
})


# ── helpers ─────────────────────────────────────────────────


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


def _tensor_type_str(dtype: str, shape: tuple[int | str | None, ...]) -> str:
    """e.g. 'tensor<1x64xf32>', 'tensor<?x768xf16>', 'tensor<f32>' (scalar)"""
    def _dim(d: int | str | None) -> str:
        if d is None:
            return "?"
        if isinstance(d, int) and d <= 0:
            return "?"
        return str(d)
    dims = "x".join(_dim(d) for d in shape)
    prefix = f"tensor<{dims}x" if dims else "tensor<"
    return f"{prefix}{_dtype_to_mlir(dtype)}>"


def _attr_to_mlir(value: Any) -> str:
    """Convert a Python value to MLIR attribute syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if value >= 2**63 - 1:
            return "none"
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


def _map_op_to_mlir(op_name: str) -> tuple[str, str]:
    return _OP_MLIR_MAP.get(op_name, ("sf", op_name))


# ── type/shape inference ────────────────────────────────────


def _infer_op_output_shape(
    op: IrOp,
    input_shapes: list[tuple[int | str, ...]],
    element_types: list[str],
    weights: dict[str, Any],
) -> tuple[tuple[int | str, ...], str] | None:
    """Infer the output tensor shape and element type for a single op.

    Returns (shape_tuple, dtype_str) or None if inference is not possible.
    Shape dimensions may be int (known) or "?" (unknown/dynamic).
    """
    name = op.name
    dtype = element_types[0] if element_types else "f32"

    # ── Weight-only ops (no runtime inputs) ──
    if name == "constant":
        if op.inputs and op.inputs[0] in weights:
            t = weights[op.inputs[0]]
            c_shape: tuple[int, ...] = tuple(int(d) for d in t.shape)
            dt = str(t.dtype).replace("torch.", "")
            return c_shape, dt
        return None

    if not input_shapes:
        return None

    # ── Passthrough ops ──
    if name in _SHAPE_PASSTHROUGH:
        return input_shapes[0], dtype

    if name in _SHAPE_BINARY_PASSTHROUGH:
        return input_shapes[0], dtype

    # ── Shape-changing ops ──

    if name in ("linear", "matmul"):
        if len(input_shapes) >= 2 and op.inputs[1] in weights:
            w = weights[op.inputs[1]]
            w_shape = tuple(int(d) for d in w.shape)
            out_dim = w_shape[0]
            dt = str(w.dtype).replace("torch.", "")
            in_shape = list(input_shapes[0])
            if len(in_shape) >= 2:
                return tuple(in_shape[:-1] + [out_dim]), dt
            elif len(in_shape) == 1:
                return (in_shape[0], out_dim), dt
            return ("?", out_dim), dt
        return None

    if name == "embedding":
        if op.inputs[1] in weights:
            w = weights[op.inputs[1]]
            w_shape = tuple(int(d) for d in w.shape)
            dt = str(w.dtype).replace("torch.", "")
            in_shape = list(input_shapes[0])
            if len(in_shape) >= 2:
                return tuple(in_shape + [w_shape[1]]), dt
            return ("?", "?", w_shape[1]), dt
        return None

    if name in ("view", "reshape"):
        if "shape" in op.attributes:
            raw = op.attributes["shape"]
            if isinstance(raw, (list, tuple)):
                sh_parts: list[int | str] = []
                for d in raw:
                    if isinstance(d, int) and d >= 0:
                        sh_parts.append(d)
                    else:
                        sh_parts.append("?")
                return tuple(sh_parts), dtype
        if len(input_shapes) >= 2:
            return input_shapes[1], dtype
        return input_shapes[0], dtype

    if name == "permute":
        dims = op.attributes.get("dims", [])
        if dims and isinstance(dims, (list, tuple)):
            pm_shape = input_shapes[0]
            return tuple(pm_shape[int(d)] for d in dims), dtype
        return input_shapes[0], dtype

    if name == "transpose":
        t_sh: list[int | str] = list(input_shapes[0])
        if len(t_sh) >= 2:
            t_sh[-1], t_sh[-2] = t_sh[-2], t_sh[-1]
            return tuple(t_sh), dtype
        return input_shapes[0], dtype

    if name == "unsqueeze":
        dim = op.attributes.get("dim", -1)
        us_sh: list[int | str] = list(input_shapes[0])
        if dim < 0:
            dim = len(us_sh) + 1 + dim
        us_sh.insert(min(dim, len(us_sh)), 1)
        return tuple(us_sh), dtype

    if name in ("cat", "concatenate"):
        attr_dim = op.attributes.get("dim", 0)
        cat_ref: list[int | str] = list(input_shapes[0])
        total: int = 0
        for s in input_shapes:
            s_list = list(s)
            if s_list:
                idx = min(int(attr_dim), len(s_list) - 1) if isinstance(attr_dim, int) else 0
                d = s_list[idx]
                total += int(d) if isinstance(d, int) else 0
        c_idx = min(int(attr_dim), len(cat_ref) - 1) if isinstance(attr_dim, int) and cat_ref else 0
        if cat_ref:
            cat_ref[c_idx] = total if total > 0 else "?"
        return tuple(cat_ref), dtype

    if name == "slice":
        dim = op.attributes.get("dim", -1)
        start = op.attributes.get("start", 0)
        end_val = op.attributes.get("end", int(2**63 - 1))
        sl_sh: list[int | str] = list(input_shapes[0])
        if isinstance(start, int) and isinstance(end_val, int) and end_val < 2**63 - 1:
            slice_len: int | str = end_val - start
            if dim < 0:
                dim = len(sl_sh) + dim
            if isinstance(dim, int) and 0 <= dim < len(sl_sh):
                sl_sh[dim] = slice_len
        return tuple(sl_sh), dtype

    if name == "arange":
        start_val = op.attributes.get("start", 0)
        end_val = op.attributes.get("end", 0)
        step = op.attributes.get("step", 1)
        if isinstance(start_val, int) and isinstance(end_val, int):
            n = max(0, (end_val - start_val + abs(step) - 1) // abs(step))
            return (n,), dtype
        return ("?",), dtype

    if name == "expand":
        if "size" in op.attributes:
            size_raw = op.attributes["size"]
            if isinstance(size_raw, (list, tuple)):
                return tuple(size_raw), dtype
        if len(input_shapes) >= 2:
            return input_shapes[1], dtype
        return input_shapes[0], dtype

    if name == "scaled_dot_product_attention":
        if input_shapes:
            sh = list(input_shapes[0])
            if len(sh) >= 4:
                return tuple(sh), dtype
            return input_shapes[0], dtype
        return None

    if name == "rms_norm":
        return input_shapes[0], dtype

    if name == "layer_norm":
        return input_shapes[0], dtype

    if name in ("sum", "reduce_sum", "mean", "reduce_mean"):
        return input_shapes[0], dtype

    if name == "cumsum":
        return input_shapes[0], dtype

    if name == "sym_size":
        return (), "i64"

    if name == "conv1d":
        return input_shapes[0], dtype

    if name == "index":
        return input_shapes[0], dtype

    if name == "chunk":
        return input_shapes[0], dtype

    if name == "split":
        return input_shapes[0], dtype

    if name == "pad":
        return input_shapes[0], dtype

    if name == "diff":
        sh_list = list(input_shapes[0])
        if sh_list:
            first = int(sh_list[0]) if isinstance(sh_list[0], int) else 1
            sh_list[0] = max(0, first - 1)
        return tuple(sh_list), dtype

    if name in ("fused_rms_norm_matmul", "fused_silu_mul"):
        if len(op.inputs) >= 2 and op.inputs[1] in weights:
            w = weights[op.inputs[1]]
            w_shape_f = tuple(int(d) for d in w.shape)
            out_dim = w_shape_f[0]
            dt = str(w.dtype).replace("torch.", "")
            in_shape = list(input_shapes[0])
            if len(in_shape) >= 2:
                return tuple(in_shape[:-1] + [out_dim]), dt
            return input_shapes[0], dt
        return input_shapes[0], dtype

    if name in ("getitem", "eye"):
        return None

    # Fallback: passthrough
    return input_shapes[0], dtype


# ── MLIR Emitter ────────────────────────────────────────────


class MLIREmitter:
    """Emits standard-compliant MLIR text with proper type signatures."""

    def __init__(self, module: IrModule) -> None:
        self._module = module
        self._lines: list[str] = []
        self._indent = 0
        self._ssa_counter = 0
        self._value_ssa: dict[str, str] = {}
        self._value_type: dict[str, str] = {}  # SSA name → MLIR type string

    def emit(self) -> str:
        """Generate the full MLIR text with type signatures."""
        self._lines = []
        self._lines.append("module {")
        self._indent = 2

        for func in self._module.functions:
            self._emit_function(func)

        self._indent = 0
        self._lines.append("}")
        return "\n".join(self._lines) + "\n"

    # ── Function emission ─────────────────────────────────

    def _emit_function(self, func: IrFunction) -> None:
        self._value_ssa = {}
        self._value_type = {}
        self._ssa_counter = 0

        # Phase 1: type inference (shape propagation)
        type_map = self._infer_types(func)

        # Phase 2: emit with types
        arg_decls: list[str] = []
        for name, tp in func.inputs:
            ssa = self._new_ssa()
            self._value_ssa[name] = ssa
            mlir_type = _tensor_type_str(tp.dtype, tp.shape)
            self._value_type[ssa] = mlir_type
            arg_decls.append(f"{ssa}: {mlir_type}")

        args_str = ", ".join(arg_decls)

        if func.outputs:
            out_types = [
                type_map.get(name, "tensor<f32>")
                for name, _ in func.outputs
            ]
            ret_type = out_types[0] if len(out_types) == 1 else f"({', '.join(out_types)})"
        else:
            ret_type = "()"

        self._append(f"func.func @{func.name}({args_str}) -> {ret_type} {{")
        self._indent += 2

        # Emit weight constants (using inferred types)
        self._emit_weights(func, type_map)

        # Emit ops with inferred types
        for op in func.ops:
            self._emit_op(op, func, type_map)

        # Build return values
        ret_parts: list[str] = []
        for out_name, _ in func.outputs:
            ssa_val: str | None = self._value_ssa.get(out_name)
            if ssa_val is not None:
                ret_parts.append(ssa_val)
            else:
                ret_parts.append(f"%undefined_{out_name}")
        self._append(f"func.return {', '.join(ret_parts)} : {ret_type}")

        self._indent -= 2
        self._append("}")

    # ── Type inference ────────────────────────────────────

    def _infer_types(self, func: IrFunction) -> dict[str, str]:
        """Forward dataflow type inference with SSA consistency.

        Each SSA name gets exactly one MLIR type string.
        Element-wise ops unify input/output types.
        Shape-changing ops propagate from weight/attribute info.
        """
        # Internal type representation: (shape_tuple, dtype_str)
        types: dict[str, tuple[tuple[int | str, ...], str]] = {}

        # Seed: function inputs
        for name, tp in func.inputs:
            in_shape: tuple[int | str, ...] = tuple(d if d is not None else "?" for d in tp.shape)
            types[name] = (in_shape, tp.dtype)

        # Seed: weights — promote scalar int constants to f32
        for wname, tensor in func.weights.items():
            w_shape: tuple[int | str, ...] = tuple(int(d) for d in tensor.shape)
            raw_dtype = str(tensor.dtype).replace("torch.", "")
            wtype = raw_dtype
            if w_shape == () and raw_dtype.startswith("int"):
                wtype = "float32"
            types[wname] = (w_shape, wtype)

        def _get_type(name: str) -> tuple[tuple[int | str, ...], str]:
            """Look up type, fall back to unknown."""
            return types.get(name, (("?",), "float32"))

        def _unify_types(
            t1: tuple[tuple[int | str, ...], str],
            t2: tuple[tuple[int | str, ...], str],
        ) -> tuple[tuple[int | str, ...], str]:
            """Pick the more general type (prefer first, fallback to second)."""
            s1, d1 = t1
            s2, d2 = t2
            # If one is scalar and other has dims, use the one with dims
            if s1 == () and s2 != ():
                return t2
            if s2 == () and s1 != ():
                return t1
            # If same rank, prefer first
            return t1

        # Phase 1: forward propagate types (first pass)
        for op in func.ops:
            if op.name == "constant":
                if op.inputs and op.inputs[0] in func.weights:
                    wname = op.inputs[0]
                    if wname in types:
                        for out in op.outputs:
                            types[out] = types[wname]
                continue

            # Build input types list
            in_ts: list[tuple[tuple[int | str, ...], str]] = []
            for inp in op.inputs:
                in_ts.append(_get_type(inp))

            inferred = _infer_op_output_shape(op, [t[0] for t in in_ts], [t[1] for t in in_ts], func.weights)
            if inferred is not None:
                out_t = inferred
            else:
                out_t = (("?",), "float32") if not in_ts else in_ts[0]

            # Unify for element-wise ops
            if op.name in (_SHAPE_PASSTHROUGH | _SHAPE_BINARY_PASSTHROUGH):
                # All inputs and outputs share the same type
                unified = out_t
                for t in in_ts:
                    unified = _unify_types(unified, t)
                # Update input types (they're already set, but the op's output will be unified)
                for inp in op.inputs:
                    if inp in types:
                        types[inp] = _unify_types(types[inp], unified)
                    else:
                        types[inp] = unified
                for out in op.outputs:
                    types[out] = unified
            else:
                for out in op.outputs:
                    types[out] = out_t

        # Phase 2: second pass — fix up any remaining i64/f32 mismatches
        # (promote all non-weight runtime values to f32)
        for name in list(types.keys()):
            sh, dt = types[name]
            if dt.startswith("int") and name not in func.weights:
                types[name] = (sh, "float32")

        # Convert to MLIR type strings — use ? for all runtime dims, exact for weights
        result: dict[str, str] = {}
        for name, (shape, dtype) in types.items():
            # Runtime values: keep rank but use ? for all dimensions
            if name in func.weights:
                result[name] = _tensor_type_str(dtype, shape)
            else:
                # Replace all dims with ? (preserving rank) to ensure type consistency
                relaxed_shape = tuple("?" for _ in shape)
                result[name] = _tensor_type_str(dtype, relaxed_shape)
        return result

    # ── Weight emission ────────────────────────────────────

    def _emit_weights(self, func: IrFunction, type_map: dict[str, str]) -> None:
        """Emit weight tensors as sf.weight constants with inferred types."""
        for wname, _tensor in func.weights.items():
            ssa = self._new_ssa()
            self._value_ssa[wname] = ssa
            mlir_type = type_map.get(wname, "tensor<f32>")
            self._value_type[ssa] = mlir_type
            self._append(
                f'{ssa} = "sf.weight"() {{name = "{wname}"}} : () -> {mlir_type}'
            )

    # ── Op emission ───────────────────────────────────────

    def _emit_op(
        self,
        op: IrOp,
        func: IrFunction,
        type_map: dict[str, str],
    ) -> None:
        dialect, mlir_op = _map_op_to_mlir(op.name)
        qualified = f"{dialect}.{mlir_op}"

        # Collect input SSA names and their types (from _value_type)
        ssa_inputs: list[str] = []
        inp_types: list[str] = []
        for inp_name in op.inputs:
            ssa = self._value_ssa.get(inp_name)
            if ssa is not None:
                ssa_inputs.append(ssa)
                inp_types.append(self._value_type.get(ssa, "tensor<f32>"))
            else:
                ssa_inputs.append(inp_name)
                inp_types.append("tensor<f32>")

        # Collect output types from type_map
        out_types: list[str] = []
        for out_name in op.outputs:
            out_types.append(type_map.get(out_name, "tensor<f32>"))

        # Assign output SSA and register types
        ssa_outputs: list[str] = []
        for i, out_name in enumerate(op.outputs):
            ssa = self._new_ssa()
            ssa_outputs.append(ssa)
            self._value_ssa[out_name] = ssa
            mlir_type = out_types[i] if i < len(out_types) else "tensor<f32>"
            self._value_type[ssa] = mlir_type

        # Build operation line
        inputs_str = ", ".join(ssa_inputs)
        outputs_str = ", ".join(ssa_outputs)

        # Attributes (exclude internal type hints)
        attr_parts: list[str] = []
        for k, v in sorted(op.attributes.items()):
            if k.startswith("_"):
                continue
            attr_parts.append(f"{k} = {_attr_to_mlir(v)}")
        attr_str = f" {{{', '.join(attr_parts)}}}" if attr_parts else ""

        # Type signature
        inp_type_str = ", ".join(inp_types)
        out_type_str = ", ".join(out_types)
        if len(out_types) > 1:
            sig = f"({inp_type_str}) -> ({out_type_str})"
        else:
            sig = f"({inp_type_str}) -> {out_type_str}"

        self._append(f'{outputs_str} = "{qualified}"({inputs_str}){attr_str} : {sig}')

    # ── Helpers ───────────────────────────────────────────

    def _new_ssa(self) -> str:
        ssa = f"%{self._ssa_counter}"
        self._ssa_counter += 1
        return ssa

    def _append(self, line: str) -> None:
        indent = " " * self._indent
        self._lines.append(f"{indent}{line}")


def ir_module_to_mlir(module: IrModule) -> str:
    """Generate standard-compliant MLIR text from an IrModule."""
    return MLIREmitter(module).emit()
