"""Lower SF dialect ops to HAL IR JSON.

Reads normalized MLIR (SF primitives only), maps each ``sf.*`` op to a
``hal.execute @op_name`` entry, and produces ``hal_ir.json``.

The pass is **read-only** on the IR — it does not modify the MLIR module.

Op mapping table (all sf.* → hal.execute entries):

  ==========================  =================================
  ``sf.matmul``               ``hal.execute @matmul``
  ``sf.rms_norm``             ``hal.execute @rms_norm``
  ``sf.layer_norm``           ``hal.execute @layer_norm``
  ``sf.softmax``              ``hal.execute @softmax``
  ``sf.add/sub/mul/div/pow``  ``hal.execute @element_wise {kind}``
  ``sf.relu/gelu/silu/...``   ``hal.execute @element_wise {kind}``
  ``sf.tanh/exp/neg/...``     ``hal.execute @element_wise {kind}``
  ``sf.sqrt/rsqrt/cos/sin``   ``hal.execute @element_wise {kind}``
  ``sf.eq/ne/gt/lt/le``       ``hal.execute @compare {kind}``
  ``sf.logical_and``          ``hal.execute @compare {kind}``
  ``sf.view/expand``          ``hal.execute @reshape``
  ``sf.unsqueeze``            ``hal.execute @unsqueeze``
  ``sf.transpose/permute``    ``hal.execute @transpose``
  ``sf.slice``                ``hal.execute @slice``
  ``sf.cat``                  ``hal.execute @concat``
  ``sf.sum/mean``             ``hal.execute @reduce {kind}``
  ``sf.cumsum``               ``hal.execute @scan {kind}``
  ``sf.identity``             *skipped (SSA rename)*
  ``sf.ones_like/new_ones``   ``hal.execute @fill {value=1.0}``
  ``sf.arange``               ``hal.execute @fill {kind=arange}``
  ``sf.embedding``            ``hal.execute @gather``
  ``sf.index``                ``hal.execute @gather (indexed load)``
  ``sf.sym_size``             ``hal.execute @shape_of``
  ``sf.constant``             *inlined as JSON constant*
  ``sf.weight``               *recorded as weight reference*
  ``cache_write``             ``hal.execute @cache_write``  *(inserted)*
  ``cache_read``              ``hal.execute @cache_read``   *(inserted)*
  ==========================  =================================
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from compiler.mlir_dialect.compile_utils import _setup_mlir_path

_log = logging.getLogger(__name__)

# ── Op mapping ──────────────────────────────────────────────────────
# Each entry: (sf_op_name, hal_execute_target, extra_attrs_fn_or_None)

_UNARY_ARITH_MAP: dict[str, str] = {
    "sf.relu": "relu",
    "sf.gelu": "gelu",
    "sf.silu": "silu",
    "sf.sigmoid": "sigmoid",
    "sf.tanh": "tanh",
    "sf.exp": "exp",
    "sf.neg": "neg",
    "sf.softplus": "softplus",
    "sf.sqrt": "sqrt",
    "sf.rsqrt": "rsqrt",
    "sf.cos": "cos",
    "sf.sin": "sin",
}

_BINARY_ARITH_MAP: dict[str, str] = {
    "sf.add": "add",
    "sf.sub": "sub",
    "sf.mul": "mul",
    "sf.div": "div",
    "sf.pow": "pow",
    "sf.max": "max",
}

_COMPARE_MAP: dict[str, str] = {
    "sf.eq": "eq",
    "sf.ne": "ne",
    "sf.gt": "gt",
    "sf.lt": "lt",
    "sf.le": "le",
    "sf.ge": "ge",
    "sf.logical_and": "logical_and",
}


def _strip_mlir_quotes(s: str) -> str:
    """Strip MLIR string attribute quotes from a value.

    MLIR ``StringAttr`` values print as ``"foo"`` (with quotes).
    """
    return s.strip().strip('"')


def _parse_sf_op_name(raw: str) -> str:
    """Normalize ``sf.op_name`` regardless of quoting style.

    In the normalized IR, some ops appear as ``sf.add`` (bare) and others
    as ``"sf.add"`` (quoted string).  Both refer to the same op.
    """
    return _strip_mlir_quotes(raw)


def _parse_mlir_int_attr(attr_str: str | None) -> int | None:
    """Parse an MLIR integer attribute to a Python int.

    Handles formats like ``"0"``, ``"0 : i64"``, ``"1 : i64"``.
    """
    if attr_str is None:
        return None
    s = str(attr_str).strip()
    # Strip type suffix like " : i64"
    if " : " in s:
        s = s.split(" : ")[0]
    try:
        return int(s)
    except ValueError:
        return None


def _infer_dtype_from_type(t: Any) -> str:
    """Infer a short dtype string from an MLIR type."""
    s = str(t)
    if "f32" in s or "float" in s:
        return "f32"
    if "f16" in s or "bfloat" in s:
        return "f16"
    if "i64" in s:
        return "i64"
    if "i32" in s:
        return "i32"
    if "i8" in s or "i1" in s or "bool" in s:
        return "i8"
    return "f32"


def _shape_from_type(t: Any) -> list[int | str]:
    """Extract shape as a list of ints or '?' for dynamic dims from an MLIR type."""
    s = str(t)
    # Extract shape from tensor<...>
    m = re.search(r"tensor<(.+?)x", s)
    if not m:
        return []
    shape_str = s[s.index("<") + 1 : s.rindex("x")]
    parts = shape_str.split("x")
    shape: list[int | str] = []
    for p in parts:
        p = p.strip()
        if p == "?":
            shape.append("?")
        else:
            try:
                shape.append(int(p))
            except ValueError:
                shape.append("?")
    return shape


# ── SSA tracker ─────────────────────────────────────────────────────


class SSATracker:
    """Tracks SSA values within a function block, assigning %local names.

    Uses the MLIR value's string representation (``str(val)``) as the
    stable identity key — this accounts for nanobind wrapper objects that
    may differ across ``id()`` calls.

    Example::

        args:  %arg0 → "%arg0"
        ops:   %result → "%0", "%1", ...
    """

    def __init__(self) -> None:
        self._val_to_name: dict[str, str] = {}
        self._name_to_val: dict[str, str] = {}
        self._counter = 0

    @staticmethod
    def _val_key(val: Any) -> str:
        """Stable key for an MLIR value — its string representation."""
        return str(val)

    def register_arg(self, arg_val: Any, name: str) -> None:
        """Register a function argument."""
        key = self._val_key(arg_val)
        self._val_to_name[key] = name
        self._name_to_val[name] = key

    def register_result(self, op_result: Any) -> str:
        """Register an op result, returning a fresh ``%N`` name."""
        name = f"%{self._counter}"
        self._counter += 1
        key = self._val_key(op_result)
        self._val_to_name[key] = name
        self._name_to_val[name] = key
        return name

    def lookup(self, val: Any) -> str:
        """Return the %name for an SSA value (operand or block arg)."""
        key = self._val_key(val)
        if key in self._val_to_name:
            return self._val_to_name[key]
        # Fallback for values not explicitly tracked (e.g. nested block results)
        return f"%x{self._counter}"


# ── HAL IR Builder ──────────────────────────────────────────────────


class HalIRBuilder:
    """Builds the HAL IR structure from a normalized MLIR module.

    Usage::

        builder = HalIRBuilder()
        builder.load_mlir(mlir_text)
        builder.load_metadata(metadata_dict)
        result = builder.build()
    """

    def __init__(self) -> None:
        self._model_name = "model"
        self._mlir_text: str = ""
        self._metadata: dict[str, Any] = {}
        self._functions: list[dict[str, Any]] = []
        self._cache_policy: dict[str, Any] = {}
        self._weight_classification: dict[str, Any] = {}
        self._hf_key_map: dict[str, str] = {}

    def load_mlir(self, mlir_text: str) -> None:
        """Parse and store the MLIR text."""
        self._mlir_text = mlir_text

    def load_metadata(self, metadata: dict[str, Any]) -> None:
        """Load compilation metadata (from metadata.json)."""
        self._metadata = metadata
        self._cache_policy = metadata.get("cache_policy", {})
        self._weight_classification = metadata.get("weight_classification", {})
        self._hf_key_map = metadata.get("hf_key_map", {})

    def set_model_name(self, name: str) -> None:
        self._model_name = name

    def build(self) -> dict[str, Any]:
        """Build the complete HAL IR structure.

        Returns a dict ready to be serialized as JSON.
        """
        _setup_mlir_path()
        import sys
        import mlir.ir as ir

        # Also add sf-dialect Python bindings to path
        from pathlib import Path as _Path
        _sf_base = _Path(__file__).resolve().parent.parent.parent.parent / "sf-dialect"
        for _sf_candidate in [
            _sf_base / "build" / "python_packages" / "sf",
            _sf_base / "python_packages" / "sf",
        ]:
            if _sf_candidate.is_dir() and str(_sf_candidate) not in sys.path:
                sys.path.insert(0, str(_sf_candidate))

        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        try:
            from mlir_sf._mlir_libs._sfDialectsNanobind import sf
            sf.register_dialects(ctx._CAPIPtr, load=True)
        except ImportError:
            _log.warning("SF dialect bindings not available; "
                         "proceeding without registration")

        with ctx, ir.Location.unknown(ctx):
            module = ir.Module.parse(self._mlir_text, ctx)
            self._build_from_module(module)

        return {
            "model_name": self._model_name,
            "num_functions": len(self._functions),
            "functions": self._functions,
        }

    def _build_from_module(self, module: Any) -> None:
        """Walk the module and build function entries."""
        for op in module.operation.regions[0].blocks[0]:
            op_name = str(op.operation.name)
            if op_name == "func.func":
                func_entry = self._build_function(op)
                self._functions.append(func_entry)

    def _extract_layer(self, func_name: str) -> int | None:
        """Extract layer number from function name pattern like main_1a, main_2b."""
        m = re.search(r"main_(\d+)", func_name)
        if m:
            return int(m.group(1))
        return None

    def _get_weight_classification(
        self, func_name: str
    ) -> dict[str, list[str]]:
        """Return the weight classification for a function (empty if missing)."""
        return self._weight_classification.get(func_name, {
            "params": [],
            "constants": [],
        })

    def _build_function(self, func_op: Any) -> dict[str, Any]:
        """Process a single func.func op and return its HAL IR entry."""
        # Get function name (strip MLIR string quotes from StringAttr)
        func_name = _strip_mlir_quotes(
            str(func_op.attributes.get("sym_name", "unknown"))
        )
        layer_idx = self._extract_layer(func_name)
        # func_type unused but available for debugging
        _ = str(func_op.attributes.get("function_type", ""))

        # Get operands and results
        func_region = func_op.regions[0]
        func_block = func_region.blocks[0]

        # ── Inputs from block args ──────────────────────────────
        inputs: list[dict[str, Any]] = []
        weight_class = self._get_weight_classification(func_name)
        param_names = weight_class.get("params", [])
        const_names = weight_class.get("constants", [])

        ssa = SSATracker()
        for i, arg in enumerate(func_block.arguments):
            arg_name = f"%arg{i}"
            ssa.register_arg(arg, arg_name)
            arg_type = str(arg.type) if hasattr(arg, "type") else "unknown"
            dtype = _infer_dtype_from_type(arg_type)
            shape = _shape_from_type(arg_type)
            inputs.append({
                "name": arg_name,
                "shape": shape,
                "dtype": dtype,
            })

        # ── Detect consumed_internally patterns ─────────────────
        # main_{N}a: Q, K, V outputs — K and V are consumed_internally
        # main_0: attention mask (14th output at index 13) is consumed_internally
        consumed_internally: list[int] = []
        is_kv_func_a = bool(
            re.match(r"main_\d+a$", func_name)
        )
        is_kv_func_b = bool(
            re.match(r"main_\d+b$", func_name)
        )
        is_embed_func = func_name == "main_0"

        if is_kv_func_a:
            # K and V are outputs 1 and 2 (0-indexed: 1, 2)
            consumed_internally = [1, 2]
        elif is_embed_func:
            # The attention mask (%250) is the 14th returned value (index 13)
            # Also the seq_len value (%213) at index 14
            consumed_internally = [13]

        # Determine if block_table is needed (for cache_read)
        needs_block_table = is_kv_func_a or is_kv_func_b

        # ── Walk ops ────────────────────────────────────────────
        ops: list[dict[str, Any]] = []
        weights: list[dict[str, Any]] = []
        constants: list[dict[str, Any]] = []
        weight_index: dict[str, int] = {}  # weight name → index

        for inner_op in func_block:
            raw_name = str(inner_op.operation.name)
            op_name = _parse_sf_op_name(raw_name)

            result_hal = self._lower_op(
                inner_op, op_name, ssa, weights, constants,
                weight_index, param_names, const_names,
            )
            if result_hal is not None:
                ops.append(result_hal)

        # ── Handle consumed_internally: insert cache_write/cache_read ──
        # Only add cache ops for KV layer functions (not embed/main_0).
        if is_kv_func_a and consumed_internally:
            self._add_cache_ops(
                ops, func_name, layer_idx,
                consumed_internally,
                is_kv_func_b,
            )
        if is_kv_func_b:
            # Add cache_read for KV layer b-functions
            self._add_cache_ops(
                ops, func_name, layer_idx,
                [], True,
            )

        # ── Determine outputs ───────────────────────────────────
        return_op = None
        for inner_op in func_block:
            if str(inner_op.operation.name) in ("func.return", "return"):
                return_op = inner_op
                break

        outputs: list[dict[str, Any]] = []
        if return_op is not None:
            for i, operand in enumerate(return_op.operands):
                ot = str(operand.type) if hasattr(operand, "type") else "unknown"
                dtype = _infer_dtype_from_type(ot)
                shape = _shape_from_type(ot)
                out_entry: dict[str, Any] = {
                    "name": ssa.lookup(operand),
                    "shape": shape,
                    "dtype": dtype,
                }
                # Mark consumed_internally in output metadata
                if i in consumed_internally:
                    out_entry["consumed_internally"] = True
                outputs.append(out_entry)

        # ── Build function entry ────────────────────────────────
        func_entry: dict[str, Any] = {
            "name": func_name,
            "layer": layer_idx,
            "inputs": inputs,
            "outputs": outputs,
            "weights": weights,
            "constants": constants,
            "ops": ops,
        }

        if needs_block_table:
            # Add block_table as an input param (from cache_policy)
            func_entry["block_table"] = True

        return func_entry

    def _lower_op(
        self,
        op: Any,
        op_name: str,
        ssa: SSATracker,
        weights: list[dict[str, Any]],
        constants: list[dict[str, Any]],
        weight_index: dict[str, int],
        param_names: list[str],
        const_names: list[str],
    ) -> dict[str, Any] | None:
        """Lower a single sf.* op to a HAL IR entry.

        Returns None for ops that should be skipped (identity, etc).
        """
        operands = list(op.operands) if hasattr(op, "operands") else []
        results = list(op.results) if hasattr(op, "results") else []

        # Get input %names
        input_names = [ssa.lookup(o) for o in operands]

        # Register results and get output %names
        output_names = [ssa.register_result(r) for r in results]

        if op_name == "sf.weight":
            # Record weight — no hal op.  Result already registered above.
            name_attr = op.attributes.get("name", "")
            weight_name = _strip_mlir_quotes(str(name_attr))
            if results:
                result_type = str(results[0].type) if hasattr(results[0], "type") else ""
                shape = _shape_from_type(results[0].type) if hasattr(results[0], "type") else []
                dtype = _infer_dtype_from_type(result_type)
                idx = len(weights)
                weight_index[weight_name] = idx
                weights.append({
                    "name": weight_name,
                    "shape": shape,
                    "dtype": dtype,
                    "hal_name": f"w{idx}",
                })
            return None

        if op_name == "sf.constant":
            # Inline constant value — result already registered above.
            return None

        if op_name == "sf.identity":
            # Skip — map result to same name as input operand
            if operands and results:
                input_name = ssa.lookup(operands[0])
                for r in results:
                    result_key = ssa._val_key(r)
                    ssa._val_to_name[result_key] = input_name
            return None

        if op_name == "sf.sym_size":
            return {
                "op": "shape_of",
                "inputs": input_names,
                "outputs": output_names,
            }

        if op_name == "sf.view" or op_name == "sf.expand":
            shape_attr = op.attributes.get("shape")
            shape_val: list[int | str] = []
            if shape_attr is not None:
                shape_str = str(shape_attr)
                shape_val = self._parse_attr_shape(shape_str)
            entry: dict[str, Any] = {
                "op": "reshape",
                "inputs": input_names,
                "outputs": output_names,
            }
            if shape_val:
                entry["shape"] = shape_val
            return entry

        if op_name == "sf.unsqueeze":
            dim = _parse_mlir_int_attr(
                str(op.attributes.get("dim")) if op.attributes.get("dim") is not None else None
            ) or 0
            return {
                "op": "unsqueeze",
                "inputs": input_names,
                "outputs": output_names,
                "dim": dim,
            }

        if op_name == "sf.transpose":
            dim0 = _parse_mlir_int_attr(
                str(op.attributes.get("dim0")) if op.attributes.get("dim0") is not None else None
            ) or 0
            dim1 = _parse_mlir_int_attr(
                str(op.attributes.get("dim1")) if op.attributes.get("dim1") is not None else None
            ) or 1
            return {
                "op": "transpose",
                "inputs": input_names,
                "outputs": output_names,
                "dims": [dim0, dim1],
            }

        if op_name == "sf.slice":
            dim = _parse_mlir_int_attr(
                str(op.attributes.get("dim")) if op.attributes.get("dim") is not None else None
            ) or 0
            start = _parse_mlir_int_attr(
                str(op.attributes.get("start")) if op.attributes.get("start") is not None else None
            ) or 0
            end_raw_attr = op.attributes.get("end")
            end_raw = str(end_raw_attr) if end_raw_attr is not None else "MAX"
            # MLIR uses MAX_INT for "to end"
            parsed_end = _parse_mlir_int_attr(end_raw)
            if parsed_end is not None and abs(parsed_end) > 1_000_000_000:
                end_raw = "MAX"
            return {
                "op": "slice",
                "inputs": input_names,
                "outputs": output_names,
                "dim": dim,
                "start": start,
                "end": end_raw,
            }

        if op_name == "sf.cat":
            dim = _parse_mlir_int_attr(
                str(op.attributes.get("dim")) if op.attributes.get("dim") is not None else None
            ) or 0
            return {
                "op": "concat",
                "inputs": input_names,
                "outputs": output_names,
                "dim": dim,
            }

        if op_name == "sf.mean":
            return {
                "op": "reduce",
                "inputs": input_names,
                "outputs": output_names,
                "kind": "mean",
            }

        if op_name == "sf.sum":
            return {
                "op": "reduce",
                "inputs": input_names,
                "outputs": output_names,
                "kind": "sum",
            }

        if op_name in _BINARY_ARITH_MAP:
            return {
                "op": "element_wise",
                "inputs": input_names,
                "outputs": output_names,
                "kind": _BINARY_ARITH_MAP[op_name],
            }

        if op_name in _UNARY_ARITH_MAP:
            return {
                "op": "element_wise",
                "inputs": input_names,
                "outputs": output_names,
                "kind": _UNARY_ARITH_MAP[op_name],
            }

        if op_name in _COMPARE_MAP:
            return {
                "op": "compare",
                "inputs": input_names,
                "outputs": output_names,
                "kind": _COMPARE_MAP[op_name],
            }

        if op_name == "sf.embedding":
            return {
                "op": "gather",
                "inputs": input_names,
                "outputs": output_names,
            }

        if op_name == "sf.index":
            return {
                "op": "gather",
                "inputs": input_names,
                "outputs": output_names,
                "mode": "indexed",
            }

        if op_name == "sf.matmul":
            return {
                "op": "matmul",
                "inputs": input_names,
                "outputs": output_names,
            }

        if op_name == "sf.softmax":
            return {
                "op": "softmax",
                "inputs": input_names,
                "outputs": output_names,
            }

        if op_name == "sf.ones_like" or op_name == "sf.new_ones":
            dtype_attr = op.attributes.get("dtype")
            raw_dtype = str(dtype_attr) if dtype_attr is not None else "f32"
            dtype_str = _strip_mlir_quotes(raw_dtype)
            return {
                "op": "fill",
                "inputs": input_names,
                "outputs": output_names,
                "value": 1.0,
                "dtype": dtype_str,
            }

        if op_name == "sf.arange":
            return {
                "op": "fill",
                "inputs": input_names,
                "outputs": output_names,
                "kind": "arange",
            }

        if op_name == "sf.rms_norm":
            return {
                "op": "rms_norm",
                "inputs": input_names,
                "outputs": output_names,
            }

        if op_name == "sf.layer_norm":
            return {
                "op": "layer_norm",
                "inputs": input_names,
                "outputs": output_names,
            }

        if op_name == "sf.silu":
            return {
                "op": "element_wise",
                "inputs": input_names,
                "outputs": output_names,
                "kind": "silu",
            }

        if op_name == "sf.relu":
            return {
                "op": "element_wise",
                "inputs": input_names,
                "outputs": output_names,
                "kind": "relu",
            }

        if op_name == "sf.cumsum":
            return {
                "op": "scan",
                "inputs": input_names,
                "outputs": output_names,
                "kind": "cumsum",
            }

        if op_name == "sf.gelu":
            return {
                "op": "element_wise",
                "inputs": input_names,
                "outputs": output_names,
                "kind": "gelu",
            }

        if op_name == "sf.sigmoid":
            return {
                "op": "element_wise",
                "inputs": input_names,
                "outputs": output_names,
                "kind": "sigmoid",
            }

        # Unknown op — skip non-SF ops silently, warn for unrecognized SF ops
        if op_name in ("func.return", "return"):
            return None
        _log.warning("Unknown SF op %s in function, passing through", op_name)
        return {
            "op": op_name.removeprefix("sf."),
            "inputs": input_names,
            "outputs": output_names,
        }

    def _parse_attr_shape(self, shape_str: str) -> list[int | str]:
        """Parse an MLIR shape attribute like ``[-1, -1, 12, 64]``."""
        shape_str = shape_str.strip()
        if shape_str.startswith("["):
            shape_str = shape_str[1:]
        if shape_str.endswith("]"):
            shape_str = shape_str[:-1]
        parts = shape_str.split(",")
        shape: list[int | str] = []
        for p in parts:
            p = p.strip()
            if p in ("-1", "?"):
                shape.append("?")
            else:
                try:
                    shape.append(int(p))
                except ValueError:
                    shape.append("?")
        return shape

    def _add_cache_ops(
        self,
        ops: list[dict[str, Any]],
        func_name: str,
        layer_idx: int | None,
        consumed_indices: list[int],
        is_b_func: bool,
    ) -> None:
        """Insert cache_write and cache_read ops for consumed_internally outputs.

        For main_{N}a: after computing K and V, add cache_write.
        For main_{N}b: cache_read happens before using K, V.
        """
        if layer_idx is None:
            return

        # We need to know which SSA values correspond to the consumed outputs.
        # These are the last 2 outputs (K and V) for main_{N}a functions.
        # We track this by looking at the function's return op later.

        if is_b_func:
            # For main_{N}b: add cache_read at the start of ops
            # The K and V inputs to this function come from cache
            cache_read_entry: dict[str, Any] = {
                "op": "cache_read",
                "layer": layer_idx,
                "inputs": ["%block_table"],
                "outputs": [f"%cache_k_{layer_idx}", f"%cache_v_{layer_idx}"],
            }
            ops.insert(0, cache_read_entry)
        else:
            # For main_{N}a: add cache_write after the ops that produce K, V
            cache_write_entry: dict[str, Any] = {
                "op": "cache_write",
                "layer": layer_idx,
                "inputs": [],  # Will be filled with K and V output names
                "outputs": [],
            }
            ops.append(cache_write_entry)


# ── Top-level entry point ────────────────────────────────────────────


def lower_sf_to_hal(
    mlir_text: str,
    model_name: str = "model",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lower normalized SF MLIR to HAL IR JSON structure.

    Args:
        mlir_text: The normalized MLIR text (SF primitives only).
        model_name: Name of the model (used in JSON output).
        metadata: Optional metadata dict from metadata.json for weight info.

    Returns:
        A dict representing the ``hal_ir.json`` structure.
    """
    builder = HalIRBuilder()
    builder.load_mlir(mlir_text)
    builder.set_model_name(model_name)
    if metadata:
        builder.load_metadata(metadata)
    return builder.build()


def lower_sf_to_hal_file(
    mlir_path: str | Path,
    output_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> str:
    """Lower a normalized MLIR file to a ``hal_ir.json`` file.

    Args:
        mlir_path: Path to the normalized MLIR file.
        output_path: Path for the output JSON. If None, derived from mlir_path.
        metadata_path: Optional path to metadata.json.

    Returns:
        The path to the written JSON file.
    """
    mlir_path = Path(mlir_path)

    # Determine model name from directory
    model_dir = mlir_path.parent
    model_name = model_dir.name

    # Read MLIR
    mlir_text = mlir_path.read_text()

    # Read metadata if available
    metadata: dict[str, Any] = {}
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text())
    else:
        # Try sibling metadata.json
        meta_candidate = model_dir / "metadata.json"
        if meta_candidate.is_file():
            metadata = json.loads(meta_candidate.read_text())

    # Determine output path
    if output_path is None:
        output_path = model_dir / "hal_ir.json"
    output_path = Path(output_path)

    # Build HAL IR
    result = lower_sf_to_hal(mlir_text, model_name, metadata)

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    # Summary
    total_ops = sum(len(f["ops"]) for f in result["functions"])
    _log.info(
        "Lowered %d ops to hal.execute across %d functions → %s",
        total_ops,
        result["num_functions"],
        output_path,
    )
    print(
        f"[HAL IR] Lowered {total_ops} ops across {result['num_functions']} functions "
        f"→ {output_path}"
    )

    return str(output_path)


# ── Main entry point ─────────────────────────────────────────────────


def main() -> None:
    """CLI entry point: lower normalized MLIR to hal_ir.json."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Lower normalized SF MLIR to HAL IR JSON",
    )
    parser.add_argument(
        "mlir_path",
        nargs="?",
        default="compiled/opt_125m_kv/model.normalized.mlir",
        help="Path to normalized MLIR file",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: <model_dir>/hal_ir.json)",
    )
    parser.add_argument(
        "--metadata", "-m",
        default=None,
        help="Path to metadata.json (default: auto from model_dir)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    result_path = lower_sf_to_hal_file(
        args.mlir_path,
        output_path=args.output,
        metadata_path=args.metadata,
    )
    print(f"Done: {result_path}")


if __name__ == "__main__":
    main()
