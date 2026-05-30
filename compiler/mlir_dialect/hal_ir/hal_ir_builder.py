"""HAL IR Builder — builds the HAL IR JSON structure from normalized MLIR.

Usage::

    builder = HalIRBuilder()
    builder.load_mlir(mlir_text)
    builder.load_metadata(metadata_dict)
    result = builder.build()
"""

from __future__ import annotations

import logging
import re
from typing import Any

from compiler.mlir_dialect.compile_utils import _setup_mlir_path
from compiler.mlir_dialect.hal_ir.op_lowering import (
    infer_dtype_from_type,
    lower_op,
    parse_sf_op_name,
    shape_from_type,
    strip_mlir_quotes,
)
from compiler.mlir_dialect.hal_ir.ssa_tracker import SSATracker

_log = logging.getLogger(__name__)


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
        from pathlib import Path as _Path

        import mlir.ir as ir

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
        func_name = strip_mlir_quotes(
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
            dtype = infer_dtype_from_type(arg_type)
            shape = shape_from_type(arg_type)
            inputs.append({
                "name": arg_name,
                "shape": shape,
                "dtype": dtype,
            })

        # ── Detect consumed_internally patterns ─────────────────
        # main_{N}a: Q, K, V outputs — K and V are consumed_internally
        # main_0: attention mask (14th output at index 13) is consumed_internally
        consumed_internally: list[int] = []
        is_kv_func_a = bool(re.match(r"main_\d+a$", func_name))
        is_kv_func_b = bool(re.match(r"main_\d+b$", func_name))
        is_embed_func = func_name == "main_0"

        if is_kv_func_a:
            # K and V are outputs 1 and 2 (0-indexed: 1, 2)
            consumed_internally = [1, 2]
        elif is_embed_func:
            # The attention mask (%250) is the 14th returned value (index 13)
            consumed_internally = [13]

        # Determine if block_table is needed (for cache_read)
        needs_block_table = is_kv_func_a or is_kv_func_b

        # ── Walk ops ────────────────────────────────────────────
        ops: list[dict[str, Any]] = []
        weights: list[dict[str, Any]] = []
        constants: list[dict[str, Any]] = []
        weight_index: dict[str, int] = {}

        for inner_op in func_block:
            raw_name = str(inner_op.operation.name)
            op_name = parse_sf_op_name(raw_name)

            result_hal = lower_op(
                inner_op, op_name, ssa, weights, constants,
                weight_index, param_names, const_names,
            )
            if result_hal is not None:
                ops.append(result_hal)

        # ── Handle consumed_internally: insert cache_write/cache_read ──
        if is_kv_func_a and consumed_internally:
            self._add_cache_ops(ops, func_name, layer_idx, consumed_internally, is_kv_func_b)
        if is_kv_func_b:
            self._add_cache_ops(ops, func_name, layer_idx, [], True)

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
                dtype = infer_dtype_from_type(ot)
                shape = shape_from_type(ot)
                out_entry: dict[str, Any] = {
                    "name": ssa.lookup(operand),
                    "shape": shape,
                    "dtype": dtype,
                }
                if i in consumed_internally:
                    out_entry["consumed_internally"] = True
                outputs.append(out_entry)

        # ── Build weight input mapping ───────────────────────────
        # Maps function input SSA names to compiled weight names,
        # using weight_classification params ordered by function
        # signature.  We skip dynamic-shaped inputs (wire/sequence),
        # and rank-1-with-single-element scalars, matching only
        # actual weight/bias tensors to the ordered param list.
        weight_inputs: dict[str, str] = {}
        w_idx = 0
        for i_def in inputs:
            shape = i_def["shape"]
            # Skip dynamic dims (hidden state wires, KV cache).
            is_dyn = any(d == "?" or d == -1 or d == "-1" for d in shape)
            # Skip scalar placeholders: [1], [1, 1], etc.
            is_scalar = bool(shape) and all(
                (isinstance(d, str) and d in ("1", "?"))
                or (isinstance(d, int) and d == 1)
                for d in shape
            )
            if not is_dyn and not is_scalar and w_idx < len(param_names):
                weight_inputs[i_def["name"]] = param_names[w_idx]
                w_idx += 1

        # ── Inject shape_of ops for wire input dynamic dims ─────
        self._inject_shape_of_ops(inputs, weight_inputs, ops)

        # ── Build function entry ────────────────────────────────
        func_entry: dict[str, Any] = {
            "name": func_name,
            "layer": layer_idx,
            "inputs": inputs,
            "outputs": outputs,
            "weights": weights,
            "constants": constants,
            "weight_inputs": weight_inputs,
            "ops": ops,
        }

        if needs_block_table:
            func_entry["block_table"] = True

        return func_entry

    def _inject_shape_of_ops(
        self,
        inputs: list[dict[str, Any]],
        weight_inputs: dict[str, str],
        ops: list[dict[str, Any]],
    ) -> None:
        """Inject shape_of ops for wire input dims and wire them to reshape/fill."""
        dyn_set = {"?", -1, "-1"}
        wire_input = None
        best_score = -999
        for inp in inputs:
            shape = inp["shape"]
            has_dyn = any(d in dyn_set for d in shape)
            is_weight = inp["name"] in weight_inputs
            rank = len(shape)
            if has_dyn and rank >= 2 and not is_weight:
                dyn_count = sum(1 for d in shape if d in dyn_set)
                score = -rank * 10 + dyn_count
                if score > best_score:
                    best_score = score
                    wire_input = inp

        if wire_input is None:
            return

        shape_of_names: list[str] = []
        shape_of_ops: list[dict[str, Any]] = []
        wire_name = wire_input["name"].replace("%", "")
        for dim_idx, dim_val in enumerate(wire_input["shape"]):
            if dim_val in ("?", -1, "-1"):
                out_name = f"%shape_of_{wire_name}_dim{dim_idx}"
                shape_of_ops.append({
                    "op": "shape_of",
                    "inputs": [wire_input["name"]],
                    "outputs": [out_name],
                    "dim": dim_idx,
                })
                shape_of_names.append(out_name)

        if not shape_of_ops:
            return

        for i, sop in enumerate(shape_of_ops):
            ops.insert(i, sop)

        dyn_set = {"?", -1, "-1"}
        for op in ops:
            if op["op"] != "reshape":
                continue
            shape = op.get("shape")
            if shape is None:
                continue
            if len(shape) <= 2:
                continue
            num_dyn = sum(1 for d in shape if d in dyn_set)
            if num_dyn > 0 and len(op["inputs"]) <= 1:
                n = min(num_dyn, len(shape_of_names))
                op["inputs"].extend(shape_of_names[:n])

    def _add_cache_ops(
        self,
        ops: list[dict[str, Any]],
        func_name: str,
        layer_idx: int | None,
        consumed_indices: list[int],
        is_b_func: bool,
    ) -> None:
        """Insert cache_write and cache_read ops for consumed_internally outputs."""
        if layer_idx is None:
            return

        if is_b_func:
            cache_read_entry: dict[str, Any] = {
                "op": "cache_read",
                "layer": layer_idx,
                "inputs": ["%block_table"],
                "outputs": [f"%cache_k_{layer_idx}", f"%cache_v_{layer_idx}"],
            }
            ops.insert(0, cache_read_entry)
        else:
            cache_write_entry: dict[str, Any] = {
                "op": "cache_write",
                "layer": layer_idx,
                "inputs": [],
                "outputs": [],
            }
            ops.append(cache_write_entry)
