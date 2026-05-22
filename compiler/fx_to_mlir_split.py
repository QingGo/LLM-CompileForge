from __future__ import annotations

import logging
import re
import sys

import torch

from compiler.mlir_artifact import MlirFunction, MlirOp

_log = logging.getLogger(__name__)


# ── Dimension-position attribute names ─────────────────────
# These encode axis/dim indices that must be adjusted when tensor
# rank changes at a function boundary during per-function splitting.

_DIM_ATTR_NAMES: set[str] = {
    "dim", "dim0", "dim1", "dimensions", "axis",
}

_DIM_LIST_ATTR_NAMES: set[str] = {
    "dims",
}


def _adjust_op_attributes(op: MlirOp, input_rank_map: dict[str, int]) -> MlirOp:
    """Clamp dim/axis attributes that are out-of-bounds for the op's
    declared input ranks at a function boundary.

    When the full CFG is split into per-function chunks, some ops
    retain ``dim``-like attributes from the original higher-rank
    context.  This clamps such values into the valid ``[0, rank-1]``
    range so the op can execute in isolation.
    """
    if op.op_name in ("weight", "constant", "_func_boundary"):
        return op

    new_attrs = dict(op.attributes)

    for _i, operand in enumerate(op.operands):
        key = operand.lstrip("%")
        rank = input_rank_map.get(key)
        if rank is None or rank <= 0:
            continue

        for attr_name in _DIM_ATTR_NAMES & set(new_attrs.keys()):
            val = new_attrs[attr_name]
            if isinstance(val, int) and val >= rank:
                new_attrs[attr_name] = rank - 1

        for attr_name in _DIM_LIST_ATTR_NAMES & set(new_attrs.keys()):
            vals = new_attrs[attr_name]
            if isinstance(vals, (list, tuple)):
                new_attrs[attr_name] = tuple(
                    (rank - 1) if isinstance(v, int) and v >= rank else v
                    for v in vals
                )

    return MlirOp(
        name=op.name, dialect=op.dialect, op_name=op.op_name,
        operands=list(op.operands), results=list(op.results),
        attributes=new_attrs,
        input_types=list(op.input_types), output_types=list(op.output_types),
    )


# ── Function splitting ──────────────────────────────────


def _split_into_functions(
    mlir_ops: list[MlirOp],
    func_inputs: list[tuple[str, str]],
    weights: dict[str, torch.Tensor],
    ops_per_func: int = 500,
    boundaries: list[int] | None = None,
) -> tuple[list[MlirOp], int]:
    """Insert sentinel ops marking function boundaries.

    If *boundaries* is provided (non-empty), insert sentinels at those
    indices (layer-aware splitting). Otherwise, insert a sentinel every
    *ops_per_func* ops (op-count splitting, existing behaviour).

    Returns (augmented_ops, num_functions).
    """
    if boundaries:
        # Layer-based: insert sentinel before each boundary op
        result: list[MlirOp] = []
        boundary_set = set(boundaries)
        for i, op in enumerate(mlir_ops):
            if i in boundary_set:
                result.append(MlirOp(
                    name="_sentinel", dialect="_sentinel", op_name="_func_boundary",
                    operands=[], results=[], attributes={},
                    input_types=[], output_types=[],
                ))
            result.append(op)
        return result, len(boundaries) + 1

    # Op-count splitting (existing behaviour)
    if len(mlir_ops) <= ops_per_func:
        return mlir_ops, 1

    result: list[MlirOp] = []
    for i, op in enumerate(mlir_ops):
        if i > 0 and i % ops_per_func == 0:
            result.append(MlirOp(
                name="_sentinel", dialect="_sentinel", op_name="_func_boundary",
                operands=[], results=[], attributes={},
                input_types=[], output_types=[],
            ))
        result.append(op)
    return result, (len(mlir_ops) + ops_per_func - 1) // ops_per_func


# ── Layer detection helpers ─────────────────────────────


def _detect_layer(nn_module_stack: dict, prev_layer: str) -> str:
    """Extract layer identifier from ``nn_module_stack`` dict.

    Returns one of:
      - ``"embed_prefix"`` — decoder/embedding prefix (before any layer)
      - ``"layer_{N}"``    — transformer layer ``N``
      - ``"output"``       — output region (final norm + lm_head)
    """
    if not nn_module_stack:
        return prev_layer

    # Look for decoder layer numbers in stack keys
    layer_num: int | None = None
    for key in nn_module_stack:
        m = re.search(r'layers\.(\d+)', str(key))
        if m:
            num = int(m.group(1))
            if layer_num is None or num > layer_num:
                layer_num = num

    if layer_num is not None:
        return f"layer_{layer_num}"

    # Check for output-module indicators
    key_str_lower = " ".join(str(k).lower() for k in nn_module_stack)
    if any(kw in key_str_lower for kw in ['lm_head', 'final_norm', 'final_layer_norm']):
        return "output"

    # If we've seen layers before and now no layer key → output region
    if prev_layer and prev_layer.startswith("layer_"):
        return "output"

    return "embed_prefix"


def _log_split_plan(mlir_ops: list, boundaries: list[int]) -> None:
    """Log layer-based split plan to stderr."""
    segments: list[tuple[str, int]] = []
    prev = 0
    for b in sorted(boundaries):
        if b <= prev:
            continue
        layer_name = mlir_ops[prev].attributes.get("dump_layer", "?")
        segments.append((layer_name, b - prev))
        prev = b
    if prev < len(mlir_ops):
        layer_name = mlir_ops[prev].attributes.get("dump_layer", "?")
        segments.append((layer_name, len(mlir_ops) - prev))

    print(
        f"[fx_to_mlir] Layer-based split: {len(segments)} functions detected",
        file=sys.stderr,
    )
    for i, (name, count) in enumerate(segments):
        print(f"[fx_to_mlir]   func_{i}: {name} — {count} ops", file=sys.stderr)


def _make_multi_functions(
    mlir_ops: list[MlirOp],
    global_inputs: list[tuple[str, str]],
    global_outputs: list[tuple[str, str, bool]],
    weights: dict[str, torch.Tensor],
    param_names: set[str],
    const_names: set[str],
    base_name: str,
) -> list[MlirFunction]:
    """Split mlir_ops at sentinel boundaries into separate MlirFunction objects."""

    # Remove sentinel ops and partition into blocks
    blocks: list[list[MlirOp]] = []
    current: list[MlirOp] = []
    for op in mlir_ops:
        if op.op_name == "_func_boundary":
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(op)
    if current:
        blocks.append(current)

    if len(blocks) <= 1:
        return [MlirFunction(
            name=base_name, inputs=global_inputs, outputs=global_outputs,
            ops=mlir_ops, weights=weights,
            param_weight_names=param_names, const_weight_names=const_names,
        )]

    # Build type map: SSA name → MLIR type string (normalize % prefix)
    type_map: dict[str, str] = {}
    for op in mlir_ops:
        for idx, r in enumerate(op.results):
            key = r.lstrip("%")
            if idx < len(op.output_types):
                type_map[key] = op.output_types[idx]
            else:
                type_map[key] = "tensor<f32>"

    # Track which function produces each result (normalize % prefix)
    producer: dict[str, int] = {}
    for fi, block in enumerate(blocks):
        for op in block:
            for r in op.results:
                producer[r.lstrip("%")] = fi

    # Track weight references per function (normalize % prefix)
    weight_refs_per_func: list[set[str]] = [set() for _ in blocks]
    for fi, block in enumerate(blocks):
        for op in block:
            for operand in op.operands:
                key = operand.lstrip("%")
                if key in weights:
                    weight_refs_per_func[fi].add(key)

    funcs: list[MlirFunction] = []

    for fi, block in enumerate(blocks):
        # External inputs needed by this block (normalize %)
        needed: set[str] = set()
        for op in block:
            for operand in op.operands:
                key = operand.lstrip("%")
                if key in producer and producer[key] != fi:
                    needed.add(key)

        # Function inputs with proper types
        f_inputs: list[tuple[str, str]]
        if fi == 0:
            f_inputs = list(global_inputs)
        else:
            f_inputs = []
            # Build producer order map: SSA name → (func_idx, op_sequence_number)
            # op_sequence_number is the index of the producing op within its function
            # block, used to preserve the original computation order across functions.
            prod_order: dict[str, tuple[int, int]] = {}
            for fii, block_i in enumerate(blocks):
                for oi, op in enumerate(block_i):
                    for r in op.results:
                        prod_order[r.lstrip("%")] = (fii, oi)

            # Sort inputs to match ciface calling convention:
            #   Non-weight from func[0] first (sorted by func[0] output index),
            #   then cross-function values (from func[fi-1] etc.),
            #   then weights from func[0] last.
            # Order inputs to match ciface calling convention:
            # 1. Non-weight values from func[0] (scalar, hidden_state, mask)
            # 2. Cross-function values (hidden state from func[fi-1])
            # 3. Weight values from func[0]
            # 4. sym_size scalars from func[0]
            # Order inputs to match func[1]'s proven-correct convention:
            # [scalar, hidden_state, mask, weights..., sym_size...]
            nw_0 = [(v, prod_order.get(v, (999, 999))) for v in needed
                    if prod_order.get(v, (999,))[0] == 0 and v not in weights]
            xfunc = [(v, prod_order.get(v, (999, 999))) for v in needed
                     if prod_order.get(v, (999,))[0] != 0 and v not in weights]
            w_0 = [(v, (999, 999)) for v in needed if v in weights]
            # First entry: earliest non-weight from func[0] (the scalar)
            nw_sorted = sorted(nw_0, key=lambda x: x[1][1])
            if nw_sorted:
                f_inputs.append((f"%{nw_sorted[0][0]}", type_map.get(nw_sorted[0][0], "tensor<f32>")))
                nw_sorted = nw_sorted[1:]
            # Then cross-function values (hidden state from func[fi-1])
            for val, _ in sorted(xfunc, key=lambda x: x[1]):
                f_inputs.append((f"%{val}", type_map.get(val, "tensor<f32>")))
            # Then remaining non-weight from func[0] (mask, etc.)
            for val, _ in nw_sorted:
                if not type_map.get(val, "").startswith("tensor<1x"):
                    f_inputs.append((f"%{val}", type_map.get(val, "tensor<f32>")))
            # Then weights
            for val, _ in sorted(w_0, key=lambda x: x[0]):
                f_inputs.append((f"%{val}", type_map.get(val, "tensor<f32>")))
            # Finally sym_size scalars
            for val, _ in nw_sorted:
                if type_map.get(val, "").startswith("tensor<1x"):
                    f_inputs.append((f"%{val}", type_map.get(val, "tensor<f32>")))

        # Values produced here that are consumed elsewhere (normalize %)
        produced_here: set[str] = set()
        for op in block:
            for r in op.results:
                produced_here.add(r.lstrip("%"))

        exported: set[str] = set()
        for val in produced_here:
            for fi2 in range(fi + 1, len(blocks)):
                for op2 in blocks[fi2]:
                    for operand in op2.operands:
                        if operand.lstrip("%") == val:
                            exported.add(val)
                            break

        # Include global outputs
        for name, _, _ in global_outputs:
            if name.lstrip("%") in produced_here:
                exported.add(name.lstrip("%"))

        f_outputs = [(f"%{v}", type_map.get(v, "tensor<f32>"), False) for v in sorted(exported)]
        if not f_outputs:
            p = list(produced_here)[0]
            f_outputs = [(f"%{p}", type_map.get(p, "tensor<f32>"), True)]

        # Compute rank of each function input from its type string
        input_rank_map: dict[str, int] = {}
        for name, tp in f_inputs:
            rank = 1 if "x" not in tp and "tensor<" in tp else tp.count("x")
            input_rank_map[name.lstrip("%")] = max(rank, 1) if "tensor<" in tp else 0

        adjusted_block = [_adjust_op_attributes(op, input_rank_map) for op in block]

        funcs.append(MlirFunction(
            name=f"{base_name}_{fi}",
            inputs=f_inputs,
            outputs=f_outputs,
            ops=adjusted_block,
            weights={k: v for k, v in weights.items() if k in weight_refs_per_func[fi]},
            param_weight_names=param_names & weight_refs_per_func[fi],
            const_weight_names=const_names & weight_refs_per_func[fi],
        ))

    return funcs
