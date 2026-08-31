from __future__ import annotations

import logging
import re
import sys
from typing import Any, cast

import torch

from compiler.artifact import MlirFunction, MlirOp  # type: ignore[attr-defined]


def _empty_execution_plan(
    global_inputs: list[tuple[str, str]],
    funcs: list[MlirFunction],
) -> bytes:
    from gen.proto.python import sfa_abi_pb2

    plan = sfa_abi_pb2.ExecutionPlan()  # type: ignore[attr-defined]
    for name, _ in global_inputs:
        plan.global_inputs.append(name)
    step = plan.steps.add()
    step.func_name = funcs[0].name
    for _name, _ in funcs[0].inputs:
        edge = step.inputs.add()
        edge.source = sfa_abi_pb2.GLOBAL_INPUT  # type: ignore[attr-defined]
        edge.source_index = 0
        edge.producer_step = 0

    return cast(bytes, plan.SerializeToString())


_log = logging.getLogger(__name__)


# ── Dimension-position attribute names ─────────────────────
# These encode axis/dim indices that must be adjusted when tensor
# rank changes at a function boundary during per-function splitting.

_DIM_ATTR_NAMES: set[str] = {
    "dim",
    "dim0",
    "dim1",
    "dimensions",
    "axis",
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
                new_attrs[attr_name] = tuple((rank - 1) if isinstance(v, int) and v >= rank else v for v in vals)

    return MlirOp(
        name=op.name,
        dialect=op.dialect,
        op_name=op.op_name,
        operands=list(op.operands),
        results=list(op.results),
        attributes=new_attrs,
        input_types=list(op.input_types),
        output_types=list(op.output_types),
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

    When CachePolicy provides SD-PA intercepts, *boundaries* may also
    include SD-PA op indices (in addition to layer boundaries), causing
    each transformer layer to split into TWO blocks at the SD-PA boundary:
      Block A (QKV proj):  ops before SD-PA
      Block B (Attn+FFN):  SD-PA and all following ops in that layer

    Returns (augmented_ops, num_functions).
    """
    if boundaries:
        # Insert sentinel before each boundary op.
        # Boundaries may be a mix of layer-start and SD-PA op indices.
        result: list[MlirOp] = []
        boundary_set = set(boundaries)
        for i, op in enumerate(mlir_ops):
            if i in boundary_set:
                result.append(
                    MlirOp(
                        name="_sentinel",
                        dialect="_sentinel",
                        op_name="_func_boundary",
                        operands=[],
                        results=[],
                        attributes={},
                        input_types=[],
                        output_types=[],
                    )
                )
            result.append(op)
        return result, len(boundaries) + 1

    # Op-count splitting (existing behaviour)
    if len(mlir_ops) <= ops_per_func:
        return mlir_ops, 1

    result_ops: list[MlirOp] = []
    for i, op in enumerate(mlir_ops):
        if i > 0 and i % ops_per_func == 0:
            result_ops.append(
                MlirOp(
                    name="_sentinel",
                    dialect="_sentinel",
                    op_name="_func_boundary",
                    operands=[],
                    results=[],
                    attributes={},
                    input_types=[],
                    output_types=[],
                )
            )
        result_ops.append(op)
    return result_ops, (len(mlir_ops) + ops_per_func - 1) // ops_per_func


# ── Layer detection helpers ─────────────────────────────


def _detect_layer(nn_module_stack: dict[str, Any], prev_layer: str) -> str:
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
        m = re.search(r"layers\.(\d+)", str(key))
        if m:
            num = int(m.group(1))
            if layer_num is None or num > layer_num:
                layer_num = num

    if layer_num is not None:
        return f"layer_{layer_num}"

    # Check for output-module indicators
    key_str_lower = " ".join(str(k).lower() for k in nn_module_stack)
    if any(kw in key_str_lower for kw in ["lm_head", "final_norm", "final_layer_norm"]):
        return "output"

    # If we've seen layers before and now no layer key → output region
    if prev_layer and prev_layer.startswith("layer_"):
        return "output"

    return "embed_prefix"


def _log_split_plan(mlir_ops: list[Any], boundaries: list[int]) -> None:
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

_GQA_REPEAT_OPS = ("unsqueeze", "expand", "view")
_MAX_GQA_TRACE = 8


def _ssa_key(name: str) -> str:
    """Normalize an SSA reference for producer-map lookups."""
    return name.lstrip("%")


def _producer_indices(mlir_ops: list[MlirOp]) -> dict[str, int]:
    """Map result SSA name -> op index; last producer wins (MLIR SSA is unique)."""
    producers: dict[str, int] = {}
    for idx, op in enumerate(mlir_ops):
        for result in op.results:
            producers[_ssa_key(result)] = idx
    return producers


def _gqa_repeat_chain_start(
    mlir_ops: list[MlirOp],
    sdpa_idx: int,
    operand_idx: int,
    producer_by_result: dict[str, int],
) -> int | None:
    """Return the op index of the first repeat-kv op feeding an SDPA operand.

    GQA models lower repeat-kv as ``unsqueeze -> expand -> view``.  The
    cache boundary belongs before the first op in that chain so K/V are
    cached at their native kv-head count.  Returns ``None`` when the operand
    does not have a recognised repeat-kv chain.
    """
    sdpa_op = mlir_ops[sdpa_idx]
    if operand_idx >= len(sdpa_op.operands):
        return None
    cur = _ssa_key(sdpa_op.operands[operand_idx])
    path: list[int] = []
    for _ in range(_MAX_GQA_TRACE):
        producer_idx = producer_by_result.get(cur)
        if producer_idx is None or producer_idx >= sdpa_idx:
            break
        op = mlir_ops[producer_idx]
        if op.op_name not in _GQA_REPEAT_OPS or not op.operands:
            break
        path.append(producer_idx)
        cur = _ssa_key(op.operands[0])
        if op.op_name == "unsqueeze":
            break
    if [mlir_ops[i].op_name for i in path] == ["view", "expand", "unsqueeze"]:
        return path[-1]
    return None


def sdpa_cache_boundary_indices(mlir_ops: list[MlirOp]) -> list[int]:
    """Compute cache split boundaries for all SDPA ops.

    Dense attention keeps the legacy boundary at the SDPA op itself.
    GQA attention moves the boundary before the ``unsqueeze -> expand ->
    view`` repeat-kv chain so the a-block caches native-head K/V.
    """
    producer_by_result = _producer_indices(mlir_ops)
    boundaries: list[int] = []
    for sdpa_idx, op in enumerate(mlir_ops):
        if op.op_name != "scaled_dot_product_attention":
            continue
        k_start = _gqa_repeat_chain_start(mlir_ops, sdpa_idx, 1, producer_by_result)
        v_start = _gqa_repeat_chain_start(mlir_ops, sdpa_idx, 2, producer_by_result)
        starts = [idx for idx in (k_start, v_start) if idx is not None]
        boundaries.append(min(starts) if len(starts) == 2 else sdpa_idx)
    return boundaries


def _trace_sdpa_cache_source(
    block: list[MlirOp],
    block_producer: dict[str, int],
    operand: str,
) -> str:
    """Trace an SDPA operand back to the a-block value that should be cached.

    In GQA graphs the SDPA K/V operands are produced in the same b-block by
    ``view(expand(unsqueeze(x)))``.  The cache contract needs ``x`` (the
    native kv-head tensor produced by the preceding a-block), not the
    expanded operand.
    """
    cur = _ssa_key(operand)
    for _ in range(_MAX_GQA_TRACE):
        producer_idx = block_producer.get(cur)
        if producer_idx is None:
            return cur
        op = block[producer_idx]
        if op.op_name not in _GQA_REPEAT_OPS or not op.operands:
            return cur
        cur = _ssa_key(op.operands[0])
    return cur


def _make_multi_functions(
    mlir_ops: list[MlirOp],
    global_inputs: list[tuple[str, str]],
    global_outputs: list[tuple[str, str, bool]],
    weights: dict[str, torch.Tensor],
    param_names: set[str],
    const_names: set[str],
    base_name: str,
    cache_policy: Any = None,
) -> tuple[list[MlirFunction], list[str], list[int], bytes, list[tuple[int, int, str]]]:
    """Split mlir_ops at sentinel boundaries into separate MlirFunction objects.

    If a block's first op is ``scaled_dot_product_attention`` (SD-PA), it is
    a "b" block (Attention+FFN half of a split layer).  The preceding block
    is its "a" block (QKV projection half).  The two share a layer number:

      main_N_a  —  QKV projection (K/V outputs get consumed_internally=True)
      main_N_b  —  Attention+FFN (K/V inputs come from executor, not SSA)

    All other blocks (embedding, unsplit layers, output) keep the plain
    ``main_N`` naming.

    Returns a 5-tuple ``(functions, chain_order, exec_plan_data,
    plan_bytes, cache_bindings)``.  ``cache_bindings`` is the KV cache
    contract table: one ``(func_index, output_index, slab_id)`` entry per
    cache-consumed K/V output of an a-block.  ``slab_id`` is resolved from
    *cache_policy*'s ``scaled_dot_product_attention`` intercepts by source
    operand (``operand[1]`` = K, ``operand[2]`` = V); the table is empty
    when *cache_policy* is ``None`` or has no SDPA intercepts.
    """

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
        # Single function: collect weight names in ops order
        single_weight_names = []
        for op in mlir_ops:
            if op.op_name == "weight":
                wname = op.attributes.get("name", "")
                if wname:
                    single_weight_names.append(wname)
        funcs = [
            MlirFunction(
                name=base_name,
                inputs=global_inputs,
                outputs=global_outputs,
                ops=mlir_ops,
                weights=weights,
                param_weight_names=param_names,
                const_weight_names=const_names,
                weight_names=single_weight_names,
            )
        ]
        empty_plan = _empty_execution_plan(global_inputs, funcs)
        return funcs, [base_name], [], empty_plan, []

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

    # ── Detect SD-PA split pairs and assign function names ──
    # A "b" block is one whose first non-preparation op is
    # scaled_dot_product_attention.  In GQA graphs the boundary is placed
    # before the repeat-kv chain, so the b-block may start with
    # unsqueeze/expand/view before SDPA.  The preceding block is its "a"
    # block (projections + native-head K/V); both share one layer number.
    _is_b_block: list[bool] = [False] * len(blocks)
    sdpa_positions: list[int | None] = [None] * len(blocks)
    for fi, block in enumerate(blocks):
        pos = next(
            (i for i, op in enumerate(block) if op.op_name == "scaled_dot_product_attention"),
            None,
        )
        sdpa_positions[fi] = pos
        if fi > 0 and pos is not None and all(
            op.op_name in _GQA_REPEAT_OPS for op in block[:pos]
        ):
            _is_b_block[fi] = True

    # Collect SSA names of Q/K/V outputs from each "b" block's SD-PA op.
    # operands[0]=Q, operands[1]=K, operands[2]=V.  K/V are traced through
    # the repeat-kv chain back to the native-head a-block values that the
    # cache should actually store.
    sdpa_kv_ssa: dict[int, tuple[str, str, str]] = {}  # block index → (Q_ssa, K_ssa, V_ssa)
    for fi, is_b in enumerate(_is_b_block):
        if not is_b:
            continue
        sdpa_pos = sdpa_positions[fi]
        assert sdpa_pos is not None
        sdpa_op = blocks[fi][sdpa_pos]
        q_ssa = _ssa_key(sdpa_op.operands[0]) if len(sdpa_op.operands) > 0 else ""
        block_producer: dict[str, int] = {}
        for oi, op in enumerate(blocks[fi]):
            for result in op.results:
                block_producer[_ssa_key(result)] = oi
        k_ssa = (
            _trace_sdpa_cache_source(blocks[fi], block_producer, sdpa_op.operands[1])
            if len(sdpa_op.operands) > 1
            else ""
        )
        v_ssa = (
            _trace_sdpa_cache_source(blocks[fi], block_producer, sdpa_op.operands[2])
            if len(sdpa_op.operands) > 2
            else ""
        )
        sdpa_kv_ssa[fi] = (q_ssa, k_ssa, v_ssa)

    # ── KV cache binding table (contract) ──
    # Map CachePolicy intercept source operands to slabs:
    #   "operand[1]" (K) → slab_id, "operand[2]" (V) → slab_id.
    # Only scaled_dot_product_attention intercepts participate.
    operand_slab: dict[int, str] = {}
    if cache_policy is not None:
        for intercept in getattr(cache_policy, "intercepts", []):
            if getattr(intercept, "op_name", None) != "scaled_dot_product_attention":
                continue
            src = getattr(intercept, "source", "")
            m = re.fullmatch(r"operand\[(\d+)\]", src)
            if m:
                operand_slab[int(m.group(1))] = intercept.slab_id
    bindings: list[tuple[int, int, str]] = []

    # Assign layer numbers and suffixes
    func_names: list[str] = [""] * len(blocks)
    layer_n: int = 0
    for fi in range(len(blocks)):
        if _is_b_block[fi]:
            # "b" block — share layer number with the "a" block preceding it
            func_names[fi - 1] = f"{base_name}_{layer_n}a"
            func_names[fi] = f"{base_name}_{layer_n}b"
            layer_n += 1
        elif fi + 1 < len(blocks) and _is_b_block[fi + 1]:
            pass  # "a" block — named above when its "b" sibling is processed
        else:
            # Standalone block (embed, unsplit layer, or output)
            func_names[fi] = f"{base_name}_{layer_n}"
            layer_n += 1

    # ── Build functions ──
    built_funcs: list[MlirFunction] = []

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
        ordered_weights: list[str] = []
        _ssa_to_wname: dict[str, str] = {}
        if fi == 0:
            f_inputs = list(global_inputs)
            for op in block:
                if op.op_name == "weight" and op.results:
                    wname = op.attributes.get("name", "")
                    if wname:
                        _ssa_to_wname[op.results[0].lstrip("%")] = wname
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
            # Order inputs to match func[1]'s proven-correct convention:
            # [scalar, hidden_state, mask, weights..., sym_size...]
            nw_0 = [
                (v, prod_order.get(v, (999, 999)))
                for v in needed
                if prod_order.get(v, (999,))[0] == 0 and v not in weights
            ]
            xfunc = [
                (v, prod_order.get(v, (999, 999)))
                for v in needed
                if prod_order.get(v, (999,))[0] != 0 and v not in weights
            ]
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
            # Then weights — sorted by SSA name to ensure deterministic ordering.
            # Record the ordered weight names for this function so downstream
            # tools and tests can construct correct input tensors.
            ordered_weights = [val for val, _ in sorted(w_0, key=lambda x: x[0])]
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

        # Determine consumed_internally for each output.
        # If this is an "a" block (has a following "b" sibling), the K and V
        # SSAs consumed by that sibling's SD-PA op are consumed_internally=True.
        is_a_block = fi + 1 < len(blocks) and _is_b_block[fi + 1]
        kv_to_mark: set[str] = set()
        if is_a_block:
            _q_ssa, k_ssa, v_ssa = sdpa_kv_ssa[fi + 1]
            if k_ssa:
                kv_to_mark.add(k_ssa)
            if v_ssa:
                kv_to_mark.add(v_ssa)

        # For main_0, weight_names ordering: consumed weights first
        # (emission order), then exported weights in return order
        # (alphabetical by SSA name, matching f_outputs).
        if fi == 0:
            exported_weight_ssa = [v for v in sorted(exported) if v in _ssa_to_wname]
            consumed_weight_ssa = [k for k in _ssa_to_wname if k not in set(exported_weight_ssa)]
            ordered_weights = [_ssa_to_wname[v] for v in consumed_weight_ssa] + [
                _ssa_to_wname[v] for v in exported_weight_ssa
            ]

        f_outputs: list[tuple[str, str, bool]] = []
        slab_for_ssa: dict[str, str] = {}
        if is_a_block:
            # Contract: a-block outputs are ordered [Q, K, V] so downstream
            # ABI consumers can rely on consumed flags [False, True, True]
            # (first consumed sub-output = K, second = V).  Never sort by
            # SSA name — lexicographic order is layer-dependent and breaks
            # the contract (e.g. %transpose_8/9/10 sorts as [10, 8, 9]).
            q_ssa, k_ssa, v_ssa = sdpa_kv_ssa[fi + 1]
            if operand_slab:
                if k_ssa and 1 in operand_slab:
                    slab_for_ssa[k_ssa] = operand_slab[1]
                if v_ssa and 2 in operand_slab:
                    slab_for_ssa[v_ssa] = operand_slab[2]
            ordered = [s for s in (q_ssa, k_ssa, v_ssa) if s and s in exported]
            ordered += [v for v in sorted(exported) if v not in ordered]
        else:
            ordered = sorted(exported)
        for v in ordered:
            consumed = v in kv_to_mark
            f_outputs.append((f"%{v}", type_map.get(v, "tensor<f32>"), consumed))
            slab_id = slab_for_ssa.get(v) if consumed else None
            if slab_id is not None:
                bindings.append((fi, len(f_outputs) - 1, slab_id))

        if not f_outputs:
            # A split block with no downstream consumer still needs one
            # output descriptor for the ABI, but it is NOT cache-consumed.
            # The old `True` fallback collided with cache-policy contract
            # checks once a model had any bound intercepts.
            p = list(produced_here)[0]
            f_outputs = [(f"%{p}", type_map.get(p, "tensor<f32>"), False)]

        # Compute rank of each function input from its type string
        input_rank_map: dict[str, int] = {}
        for name, tp in f_inputs:
            rank = 1 if "x" not in tp and "tensor<" in tp else tp.count("x")
            input_rank_map[name.lstrip("%")] = max(rank, 1) if "tensor<" in tp else 0

        adjusted_block = [_adjust_op_attributes(op, input_rank_map) for op in block]

        built_funcs.append(
            MlirFunction(
                name=func_names[fi],
                inputs=f_inputs,
                outputs=f_outputs,
                ops=adjusted_block,
                weights={k: v for k, v in weights.items() if k in weight_refs_per_func[fi]},
                param_weight_names=param_names & weight_refs_per_func[fi],
                const_weight_names=const_names & weight_refs_per_func[fi],
                weight_names=ordered_weights,
            )
        )

    chain_order = [f.name for f in built_funcs]

    # Build global input names (match proto ExecutionPlan.global_inputs order)
    global_names = [name for name, _ in global_inputs]
    for op in built_funcs[0].ops:
        if op.op_name == "weight":
            wname = op.attributes.get("name", "")
            if wname and wname not in global_names and not wname.startswith("_const_"):
                global_names.append(wname)

    # Build per-step input bindings as structured data (source, source_index, producer_step)
    step_inputs: list[list[tuple[int, int, int]]] = []
    for fi, func in enumerate(built_funcs):
        sis: list[tuple[int, int, int]] = []
        for name, _ in func.inputs:
            key = name.lstrip("%")
            if fi == 0:
                try:
                    gi = global_names.index(key)
                except ValueError:
                    gi = global_names.index(name)
                sis.append((0, gi, 0))
            else:
                if key in weights:
                    out_idx = -1
                    for oi, (oname, _, _) in enumerate(built_funcs[0].outputs):
                        if oname.lstrip("%") == key:
                            out_idx = oi
                            break
                    if out_idx >= 0:
                        sis.append((1, 0, out_idx))
                    else:
                        try:
                            gi = global_names.index(key)
                        except ValueError:
                            gi = global_names.index(name)
                        sis.append((0, gi, 0))
                else:
                    prod = producer.get(key)
                    if prod is not None and prod > 0:
                        producer_func = built_funcs[prod]
                        out_idx = 0
                        for oi, (oname, _, _) in enumerate(producer_func.outputs):
                            if oname.lstrip("%") == key:
                                out_idx = oi
                                break
                        sis.append((1, prod, out_idx))
                    else:
                        out_idx = 0
                        for oi, (oname, _, _) in enumerate(built_funcs[0].outputs):
                            if oname.lstrip("%") == key:
                                out_idx = oi
                                break
                        sis.append((1, 0, out_idx))
        step_inputs.append(sis)

    # ── Generate ExecutionPlan proto (single source of truth) ──
    from gen.proto.python import sfa_abi_pb2

    plan = sfa_abi_pb2.ExecutionPlan()  # type: ignore[attr-defined]
    for gn in global_names:
        plan.global_inputs.append(gn)

    for _fi, (func, sis) in enumerate(zip(built_funcs, step_inputs, strict=False)):
        step = plan.steps.add()
        step.func_name = func.name
        for src, source_index, producer_step in sis:
            edge = step.inputs.add()
            if src == 0:
                edge.source = sfa_abi_pb2.GLOBAL_INPUT  # type: ignore[attr-defined]
            else:
                edge.source = sfa_abi_pb2.STEP_OUTPUT  # type: ignore[attr-defined]
            edge.source_index = source_index
            edge.producer_step = producer_step

    plan_bytes = plan.SerializeToString()

    # Derive flat int array FROM proto (not from step_inputs) —
    # ensures protocol is the single source of truth for the wire format.
    exec_plan_data = _flatten_execution_plan(plan)

    return built_funcs, chain_order, exec_plan_data, plan_bytes, bindings


def _flatten_execution_plan(plan: Any) -> list[int]:
    """Serialize ExecutionPlan proto into the flat int array consumed by
    SfChainWrapper.cpp and stored as sf.exec_plan_data MLIR module attribute.

    Wire format (matching ``EdgeInput`` proto fields):
      [num_steps, num_global_inputs,
       step0_num_inputs, src, source_index, producer_step, ...,
       step1_num_inputs, src, source_index, producer_step, ...]

    Each edge is 3 ints: (EdgeSource, source_index, producer_step).
    This encoding is verified against the proto at generation time.
    """
    result: list[int] = [len(plan.steps), len(plan.global_inputs)]
    for step in plan.steps:
        result.append(len(step.inputs))
        for edge in step.inputs:
            result.append(edge.source)
            result.append(edge.source_index)
            result.append(edge.producer_step)
    return result
