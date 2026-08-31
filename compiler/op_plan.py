"""OpPlan generation — sf-dialect op-level HAL kernel graph (Phase 5 M1).

The source of truth is ``model.mlir``'s pre-lowering sf-dialect operation
list (``MlirFunction.ops``).  We never reconstruct the plan from lowered
linalg.  M1 covers the decoder-layer function pairs only; the embedding
prefix, final norm, slice, and lm_head functions remain on the func-level
fast paths.  Every covered function still exists in ``sfa_abi`` — the op
plan is a pure additive symbol.

Canonical op-name mapping
=========================
``sf.layer_norm``                    -> ``layer_norm``
``sf.linear`` (weight [N,K] + bias)  -> ``linear_transb``
``sf.scaled_dot_product_attention``  -> ``attention_causal``
``sf.add`` / ``sf.relu``             -> ``add`` / ``relu``
layout helpers that M1 needs for a faithful graph:
``sf.mul`` / ``sf.view`` / ``sf.transpose`` / ``sf.identity``
"""

from __future__ import annotations

import re
from typing import Any, cast

# Canonical sf.op_name → HAL kernel name.
_CANONICAL: dict[str, str] = {
    "layer_norm": "layer_norm",
    "linear": "linear_transb",
    "scaled_dot_product_attention": "attention_causal",
    "add": "add",
    "relu": "relu",
    "mul": "mul",
    "view": "view",
    "transpose": "transpose",
    "identity": "identity",
}

# Every op name that may appear in a generated plan (catalog assertion).
PLAN_OP_NAMES: set[str] = set(_CANONICAL.values())

_TENSOR_TYPE_RE = re.compile(r"tensor<([^>]*)>")


def parse_tensor_type(type_str: str) -> tuple[list[int], str]:
    """Parse ``tensor<?x12x?x64xf32>`` into ([0,12,0,64], "float32")."""
    m = _TENSOR_TYPE_RE.search(type_str)
    if not m:
        return [], "float32"
    inner = m.group(1).strip()
    if not inner:
        return [], "float32"
    parts = [p.strip() for p in inner.split("x")]
    if len(parts) == 1:
        return [], _canonical_dtype(parts[0])
    dims: list[int] = []
    for part in parts[:-1]:
        if part == "?" or part == "dyn":
            dims.append(0)
        else:
            try:
                dims.append(int(part))
            except ValueError:
                dims.append(0)
    return dims, _canonical_dtype(parts[-1])


def _canonical_dtype(dtype: str) -> str:
    """Normalize MLIR element type strings to the OpTensorSpec vocabulary."""
    return {
        "f32": "float32",
        "float": "float32",
        "f64": "float64",
        "double": "float64",
        "f16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "i64": "int64",
        "i32": "int32",
        "i8": "int8",
        "i1": "bool",
    }.get(dtype, dtype)


def _ssa_key(name: str) -> str:
    return name.lstrip("%")


def _attr_to_string(value: Any) -> str:
    """Serialize one sf-dialect attribute for the plan attribute map."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_attr_to_string(v) for v in value) + "]"
    return str(value)


def _covered_decoder_pairs(functions: list[Any]) -> list[tuple[int, int]]:
    """Return consecutive ``main_Xa`` / ``main_Xb`` decoder-layer pairs.

    The pair is accepted only when the sibling starts with
    ``scaled_dot_product_attention`` and the a-half contains a
    ``layer_norm`` and at least one ``linear`` — this is the structural
    OPT/LLaMA KV split contract.  Everything else stays on the func path.
    """
    pairs: list[tuple[int, int]] = []
    i = 0
    while i + 1 < len(functions):
        a, b = functions[i], functions[i + 1]
        if (
            a.name.endswith("a")
            and b.name == a.name[:-1] + "b"
            and any(op.op_name == "layer_norm" for op in a.ops)
            and any(op.op_name == "linear" for op in a.ops)
            and bool(b.ops)
            and b.ops[0].op_name == "scaled_dot_product_attention"
        ):
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    return pairs


def _build_ssa_to_func_output(functions: list[Any]) -> dict[str, tuple[int, int]]:
    """First producer mapping: SSA name → (func_index, output_index)."""
    out: dict[str, tuple[int, int]] = {}
    for fi, func in enumerate(functions):
        for oi, (name, _tp, _consumed) in enumerate(func.outputs):
            key = _ssa_key(name)
            out.setdefault(key, (fi, oi))
    return out


def _build_type_map(functions: list[Any]) -> dict[str, str]:
    """SSA name → MLIR type string for every op result."""
    types: dict[str, str] = {}
    for func in functions:
        for op in func.ops:
            for idx, result in enumerate(op.results):
                key = _ssa_key(result)
                if idx < len(op.output_types):
                    types[key] = op.output_types[idx]
                else:
                    types.setdefault(key, "tensor<f32>")
    return types


def _make_spec(type_map: dict[str, str], name: str, fallback: str = "tensor<f32>") -> Any:
    dims, dtype = parse_tensor_type(type_map.get(_ssa_key(name), fallback))
    return dims, dtype


def _set_spec(spec: Any, dims: list[int], dtype: str) -> None:
    spec.rank = len(dims)
    spec.dtype = dtype or "float32"
    del spec.dims[:]
    spec.dims.extend(int(d) for d in dims)


def _resolve_input(
    *,
    operand: str,
    func: Any,
    plan_map: dict[str, tuple[int, int]],
    ssa_to_func_output: dict[str, tuple[int, int]],
    type_map: dict[str, str],
    input_type: str | None,
) -> Any:
    """Build one OpPlanInput from an sf op operand."""
    from gen.proto.python import sfa_abi_pb2

    key = _ssa_key(operand)
    inp = sfa_abi_pb2.OpPlanInput()  # type: ignore[attr-defined]
    fallback = input_type or "tensor<f32>"
    dims, dtype = parse_tensor_type(fallback)
    _set_spec(inp.spec, dims, dtype)

    if key in plan_map:
        node_idx, out_idx = plan_map[key]
        inp.source = sfa_abi_pb2.OpPlanInput.SSA  # type: ignore[attr-defined]
        inp.producer.node_index = node_idx
        inp.producer.output_index = out_idx
        return inp

    if key in func.param_weight_names or key in getattr(func, "weights", {}):
        inp.source = sfa_abi_pb2.OpPlanInput.WEIGHT  # type: ignore[attr-defined]
        inp.weight_name = key
        return inp

    if key in func.const_weight_names:
        inp.source = sfa_abi_pb2.OpPlanInput.CONSTANT  # type: ignore[attr-defined]
        inp.constant_name = key
        return inp

    func_ref = ssa_to_func_output.get(key)
    if func_ref is not None:
        inp.source = sfa_abi_pb2.OpPlanInput.FUNC_OUTPUT  # type: ignore[attr-defined]
        inp.func_producer.func_index = func_ref[0]
        inp.func_producer.output_index = func_ref[1]
        return inp

    raise ValueError(
        f"cannot resolve op operand %{key} in {func.name}: not an op result, weight, constant, or function output"
    )


def _build_nodes(
    *,
    functions: list[Any],
    covered: list[int],
    plan_map: dict[str, tuple[int, int]],
    ssa_to_func_output: dict[str, tuple[int, int]],
    type_map: dict[str, str],
    plan: Any,
) -> list[dict[str, Any]]:
    """Emit one node per sf op in every covered function."""
    node_meta: list[dict[str, Any]] = []
    for fi in covered:
        func = functions[fi]
        for op in func.ops:
            if op.op_name not in _CANONICAL:
                raise ValueError(
                    f"uncovered sf op {op.op_name!r} in {func.name} — M1 op plan cannot emit a node for it"
                )
            node = plan.nodes.add()
            node.index = len(plan.nodes) - 1
            node.op_name = _CANONICAL[op.op_name]
            node.source_func_indices.append(fi)
            for attr_key, attr_value in op.attributes.items():
                node.attributes[attr_key] = _attr_to_string(attr_value)
            for idx, operand in enumerate(op.operands):
                input_type = op.input_types[idx] if idx < len(op.input_types) else None
                inp = _resolve_input(
                    operand=operand,
                    func=func,
                    plan_map=plan_map,
                    ssa_to_func_output=ssa_to_func_output,
                    type_map=type_map,
                    input_type=input_type,
                )
                node.inputs.append(inp)
            output = node.outputs.add()
            output.lifetime = 2  # BUFFER_LAYER_PERSISTENT (default)
            out_type = op.output_types[0] if op.output_types else "tensor<f32>"
            dims, dtype = parse_tensor_type(out_type)
            _set_spec(output.spec, dims, dtype)
            for result in op.results:
                key = _ssa_key(result)
                plan_map[key] = (node.index, 0)
                node_meta.append(
                    {
                        "func_index": fi,
                        "ssa": key,
                        "node_index": node.index,
                        "output_index": 0,
                        "op_name": op.op_name,
                    }
                )
    return node_meta


def _external_consumer_ssa(functions: list[Any], covered: set[int]) -> set[str]:
    """SSA names consumed by any non-covered function."""
    out: set[str] = set()
    for fi, func in enumerate(functions):
        if fi in covered:
            continue
        for op in func.ops:
            for operand in op.operands:
                out.add(_ssa_key(operand))
    return out


def _consumer_counts(functions: list[Any], covered: set[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fi in covered:
        for op in functions[fi].ops:
            for operand in op.operands:
                key = _ssa_key(operand)
                counts[key] = counts.get(key, 0) + 1
    return counts


def _finalize_outputs(
    *,
    functions: list[Any],
    covered: list[int],
    plan: Any,
    plan_map: dict[str, tuple[int, int]],
    binding_slab: dict[tuple[int, int], str],
    external_consumers: set[str],
    consumer_counts: dict[str, int],
) -> None:
    """Project func-level outputs, mark cache intercepts and lifetimes."""
    projected_bindings: set[tuple[int, int]] = set()

    for fi in covered:
        func = functions[fi]
        for oi, (name, _tp, consumed) in enumerate(func.outputs):
            key = _ssa_key(name)
            ref = plan_map.get(key)
            if ref is None:
                raise ValueError(f"covered output %{key} of {func.name} has no plan node")

            fo = plan.func_outputs.add()
            fo.func_index = fi
            fo.output_index = oi
            fo.value.node_index = ref[0]
            fo.value.output_index = ref[1]
            fo.consumed_internally = consumed

            node_idx, out_idx = ref
            output = plan.nodes[node_idx].outputs[out_idx]

            if consumed:
                slab_id = binding_slab.get((fi, oi))
                if slab_id is None:
                    raise ValueError(
                        f"consumed output {func.name}[{oi}] has no cache binding "
                        "(metadata.cache_bindings or cache_policy required)"
                    )
                output.lifetime = 3  # BUFFER_CACHE_OWNED
                output.cache.slab_id = slab_id
                output.cache.source_func_index = fi
                output.cache.source_output_index = oi
                projected_bindings.add((fi, oi))
            elif key in external_consumers:
                output.lifetime = 2  # BUFFER_GRAPH_PERSISTENT
            elif consumer_counts.get(key, 0) == 1:
                output.lifetime = 0  # BUFFER_TRANSIENT
            else:
                output.lifetime = 1  # BUFFER_LAYER_PERSISTENT

    # Cross-assert the full cache projection in both directions.
    for (fi, oi), slab_id in binding_slab.items():
        if fi in set(covered) and (fi, oi) not in projected_bindings:
            raise ValueError(f"cache binding ({fi}, {oi}, {slab_id}) not projected onto the op plan")
    for node in plan.nodes:
        for output in node.outputs:
            if output.HasField("cache"):
                cache_key = (
                    output.cache.source_func_index,
                    output.cache.source_output_index,
                )
                if binding_slab.get(cache_key) != output.cache.slab_id:
                    raise ValueError(
                        f"op-plan cache projection {cache_key} slab {output.cache.slab_id!r} "
                        f"does not match cache binding {binding_slab.get(cache_key)!r}"
                    )


def _cache_binding_slab(metadata: dict[str, Any]) -> dict[tuple[int, int], str]:
    """(func_index, output_index) → slab_id from the compile metadata."""
    out: dict[tuple[int, int], str] = {}
    for entry in metadata.get("cache_bindings") or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 3:
            out[(int(entry[0]), int(entry[1]))] = str(entry[2])
    return out


def generate_op_plan(module: Any, metadata: dict[str, Any] | None = None) -> bytes | None:
    """Generate the OpPlan proto for a parsed MlirModule.

    Returns ``None`` when no decoder-layer KV split pair exists (nothing to
    cover — the runtime keeps the func-level path).
    """
    from gen.proto.python import sfa_abi_pb2

    metadata = metadata or {}
    functions = module.functions
    pairs = _covered_decoder_pairs(functions)
    if not pairs:
        return None

    covered: list[int] = []
    for a, b in pairs:
        covered.extend([a, b])
    covered_set = set(covered)

    plan = sfa_abi_pb2.OpPlan()  # type: ignore[attr-defined]
    for name, _tp in functions[0].inputs:
        plan.global_inputs.append(name)
    for op in functions[0].ops:
        if op.op_name == "weight":
            wname = op.attributes.get("name", "")
            if wname and wname not in plan.global_inputs and not wname.startswith("_const_"):
                plan.global_inputs.append(wname)

    plan_map: dict[str, tuple[int, int]] = {}
    ssa_to_func_output = _build_ssa_to_func_output(functions)
    type_map = _build_type_map(functions)
    binding_slab = _cache_binding_slab(metadata)

    _build_nodes(
        functions=functions,
        covered=covered,
        plan_map=plan_map,
        ssa_to_func_output=ssa_to_func_output,
        type_map=type_map,
        plan=plan,
    )

    external_consumers = _external_consumer_ssa(functions, covered_set)
    consumer_counts = _consumer_counts(functions, covered_set)
    _finalize_outputs(
        functions=functions,
        covered=covered,
        plan=plan,
        plan_map=plan_map,
        binding_slab=binding_slab,
        external_consumers=external_consumers,
        consumer_counts=consumer_counts,
    )

    # Global plan output: the last covered function's last visible output.
    last_func = functions[covered[-1]]
    for _oi, (name, _tp, consumed) in reversed(list(enumerate(last_func.outputs))):
        if consumed:
            continue
        ref = plan_map.get(_ssa_key(name))
        if ref is not None:
            plan.global_output.node_index = ref[0]
            plan.global_output.output_index = ref[1]
            break

    validate_op_plan(plan, module, metadata)
    return cast(bytes, plan.SerializeToString())


def validate_op_plan(plan: Any, module: Any, metadata: dict[str, Any] | None = None) -> None:
    """Hard invariant checks for both the compiler seam and load time."""
    metadata = metadata or {}
    functions = module.functions
    num_funcs = len(functions)

    if not plan.nodes:
        raise ValueError("OpPlan has no nodes")
    for idx, node in enumerate(plan.nodes):
        if node.index != idx:
            raise ValueError(f"node index monotonicity broken at {idx} (got {node.index})")
        if node.op_name not in PLAN_OP_NAMES:
            raise ValueError(f"node {idx} op_name {node.op_name!r} not in M1 op catalog")
        if not node.source_func_indices:
            raise ValueError(f"node {idx} missing source_func_indices")
        for sfi in node.source_func_indices:
            if sfi >= num_funcs:
                raise ValueError(f"node {idx} source func {sfi} out of range")
        for inp in node.inputs:
            if inp.source == 4:  # FUNC_OUTPUT
                fi = inp.func_producer.func_index
                oi = inp.func_producer.output_index
                if fi >= num_funcs or oi >= len(functions[fi].outputs):
                    raise ValueError(f"node {idx} FUNC_OUTPUT reference ({fi}, {oi}) out of range")
            elif inp.source == 2:  # SSA
                ni = inp.producer.node_index
                oi = inp.producer.output_index
                if ni >= idx:
                    raise ValueError(f"node {idx} SSA producer {ni} is not strictly earlier")
                if oi >= len(plan.nodes[ni].outputs):
                    raise ValueError(f"node {idx} SSA output index {oi} out of range")

    binding_slab = _cache_binding_slab(metadata)
    projected: set[tuple[int, int]] = set()
    covered_funcs: set[int] = set()
    for node in plan.nodes:
        covered_funcs.update(node.source_func_indices)
        for output in node.outputs:
            if output.HasField("cache"):
                key = (output.cache.source_func_index, output.cache.source_output_index)
                if binding_slab.get(key) != output.cache.slab_id:
                    raise ValueError(f"cache projection {key} does not match cache_bindings")
                projected.add(key)

    for fi, oi in binding_slab:
        if fi in covered_funcs and (fi, oi) not in projected:
            raise ValueError(f"cache binding ({fi}, {oi}) missing from op plan projection")

    all_weights: set[str] = set()
    all_constants: set[str] = set()
    for func in functions:
        all_weights.update(func.param_weight_names)
        all_weights.update(getattr(func, "weights", {}) or {})
        all_constants.update(func.const_weight_names)
    for node in plan.nodes:
        for inp in node.inputs:
            if inp.source == 1 and inp.weight_name not in all_weights:
                raise ValueError(f"node {node.index} references unknown weight {inp.weight_name!r}")
            if inp.source == 3 and inp.constant_name not in all_constants:
                raise ValueError(f"node {node.index} references unknown constant {inp.constant_name!r}")
