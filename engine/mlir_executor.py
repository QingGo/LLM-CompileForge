"""MLIR-based model executor — bridges compiled MLIR to HAL.

Walks the parsed MlirModule (from model.mlir) and dispatches each
operation to the HAL OpExecutor.  This is the MLIR-native counterpart
to the legacy Executor (which walks Python IrModule).

Key features:
  - Weight tensor lookup from MlirFunction.weights
  - Same HAL dispatch as the Python IR executor
  - CacheManager-based KV cache (policy-driven, no heuristic guessing)
  - Backward compatible with old compiled models (fallback to legacy path)
  - Full forward() and forward_with_kv() API compatibility
"""

from __future__ import annotations

import os
from typing import Any

import torch

from compiler.mlir_artifact import MlirFunction, MlirModule, MlirOp
from engine._kv_cache import _KVCacheMixin, _normalize_kv_for_cache
from hal.interface import OpExecutor

_WEIGHT_OPS = frozenset({"sf.weight", "sf.constant", "constant"})


class MlirExecutor(_KVCacheMixin):
    """Executes a compiled model from MLIR artifact through the HAL.

    Cache Manager (preferred, policy-driven):
        When the compiled artifact carries a CachePolicy in its metadata,
        the executor delegates KV cache I/O to CacheManager.  No heuristic
        op name matching — the policy declares exactly which ops to intercept.

    Legacy path (backward compatible):
        Old compiled models without cache_policy fall back to the original
        "sdpa" substring match + _KVCacheMixin path.

    Usage:
        module = load_mlir_artifact("./compiled/model")
        backend = PyTorchBackend("cpu")
        executor = MlirExecutor(module, backend)
        logits = executor.forward(input_ids)
    """

    def __init__(
        self,
        module: MlirModule,
        hal_backend: OpExecutor,
        dump_dir: str | None = None,
    ) -> None:
        self._module = module
        self._hal = hal_backend
        self._function = module.main

        # Load weights from ALL functions (not just main).
        self._weights: dict[str, torch.Tensor] = {}
        for func in module.functions:
            for name, tensor in func.weights.items():
                if name not in self._weights:
                    self._weights[name] = tensor

        # Cast all floating-point weights to float32 (dylib uses f32, but
        # checkpoint may store f16 — mismatch causes ~6% error per dot product).
        for name, tensor in self._weights.items():
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                self._weights[name] = tensor.float()

        # Build compiled-name → tensor mapping via hf_key_map.
        hfk = module.metadata.get("hf_key_map", {})
        for compiled_name, hf_key in hfk.items():
            if compiled_name not in self._weights and hf_key in self._weights:
                self._weights[compiled_name] = self._weights[hf_key]

        # Constants are stored with function prefix, ops reference by bare name.
        for func in module.functions:
            prefix = func.name + "."
            for key in list(self._weights.keys()):
                if key.startswith(prefix):
                    bare = key[len(prefix):]
                    if bare not in self._weights:
                        self._weights[bare] = self._weights[key]

        # Handle tied weights (may be at metadata.tied_weights or metadata.weight_source.tied_weights).
        tied: dict[str, str] = {}
        if module.metadata:
            tied = module.metadata.get("tied_weights", {})
            if not tied:
                ws = module.metadata.get("weight_source", {})
                tied = ws.get("tied_weights", {})
        for alias, primary in tied.items():
            if alias not in self._weights and primary in self._weights:
                self._weights[alias] = self._weights[primary]

        # ── Cache manager (new path) ──────────────────────
        self._cache_mgr: Any = None
        self._intercepts_by_op: dict[str, list[Any]] = {}
        self._intercept_slab_ops: set[str] = set()
        self._uses_cache_manager = False
        self._uses_static_shape = self._detect_static_shape()

        raw_policy = module.metadata.get("cache_policy")
        if raw_policy and not self._uses_static_shape:
            from compiler.cache_policy import CachePolicy
            from engine.cache_manager import CacheManager

            policy = CachePolicy.from_dict(raw_policy)
            if not policy.is_empty:
                num_blocks = module.metadata.get("num_blocks", 1000)
                self._cache_mgr = CacheManager(policy, num_blocks=int(num_blocks))
                self._uses_cache_manager = True
                for idef in policy.intercepts:
                    self._intercepts_by_op.setdefault(idef.op_name, []).append(idef)
                    self._intercept_slab_ops.add(idef.op_name)

        # ── Legacy KV cache state ──────────────────────────
        self._kv_cache: torch.Tensor | None = None
        self._block_tables: dict[str, list[int]] = {}
        self._block_size: int = 16
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

        self._current_positions: torch.Tensor | None = None
        self._current_is_decode: bool = False
        self._sda_layer_count: int = 0

        # ── Function output dump (for per-layer diagnosis) ──
        self._dump_dir: str | None = dump_dir or os.environ.get("DUMP_PY_LAYERS")
        self._dump_call_counter: dict[int, int] = {}

    def _detect_static_shape(self) -> bool:
        for _name, tp in self._function.inputs:
            import re
            m = re.search(r"tensor<(\d+)x(\d+)x", tp)
            if m and m.group(2) != "?":
                return True
        return False

    @property
    def function(self) -> MlirFunction:
        return self._function

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        logits, _ = self._run_forward(input_ids, capture_kv=False, **kwargs)
        return logits

    def forward_with_kv(
        self, input_ids: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        return self._run_forward(input_ids, capture_kv=True, **kwargs)

    def _run_forward(
        self, input_ids: torch.Tensor, capture_kv: bool, **kwargs: Any
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        """Run ALL functions sequentially, chaining outputs→inputs."""
        ssa_values: dict[str, torch.Tensor] = {}

        for fi, func in enumerate(self._module.functions):
            self._function = func  # switch current function for weight lookups

            # ── Initialize inputs for this function ──────────
            if fi == 0:
                # First function: input_ids is the GlobalInput
                if func.inputs:
                    first_input = func.inputs[0][0]
                    ssa_values[first_input] = input_ids
                    ssa_values[first_input.lstrip("%")] = input_ids
                    for named_input, _ in func.inputs[1:]:
                        clean = named_input.replace("%", "")
                        if clean in kwargs:
                            ssa_values[named_input] = kwargs[clean]
                            ssa_values[clean] = kwargs[clean]
            else:
                # Subsequent functions: all inputs are already in ssa_values
                # from previous function's outputs. Just ensure they exist.
                for inp_name, _ in func.inputs:
                    clean = inp_name.replace("%", "")
                    if clean not in ssa_values:
                        ssa_values.setdefault(inp_name, torch.tensor([]))
                        ssa_values.setdefault(clean, torch.tensor([]))

            self._reset_forward_state(kwargs)

            if self._uses_cache_manager and self._cache_mgr is not None:
                self._cache_mgr.begin_step(self._block_tables)

            # ── Execute ops ─────────────────────────────────
            for op in func.ops:
                result = self._execute_op(op, ssa_values)
                if result is not None and op.results:
                    ssa_values[op.results[0]] = result

            # ── Dump function outputs (all, not just first) ──
            if self._dump_dir:
                import numpy as np

                os.makedirs(self._dump_dir, exist_ok=True)
                call_idx = self._dump_call_counter.get(fi, 0)
                for i, (out_name, _, _) in enumerate(func.outputs):
                    clean = out_name.replace("%", "")
                    if clean in ssa_values:
                        fpath = os.path.join(
                            self._dump_dir,
                            f"py_func_{fi}_out{i}_{clean}.npy"
                        )
                        np.save(fpath, ssa_values[clean].detach().cpu().numpy())
                self._dump_call_counter[fi] = call_idx + 1

            # ── Store outputs for next function ──────────────
            for out_name, _, _ in func.outputs:
                clean = out_name.replace("%", "")
                if clean not in ssa_values:
                    ssa_values[clean] = ssa_values.get(clean, torch.tensor([]))

        # ── Return last function's output ───────────────────
        last_func = self._module.functions[-1]
        if not capture_kv:
            output_ssa = None
            if last_func.outputs:
                output_ssa = last_func.outputs[0][0]
            if output_ssa is None and last_func.ops:
                last_op = last_func.ops[-1]
                if last_op.results:
                    output_ssa = last_op.results[-1]
            if output_ssa and output_ssa in ssa_values:
                return ssa_values[output_ssa], []
            if ssa_values:
                return list(ssa_values.values())[-1], []
            return torch.tensor([]), []

        logits = torch.tensor([])
        kv_tensors: list[tuple[str, torch.Tensor]] = []
        for i, (out_name, _, _) in enumerate(last_func.outputs):
            if out_name in ssa_values:
                if i == 0:
                    logits = ssa_values[out_name]
                else:
                    kv_tensors.append((out_name, ssa_values[out_name]))
        if logits.numel() == 0 and last_func.ops:
            last_op = last_func.ops[-1]
            if last_op.results and last_op.results[-1] in ssa_values:
                logits = ssa_values[last_op.results[-1]]
        return logits, kv_tensors

    def _reset_forward_state(self, kwargs: dict[str, Any]) -> None:
        self._sda_layer_count = 0
        self._current_positions = kwargs.get("positions", None)
        self._current_is_decode = kwargs.get("is_decode", False)

    def _execute_op(self, op: MlirOp, ssa_values: dict[str, torch.Tensor]) -> torch.Tensor | None:
        if op.name in _WEIGHT_OPS:
            wname = op.attributes.get("name", "") or op.attributes.get('"name"', "")
            if not wname and op.operands:
                wname = op.operands[0]
            if wname in self._weights:
                return self._weights[wname]
            if wname in self._function.weights:
                return self._function.weights[wname]
            return None

        # ── Cache Manager path ─────────────────────────
        if self._uses_cache_manager and op.op_name in self._intercept_slab_ops:
            self._handle_cache_intercept(op, ssa_values)

        # ── Legacy heuristic path ──────────────────────
        elif "sdpa" in op.name.lower() and self._kv_cache is not None and self._block_tables:
            self._intercept_sdpa_legacy(op, ssa_values)

        tensor_inputs: list[torch.Tensor] = []
        for inp_name in op.operands:
            if inp_name in ssa_values:
                tensor_inputs.append(ssa_values[inp_name])
            elif inp_name in self._weights:
                tensor_inputs.append(self._weights[inp_name])
            elif inp_name in self._function.weights:
                tensor_inputs.append(self._function.weights[inp_name])
            elif inp_name.startswith("%"):
                clean = inp_name.replace("%", "")
                if clean in ssa_values:
                    tensor_inputs.append(ssa_values[clean])
                elif clean in self._weights:
                    tensor_inputs.append(self._weights[clean])
                elif clean in self._function.weights:
                    tensor_inputs.append(self._function.weights[clean])
                else:
                    available = sorted(ssa_values.keys())[:20]
                    raise KeyError(
                        f"Op '{op.name}' needs input '{inp_name}' which was never produced. "
                        f"Available SSA values (first 20): {available}"
                    )
            elif inp_name.lstrip("%") in ssa_values:
                tensor_inputs.append(ssa_values[inp_name.lstrip("%")])
            elif inp_name.lstrip("%") in self._weights:
                tensor_inputs.append(self._weights[inp_name.lstrip("%")])
            else:
                available = sorted(ssa_values.keys())[:20]
                raise KeyError(
                    f"Op '{op.name}' needs input '{inp_name}' which was never produced. "
                    f"Available SSA values (first 20): {available}"
                )

        result = self._hal.execute(op.op_name, tensor_inputs, **op.attributes)

        if op.op_name in ("copy_", "copy") and op.operands and op.results:
            dst_name = op.operands[0]
            if dst_name in ssa_values and result is not None:
                ssa_values[dst_name] = result

        return result

    # ── Cache Manager intercept ───────────────────────────

    def _handle_cache_intercept(self, op: MlirOp, ssa_values: dict[str, torch.Tensor]) -> None:
        if self._cache_mgr is None:
            return
        for idef in self._intercepts_by_op.get(op.op_name, []):
            if idef.direction not in ("write", "read_write"):
                continue

            slab_id = idef.slab_id
            data = self._extract_source(idef, op, ssa_values)
            if data is None:
                continue

            layer_idx = self._cache_mgr.resolve_layer(slab_id)
            positions = self._current_positions
            if positions is not None:
                flat_pos = positions.squeeze(0) if positions.dim() >= 2 else positions
                self._cache_mgr.write_paged(slab_id, layer_idx, data, flat_pos)

            if idef.direction == "read_write" and self._block_tables and self._current_is_decode:
                max_seq = self._max_seq_from_tables(self._block_tables)
                gathered = self._cache_mgr.read_paged(slab_id, layer_idx, max_seq)
                source_key = self._get_source_ssa_key(idef, op)
                if source_key and source_key in ssa_values:
                    orig_dtype = ssa_values[source_key].dtype
                    if gathered.dim() == 4:
                        gathered = gathered.permute(0, 2, 1, 3)
                    ssa_values[source_key] = gathered.to(orig_dtype)

    def _extract_source(self, idef: Any, op: MlirOp, ssa_values: dict[str, torch.Tensor]) -> torch.Tensor | None:
        source = idef.source
        if source.startswith("operand["):
            idx_str = source[len("operand["):-1]
            idx = int(idx_str)
            if idx < len(op.operands):
                key = op.operands[idx]
                return ssa_values.get(key)
        elif source == "output":
            if op.results:
                key = op.results[0]
                return ssa_values.get(key)
        return None

    def _get_source_ssa_key(self, idef: Any, op: MlirOp) -> str | None:
        source = idef.source
        if source.startswith("operand["):
            idx_str = source[len("operand["):-1]
            idx = int(idx_str)
            if idx < len(op.operands):
                return op.operands[idx]
        return None

    # ── Legacy intercept (backward compat) ─────────────────

    def _intercept_sdpa_legacy(self, op: MlirOp, ssa_values: dict[str, torch.Tensor]) -> None:
        layer_idx = self._sda_layer_count
        self._sda_layer_count += 1

        if len(op.operands) < 3:
            return

        k_name = op.operands[1]
        v_name = op.operands[2]
        k_new = ssa_values.get(k_name)
        v_new = ssa_values.get(v_name)

        if k_new is None or v_new is None:
            return

        k_new_sq, v_new_sq = _normalize_kv_for_cache(
            k_new, v_new, self._num_kv_heads, self._head_dim
        )

        positions = self._current_positions
        if positions is not None:
            flat_pos = positions.squeeze(0) if positions.dim() >= 2 else positions
            self._write_kv_flat(k_new_sq, v_new_sq, flat_pos, self._block_tables, layer_idx)

        token_count = k_new_sq.shape[0] if k_new_sq.dim() >= 1 else 1
        if token_count == 1 and self._block_tables and self._current_is_decode:
            max_seq = self._max_seq_from_tables(self._block_tables)
            k_full, v_full = self._gather_kv_flat(self._block_tables, max_seq, layer_idx)
            if k_full.dim() == 4:
                k_gathered = k_full.permute(0, 2, 1, 3)
                v_gathered = v_full.permute(0, 2, 1, 3)
            else:
                k_gathered = k_full
                v_gathered = v_full
            ssa_values[k_name] = k_gathered.to(k_new.dtype)
            ssa_values[v_name] = v_gathered.to(v_new.dtype)

    # ── Backward-compat public interface ──────────────────

    def set_kv_cache(
        self,
        kv_cache: torch.Tensor | None,
        block_tables: dict[str, list[int]] | None = None,
        block_size: int = 16,
        num_kv_heads: int = 0,
        head_dim: int = 0,
    ) -> None:
        if self._uses_cache_manager:
            self._block_tables = block_tables or {}
            return
        self._kv_cache = kv_cache
        self._block_tables = block_tables or {}
        self._block_size = block_size
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

    def prepare_kv_blocks(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        if self._uses_cache_manager:
            return torch.zeros(1)
        shape = (num_blocks, num_layers, 2, block_size, num_kv_heads, head_dim)
        return torch.zeros(shape, dtype=dtype)

    def write_kv_to_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        block_tables: dict[str, list[int]],
        layer_idx: int = 0,
    ) -> None:
        if self._uses_cache_manager:
            return
        self._write_kv_flat(key, value, positions, block_tables, layer_idx)

    def gather_kv_from_cache(
        self,
        block_tables: dict[str, list[int]],
        max_seq_len: int,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._uses_cache_manager and self._cache_mgr is not None:
            k = self._cache_mgr.read_paged("k", layer_idx, max_seq_len)
            v = self._cache_mgr.read_paged("v", layer_idx, max_seq_len)
            return k, v
        return self._gather_kv_flat(block_tables, max_seq_len, layer_idx)
