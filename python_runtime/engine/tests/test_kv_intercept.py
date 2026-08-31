"""Tests for MlirExecutor cache-policy intercept machinery.

These white-box tests pin the contract-driven intercept behavior:

  1. Intercepts keyed by (func_index, output_index) must apply ONLY to the
     SDPA op whose source operand was produced by that exact function
     output — no op-name fan-out across layers.
  2. Layer resolution is sequential across a forward (begin_step once),
     so layer L's K/V land in slab layer L.
  3. Decode gathers exactly the prefix through the fed token
     (max_position + 1), never a block-multiple with zero padding.
  4. Prefill (multi-token) writes K/V but never replaces the SDPA inputs.
"""

from __future__ import annotations

import pytest
import torch

from compiler.artifact import MlirFunction, MlirModule, MlirOp
from python_runtime.engine.cache_manager import CacheManager, _dict_to_proto_cache_policy
from python_runtime.engine.mlir_executor import MlirExecutor
from python_runtime.hal.pytorch_backend import PyTorchBackend

_NH, _HD = 2, 4


def _policy_dict(num_layers: int = 2) -> dict:
    """Cache policy mirroring the compiler's SDPA split output shape."""
    slabs = [
        {
            "slab_id": "k",
            "storage": "paged",
            "dims": {"layers": num_layers, "heads": _NH, "dim": _HD},
            "layout": "BNLD",
            "dtype": "float32",
        },
        {
            "slab_id": "v",
            "storage": "paged",
            "dims": {"layers": num_layers, "heads": _NH, "dim": _HD},
            "layout": "BNLD",
            "dtype": "float32",
        },
    ]
    intercepts = []
    for layer in range(num_layers):
        fi = 1 + layer * 2  # producer (Xa) function index
        for slab, oi in (("k", 0), ("v", 1)):
            intercepts.append(
                {
                    "slab_id": slab,
                    "op_name": "scaled_dot_product_attention",
                    "direction": "read_write",
                    "source": f"operand[{1 if slab == 'k' else 2}]",
                    "layer": "sequential",
                    "func_index": fi,
                    "output_index": oi,
                }
            )
    return {"slabs": slabs, "intercepts": intercepts, "block_size": 16, "max_requests": 256}


def _make_kv_module(num_layers: int = 2) -> MlirModule:
    """Synthetic split model: main_0 → Xa (K/V producers) → Xb (SDPA).

    Function index layout mirrors the real split:
      0: main_0      (produces %q0)
      1: main_1a     (produces %k0 @ oi0, %v0 @ oi1)
      2: main_1b     (SDPA consuming %q0/%k0/%v0)
      3: main_2a     (produces %k1 @ oi0, %v1 @ oi1)
      4: main_2b     (SDPA consuming %k1/%v1)
    """
    funcs = [
        MlirFunction(
            name="main_0",
            inputs=[("%input_ids", "tensor<?x?xi64>")],
            outputs=[("%q0", f"tensor<?x{_NH}x?x{_HD}xf32>", False)],
            ops=[
                MlirOp(
                    name="sf.identity",
                    dialect="sf",
                    op_name="identity",
                    operands=["%input_ids"],
                    results=["%q0"],
                ),
            ],
        )
    ]
    for layer in range(num_layers):
        producer = MlirFunction(
            name=f"main_{1 + layer * 2}a",
            inputs=[("%q_in", f"tensor<?x{_NH}x?x{_HD}xf32>")],
            outputs=[
                (f"%k{layer}", f"tensor<?x{_NH}x?x{_HD}xf32>", True),
                (f"%v{layer}", f"tensor<?x{_NH}x?x{_HD}xf32>", True),
            ],
            ops=[
                MlirOp(
                    name="sf.identity",
                    dialect="sf",
                    op_name="identity",
                    operands=["%q_in"],
                    results=[f"%k{layer}"],
                ),
                MlirOp(
                    name="sf.identity",
                    dialect="sf",
                    op_name="identity",
                    operands=["%q_in"],
                    results=[f"%v{layer}"],
                ),
            ],
        )
        consumer = MlirFunction(
            name=f"main_{2 + layer * 2}b",
            inputs=[
                (f"%q{layer}", f"tensor<?x{_NH}x?x{_HD}xf32>"),
                (f"%k{layer}", f"tensor<?x{_NH}x?x{_HD}xf32>"),
                (f"%v{layer}", f"tensor<?x{_NH}x?x{_HD}xf32>"),
                ("%attn_mask", "tensor<?x1x?x?xf32>"),
            ],
            outputs=[(f"%out{layer}", f"tensor<?x{_NH}x?x{_HD}xf32>", False)],
            ops=[
                MlirOp(
                    name="sf.scaled_dot_product_attention",
                    dialect="sf",
                    op_name="scaled_dot_product_attention",
                    operands=[f"%q{layer}", f"%k{layer}", f"%v{layer}", "%attn_mask"],
                    results=[f"%out{layer}"],
                ),
            ],
        )
        funcs.extend([producer, consumer])
    return MlirModule(functions=funcs, metadata={"cache_policy": _policy_dict(num_layers)})


def _make_executor(num_layers: int = 2) -> MlirExecutor:
    return MlirExecutor(_make_kv_module(num_layers), PyTorchBackend("cpu"))


def _sdpa_op(layer: int) -> MlirOp:
    return MlirOp(
        name="sf.scaled_dot_product_attention",
        dialect="sf",
        op_name="scaled_dot_product_attention",
        operands=[f"%q{layer}", f"%k{layer}", f"%v{layer}", "%attn_mask"],
        results=[f"%out{layer}"],
    )


def _kv_tensor(seq: int, base: float) -> torch.Tensor:
    """[1, nh, seq, hd] K/V with a per-layer identifiable value."""
    return torch.full((1, _NH, seq, _HD), base)


@pytest.mark.unit
class TestInterceptContractFiltering:
    """(func_index, output_index) contract filtering and layer isolation."""

    def test_each_layer_writes_its_own_slab_layer(self) -> None:
        ex = _make_executor(num_layers=2)
        assert ex._uses_cache_manager
        ex._block_tables = {"r0": [0, 1]}
        ex._cache_mgr.begin_step(ex._block_tables)
        ex._current_positions = torch.tensor([0, 1, 2])

        ssa: dict[str, torch.Tensor] = {}
        for layer, base in ((0, 1.0), (1, 2.0)):
            ssa[f"%k{layer}"] = _kv_tensor(3, base)
            ssa[f"%v{layer}"] = _kv_tensor(3, base + 10)
            ex._handle_cache_intercept(_sdpa_op(layer), ssa)

        for layer, base in ((0, 1.0), (1, 2.0)):
            k = ex._cache_mgr.read_paged("k", layer, max_seq_len=3)
            v = ex._cache_mgr.read_paged("v", layer, max_seq_len=3)
            assert torch.allclose(k, torch.full((1, 3, _NH, _HD), base)), f"layer {layer} K corrupted"
            assert torch.allclose(v, torch.full((1, 3, _NH, _HD), base + 10)), f"layer {layer} V corrupted"

    def test_intercept_ignores_foreign_producer(self) -> None:
        """An SDPA op whose K operand comes from an unknown producer must
        not match any intercept (contract pinning)."""
        ex = _make_executor(num_layers=1)
        ex._block_tables = {"r0": [0]}
        ex._cache_mgr.begin_step(ex._block_tables)
        ex._current_positions = torch.tensor([0])

        ssa = {
            "%k_foreign": _kv_tensor(1, 7.0),
            "%v_foreign": _kv_tensor(1, 8.0),
        }
        op = MlirOp(
            name="sf.scaled_dot_product_attention",
            dialect="sf",
            op_name="scaled_dot_product_attention",
            operands=["%q0", "%k_foreign", "%v_foreign", "%attn_mask"],
            results=["%out0"],
        )
        ex._handle_cache_intercept(op, ssa)

        # Nothing written: slab layer 0 stays zeroed.
        k = ex._cache_mgr.read_paged("k", 0, max_seq_len=1)
        assert torch.all(k == 0.0), "foreign producer must not match intercepts"


@pytest.mark.unit
class TestDecodeGather:
    """Decode reads must return exactly the prefix through the fed token."""

    def test_gather_is_exact_not_block_multiple(self) -> None:
        ex = _make_executor(num_layers=1)
        ex._block_tables = {"r0": [0, 1]}  # 2 blocks = 32 slots
        ex._cache_mgr.begin_step(ex._block_tables)

        # Prefill positions 0..6 with distinct values.
        ex._current_positions = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        ex._current_is_decode = False
        ssa = {"%k0": _kv_tensor(7, 1.0), "%v0": _kv_tensor(7, 11.0)}
        ex._handle_cache_intercept(_sdpa_op(0), ssa)

        # Decode: a NEW forward step (begin_step resets layer counters).
        # Fed token at position 7 writes its own K/V, then the gather must
        # span exactly 8 rows (0..7) — not the 32-row block multiple which
        # would pad attention with zero rows.
        ex._cache_mgr.begin_step(ex._block_tables)
        ex._current_positions = torch.tensor([7])
        ex._current_is_decode = True
        new_k = torch.full((1, _NH, 1, _HD), 99.0)
        new_v = torch.full((1, _NH, 1, _HD), 109.0)
        ssa = {"%k0": new_k, "%v0": new_v}
        ex._handle_cache_intercept(_sdpa_op(0), ssa)

        # ssa K/V replaced with gathered [1, nh, 8, hd].
        replaced_k = ssa["%k0"]
        assert replaced_k.shape == (1, _NH, 8, _HD), f"gather must be exact prefix, got {replaced_k.shape}"
        assert torch.allclose(replaced_k[0, :, :7, :], torch.full((_NH, 7, _HD), 1.0))
        assert torch.allclose(replaced_k[0, :, 7:8, :], torch.full((_NH, 1, _HD), 99.0))

    def test_prefill_does_not_replace_sdpa_inputs(self) -> None:
        ex = _make_executor(num_layers=1)
        ex._block_tables = {"r0": [0]}
        ex._cache_mgr.begin_step(ex._block_tables)
        ex._current_positions = torch.tensor([0, 1])
        ex._current_is_decode = False

        original_k = _kv_tensor(2, 3.0)
        original_v = _kv_tensor(2, 4.0)
        ssa = {"%k0": original_k, "%v0": original_v}
        ex._handle_cache_intercept(_sdpa_op(0), ssa)

        assert ssa["%k0"] is original_k, "prefill must not replace K with gathered cache"
        assert ssa["%v0"] is original_v, "prefill must not replace V with gathered cache"


@pytest.mark.unit
class TestBeginStepPlacement:
    """begin_step must run once per forward, not once per function."""

    def test_begin_step_once_per_forward(self) -> None:
        # Identity-only module (no SDPA) with a policy present, so the
        # cache manager is active but no intercept fires.
        mod = MlirModule(
            functions=[
                MlirFunction(
                    name="main_0",
                    inputs=[("%in", "tensor<?x?xf32>")],
                    outputs=[("%out0", "tensor<?x?xf32>", False)],
                    ops=[
                        MlirOp(
                            name="sf.identity",
                            dialect="sf",
                            op_name="identity",
                            operands=["%in"],
                            results=["%out0"],
                        ),
                    ],
                ),
                MlirFunction(
                    name="main_1",
                    inputs=[("%out0", "tensor<?x?xf32>")],
                    outputs=[("%out1", "tensor<?x?xf32>", False)],
                    ops=[
                        MlirOp(
                            name="sf.identity",
                            dialect="sf",
                            op_name="identity",
                            operands=["%out0"],
                            results=["%out1"],
                        ),
                    ],
                ),
            ],
            metadata={"cache_policy": _policy_dict(1)},
        )
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))

        begin_calls = [0]

        def counting_begin(block_tables: dict) -> None:
            begin_calls[0] += 1
            CacheManager.begin_step(ex._cache_mgr, block_tables)

        ex._cache_mgr.begin_step = counting_begin  # type: ignore[method-assign]
        ex._block_tables = {"r0": [0, 1]}
        ex.forward(torch.zeros(1, 3, dtype=torch.float32))
        assert begin_calls[0] == 1, f"begin_step must run once per forward, got {begin_calls[0]}"


@pytest.mark.unit
class TestJsonPolicyContractKeys:
    """The JSON→proto fallback must preserve func_index/output_index."""

    def test_param_indices_preserved(self) -> None:
        policy = _dict_to_proto_cache_policy(_policy_dict(2))
        keys = [(list(i.param_indices)) for i in policy.intercepts]
        assert keys == [[1, 0], [1, 1], [3, 0], [3, 1]], f"contract keys lost: {keys}"


@pytest.mark.unit
class TestEngineSlabAlignment:
    """LLMEngine must align the executor's slab capacity with the block pool."""

    def test_engine_realigns_slab_to_block_pool(self) -> None:
        from python_runtime.engine.llm_engine import LLMEngine

        mod = _make_kv_module(1)
        # Executor constructed BEFORE the engine → slab sized from the
        # metadata default (no num_blocks injected yet).
        ex = MlirExecutor(mod, PyTorchBackend("cpu"))
        assert ex._cache_mgr._num_blocks != 32

        LLMEngine(mod, PyTorchBackend("cpu"), executor=ex, num_blocks=32, max_batch_size=2)
        assert ex._cache_mgr._num_blocks == 32, "engine must realign slab capacity to the block pool"
