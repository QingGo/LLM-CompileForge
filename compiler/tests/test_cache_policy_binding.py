"""Tests for cache policy binding + proto serialization.

Covers:
  1. ``bind_cache_policy`` expands SDPA intercept templates into concrete
     per-``(func_index, output_index)`` entries (and preserves non-SDPA
     intercepts such as RWKV state).
  2. ``serialize_cache_policy`` emits the ``SfaCachePolicy`` proto the Rust
     runtime decodes via the ``sfa_cache_policy`` dylib symbol —
     ``param_indices == [func_index, output_index]``.
"""

from __future__ import annotations

from compiler.cache_policy import (
    CachePolicy,
    bind_cache_policy,
    serialize_cache_policy,
)
from gen.proto.python import sfa_abi_pb2


def _llama(num_layers: int = 2) -> CachePolicy:
    return CachePolicy.for_llama(num_layers=num_layers, num_kv_heads=12, head_dim=64)


TWO_LAYER_BINDINGS = [
    (0, 1, "k"),
    (0, 2, "v"),
    (2, 1, "k"),
    (2, 2, "v"),
]


class TestBindCachePolicy:
    def test_expands_sdpa_templates_to_per_output_entries(self) -> None:
        policy = bind_cache_policy(_llama(2), TWO_LAYER_BINDINGS)

        assert len(policy.intercepts) == 4
        for binding, intercept in zip(TWO_LAYER_BINDINGS, policy.intercepts, strict=True):
            fi, oi, slab_id = binding
            assert intercept.slab_id == slab_id
            assert intercept.func_index == fi
            assert intercept.output_index == oi
            assert intercept.op_name == "scaled_dot_product_attention"
            assert intercept.direction == "read_write"
            assert intercept.source == ("operand[1]" if slab_id == "k" else "operand[2]")

    def test_no_bindings_keeps_templates_unbound(self) -> None:
        policy = bind_cache_policy(_llama(2), [])
        assert len(policy.intercepts) == 2
        assert all(i.func_index is None for i in policy.intercepts)
        assert all(i.output_index is None for i in policy.intercepts)

    def test_preserves_non_sdpa_intercepts(self) -> None:
        rwkv = CachePolicy.for_rwkv(num_layers=2, state_dim=64)
        policy = bind_cache_policy(rwkv, TWO_LAYER_BINDINGS)
        # RWKV intercepts have no SDPA templates; bindings produce entries
        # with default template fallback, RWKV state entry stays intact.
        state = [i for i in policy.intercepts if i.op_name == "state_evolve"]
        assert len(state) == 1
        assert state[0].func_index is None

    def test_does_not_mutate_input_policy(self) -> None:
        original = _llama(2)
        _bound = bind_cache_policy(original, TWO_LAYER_BINDINGS)
        assert len(original.intercepts) == 2
        assert all(i.func_index is None for i in original.intercepts)


class TestSerializeCachePolicy:
    def test_roundtrip_param_indices(self) -> None:
        policy = bind_cache_policy(_llama(2), TWO_LAYER_BINDINGS)
        blob = serialize_cache_policy(policy)

        proto = sfa_abi_pb2.SfaCachePolicy()  # type: ignore[attr-defined]
        proto.ParseFromString(blob)

        assert len(proto.slabs) == 2
        assert proto.slabs[0].name == "k"
        assert proto.slabs[0].num_layers == 2
        assert proto.slabs[0].num_heads == 12
        assert proto.slabs[0].head_dim == 64
        assert proto.slabs[0].block_size == 16
        assert proto.slabs[1].name == "v"

        assert len(proto.intercepts) == 4
        for binding, ip in zip(TWO_LAYER_BINDINGS, proto.intercepts, strict=True):
            fi, oi, slab_id = binding
            assert ip.slab_id == slab_id
            assert list(ip.param_indices) == [fi, oi], ip.param_indices

        assert proto.block_size == 16
        assert proto.max_requests == 256

    def test_unbound_intercepts_serialize_without_param_indices(self) -> None:
        blob = serialize_cache_policy(_llama(1))
        proto = sfa_abi_pb2.SfaCachePolicy()  # type: ignore[attr-defined]
        proto.ParseFromString(blob)
        assert len(proto.intercepts) == 2
        assert all(len(ip.param_indices) == 0 for ip in proto.intercepts)
