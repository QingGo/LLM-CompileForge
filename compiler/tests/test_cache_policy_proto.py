"""Proto round-trip tests for SfaCachePolicy messages.

Verifies that:
  1. SfaCachePolicy can be constructed, serialized, and deserialized
  2. All SfaSlabSpec and SfaInterceptSpec fields survive round-trip
  3. Binary serialization is deterministic (equal inputs → equal bytes)
"""

from __future__ import annotations

import os
import sys

# Ensure gen/proto/python is importable
_gen_proto = os.path.join(os.path.dirname(__file__), "..", "..", "gen", "proto", "python")
if _gen_proto not in sys.path:
    sys.path.insert(0, _gen_proto)

import sfa_abi_pb2  # noqa: E402  # type: ignore[import-untyped]


def _make_sample_policy() -> sfa_abi_pb2.SfaCachePolicy:
    """Build a cache policy with 1 paged slab + 1 intercept.

    Mirrors the structure produced by CachePolicy.for_llama(12, 8, 64)
    but with only the k slab for minimal round-trip testing.
    """
    slab = sfa_abi_pb2.SfaSlabSpec()
    slab.name = "k"
    slab.slab_type = "paged"
    slab.layout = "BNLD"
    slab.dtype = "float32"
    slab.num_blocks = 0  # computed at runtime
    slab.block_size = 16
    slab.num_layers = 12
    slab.num_heads = 8
    slab.head_dim = 64

    intercept = sfa_abi_pb2.SfaInterceptSpec()
    intercept.slab_id = "k"
    intercept.op_name_pattern = "scaled_dot_product_attention"
    intercept.intercept_type = "read_write"
    intercept.source = "operand[1]"
    intercept.layer = "sequential"
    intercept.param_indices.extend([1])

    policy = sfa_abi_pb2.SfaCachePolicy()
    policy.slabs.append(slab)
    policy.intercepts.append(intercept)
    policy.block_size = 16
    policy.max_requests = 256

    return policy


def test_proto_roundtrip_serialize_deserialize() -> None:
    """Serialize SfaCachePolicy to bytes and deserialize back — all fields match."""
    original = _make_sample_policy()

    # Serialize
    blob = original.SerializeToString()
    assert len(blob) > 0, "serialized blob must be non-empty"

    # Deserialize
    recovered = sfa_abi_pb2.SfaCachePolicy()
    recovered.ParseFromString(blob)

    # ── Top-level fields ──
    assert recovered.block_size == original.block_size
    assert recovered.max_requests == original.max_requests
    assert len(recovered.slabs) == len(original.slabs)
    assert len(recovered.intercepts) == len(original.intercepts)

    # ── Slab fields ──
    o_slab = original.slabs[0]
    r_slab = recovered.slabs[0]
    assert r_slab.name == o_slab.name == "k"
    assert r_slab.slab_type == o_slab.slab_type == "paged"
    assert r_slab.layout == o_slab.layout == "BNLD"
    assert r_slab.dtype == o_slab.dtype == "float32"
    assert r_slab.num_blocks == o_slab.num_blocks == 0
    assert r_slab.block_size == o_slab.block_size == 16
    assert r_slab.num_layers == o_slab.num_layers == 12
    assert r_slab.num_heads == o_slab.num_heads == 8
    assert r_slab.head_dim == o_slab.head_dim == 64

    # ── Intercept fields ──
    o_int = original.intercepts[0]
    r_int = recovered.intercepts[0]
    assert r_int.slab_id == o_int.slab_id == "k"
    assert r_int.op_name_pattern == o_int.op_name_pattern == "scaled_dot_product_attention"
    assert r_int.intercept_type == o_int.intercept_type == "read_write"
    assert r_int.source == o_int.source == "operand[1]"
    assert r_int.layer == o_int.layer == "sequential"
    assert list(r_int.param_indices) == list(o_int.param_indices) == [1]


def test_proto_roundtrip_deterministic() -> None:
    """Equal inputs produce equal serialized bytes (deterministic encoding)."""
    p1 = _make_sample_policy()
    p2 = _make_sample_policy()

    assert p1.SerializeToString() == p2.SerializeToString(), "serialization must be deterministic"


def test_proto_roundtrip_empty_policy() -> None:
    """Empty cache policy round-trips correctly."""
    original = sfa_abi_pb2.SfaCachePolicy()
    # defaults: block_size=0, max_requests=0 (proto3 zero defaults)

    blob = original.SerializeToString()
    assert len(blob) == 0, "empty proto3 message should serialize to zero bytes"

    recovered = sfa_abi_pb2.SfaCachePolicy()
    recovered.ParseFromString(blob)

    assert len(recovered.slabs) == 0
    assert len(recovered.intercepts) == 0
    assert recovered.block_size == 0
    assert recovered.max_requests == 0


def test_proto_roundtrip_multiple_intercepts() -> None:
    """Multiple intercepts on the same slab round-trip correctly."""
    policy = _make_sample_policy()

    # Add a second intercept for v slab
    intercept2 = sfa_abi_pb2.SfaInterceptSpec()
    intercept2.slab_id = "v"
    intercept2.op_name_pattern = "scaled_dot_product_attention"
    intercept2.intercept_type = "read_write"
    intercept2.source = "operand[2]"
    intercept2.layer = "sequential"
    intercept2.param_indices.extend([2])
    policy.intercepts.append(intercept2)

    blob = policy.SerializeToString()
    recovered = sfa_abi_pb2.SfaCachePolicy()
    recovered.ParseFromString(blob)

    assert len(recovered.intercepts) == 2
    assert recovered.intercepts[1].slab_id == "v"
    assert recovered.intercepts[1].source == "operand[2]"
    assert list(recovered.intercepts[1].param_indices) == [2]


def test_proto_default_field_values() -> None:
    """Proto3 default values: unset numeric fields are 0, strings are ''."""
    slab = sfa_abi_pb2.SfaSlabSpec()
    assert slab.name == ""
    assert slab.num_layers == 0
    assert slab.num_heads == 0
    # Only set some fields — defaults fill the rest
    slab.name = "test"
    slab.num_layers = 6

    blob = slab.SerializeToString()
    recovered = sfa_abi_pb2.SfaSlabSpec()
    recovered.ParseFromString(blob)

    assert recovered.name == "test"
    assert recovered.num_layers == 6
    assert recovered.num_heads == 0  # unset → default
    assert recovered.layout == ""  # unset → default
