"""Seam tests for CachePolicy.to_dict() / from_dict() round-trip.

Tests the serialization round-trip of ``CachePolicy`` from
``compiler.cache_policy`` — pure-Python dataclass with no MLIR,
no compiled dylib, no torch.
"""

from compiler.cache_policy import (
    CachePolicy,
    _InterceptSpec,
    _SlabSpec,
)


class TestCachePolicyRoundTrip:
    """Verify CachePolicy survives to_dict() → from_dict() round-trip."""

    def test_full_policy_roundtrip(self) -> None:
        """Create a CachePolicy with slabs + intercepts, round-trip, verify equality."""
        original = CachePolicy(
            slabs=[
                _SlabSpec(
                    slab_id="k",
                    storage="paged",
                    dims={"layers": 12, "heads": 8, "dim": 64},
                    layout="BNLD",
                    dtype="float32",
                ),
                _SlabSpec(
                    slab_id="v",
                    storage="paged",
                    dims={"layers": 12, "heads": 8, "dim": 64},
                    layout="BNLD",
                    dtype="float32",
                ),
            ],
            intercepts=[
                _InterceptSpec(
                    slab_id="k",
                    op_name="scaled_dot_product_attention",
                    direction="read_write",
                    source="operand[1]",
                    layer="sequential",
                ),
                _InterceptSpec(
                    slab_id="v",
                    op_name="scaled_dot_product_attention",
                    direction="read_write",
                    source="operand[2]",
                    layer="sequential",
                ),
            ],
            block_size=16,
            max_requests=256,
        )

        d = original.to_dict()
        recovered = CachePolicy.from_dict(d)

        # Top-level fields
        assert recovered.block_size == original.block_size
        assert recovered.max_requests == original.max_requests
        assert recovered.is_empty == original.is_empty

        # Slabs
        assert len(recovered.slabs) == len(original.slabs)
        for r_slab, o_slab in zip(recovered.slabs, original.slabs, strict=False):
            assert r_slab.slab_id == o_slab.slab_id
            assert r_slab.storage == o_slab.storage
            assert r_slab.dims == o_slab.dims
            assert r_slab.layout == o_slab.layout
            assert r_slab.dtype == o_slab.dtype

        # Intercepts
        assert len(recovered.intercepts) == len(original.intercepts)
        for r_int, o_int in zip(recovered.intercepts, original.intercepts, strict=False):
            assert r_int.slab_id == o_int.slab_id
            assert r_int.op_name == o_int.op_name
            assert r_int.direction == o_int.direction
            assert r_int.source == o_int.source
            assert r_int.layer == o_int.layer

    def test_empty_policy_roundtrip(self) -> None:
        """CachePolicy.none() (empty) survives round-trip."""
        original = CachePolicy.none()
        d = original.to_dict()
        recovered = CachePolicy.from_dict(d)

        assert recovered.is_empty
        assert recovered.slabs == []
        assert recovered.intercepts == []
        assert recovered.block_size == original.block_size

    def test_from_dict_none_returns_empty(self) -> None:
        """from_dict(None) returns an empty CachePolicy."""
        policy = CachePolicy.from_dict(None)
        assert policy.is_empty
        assert policy.slabs == []
        assert policy.intercepts == []

    def test_slab_spec_roundtrip(self) -> None:
        """_SlabSpec alone survives to_dict() → from_dict() round-trip."""
        original = _SlabSpec(
            slab_id="latent",
            storage="flat",
            dims={"layers": 4, "dim": 256},
            layout="NLD",
            dtype="bfloat16",
        )
        recovered = _SlabSpec.from_dict(original.to_dict())
        assert recovered.slab_id == original.slab_id
        assert recovered.storage == original.storage
        assert recovered.dims == original.dims
        assert recovered.layout == original.layout
        assert recovered.dtype == original.dtype

    def test_intercept_spec_roundtrip(self) -> None:
        """_InterceptSpec alone survives to_dict() → from_dict() round-trip."""
        original = _InterceptSpec(
            slab_id="state",
            op_name="state_evolve",
            direction="read_write",
            source="output",
            layer="sequential",
        )
        recovered = _InterceptSpec.from_dict(original.to_dict())
        assert recovered.slab_id == original.slab_id
        assert recovered.op_name == original.op_name
        assert recovered.direction == original.direction
        assert recovered.source == original.source
        assert recovered.layer == original.layer
