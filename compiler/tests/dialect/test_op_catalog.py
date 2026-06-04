"""Seam tests for build_op_catalog() — populates SfaOpCatalog proto.

Tests that ``build_op_catalog()`` from ``compiler.mlir_dialect.op_catalog``
returns a populated ``SfaOpCatalog`` with 30+ HAL operators grouped by kind.
"""

from compiler.dialect.op_catalog import build_op_catalog


class TestBuildOpCatalog:
    """Verify build_op_catalog() produces a valid populated catalog."""

    def test_returns_non_empty_catalog(self) -> None:
        """Catalog contains 30+ HAL operators."""
        catalog = build_op_catalog()
        assert len(catalog.ops) >= 30, (
            f"Expected at least 30 ops in catalog, got {len(catalog.ops)}"
        )

    def test_all_ops_have_kind_and_name(self) -> None:
        """Every op in the catalog has a non-empty kind and name."""
        catalog = build_op_catalog()
        for op_def in catalog.ops:
            assert op_def.name, "Op has empty name"
            assert op_def.kind, f"Op '{op_def.name}' has empty kind"

    def test_all_ops_have_params(self) -> None:
        """Every op has at least one parameter defined."""
        catalog = build_op_catalog()
        for op_def in catalog.ops:
            assert len(op_def.params) >= 1, (
                f"Op '{op_def.name}' has no params"
            )

    def test_catalog_contains_core_ops(self) -> None:
        """Core ops (matmul, add, relu, softmax, layer_norm) are present."""
        catalog = build_op_catalog()
        op_names = {op_def.name for op_def in catalog.ops}
        expected = {"matmul", "add", "relu", "softmax", "layer_norm", "linear",
                    "gelu", "silu", "view", "embedding"}
        missing = expected - op_names
        assert not missing, f"Missing core ops: {missing}"
