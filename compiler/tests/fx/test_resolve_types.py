"""Tests for _resolve_op_types scalar shape fix.

The fix at lines 117/121 of compiler/fx_to_mlir.py converts scalar tensor
shapes from () to (1,) when looking up weight shapes, producing
tensor<1xf32> instead of the invalid tensor<f32>.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.unit
class TestResolveOpTypesScalarFix:
    def test_scalar_weight_gets_shape_1(self):
        """Scalar weight with shape () should produce tensor<1xf32>."""
        from compiler.fx.utils import _resolve_op_types

        weights = {"w": torch.tensor(3.14)}
        ssa_map: dict[str, str] = {}
        shape_map: dict[str, tuple[tuple, str]] = {}

        input_types, _ = _resolve_op_types(
            "add",
            ["%w"],
            ssa_map,
            shape_map,
            weights,
            {},
        )
        assert any("tensor<1xf32>" in t for t in input_types), f"Expected tensor<1xf32> in {input_types}"

    def test_1d_weight_unchanged(self):
        """1-D tensor should still produce tensor<Nxf32>."""
        from compiler.fx.utils import _resolve_op_types

        weights = {"w": torch.randn(64)}
        ssa_map: dict[str, str] = {}
        shape_map: dict[str, tuple[tuple, str]] = {}

        input_types, _ = _resolve_op_types(
            "add",
            ["%w"],
            ssa_map,
            shape_map,
            weights,
            {},
        )
        assert any("tensor<64xf32>" in t for t in input_types), f"Expected tensor<64xf32> in {input_types}"

    def test_2d_weight_unchanged(self):
        """2-D tensor should still produce tensor<NxMxf32>."""
        from compiler.fx.utils import _resolve_op_types

        weights = {"w": torch.randn(64, 128)}
        ssa_map: dict[str, str] = {}
        shape_map: dict[str, tuple[tuple, str]] = {}

        input_types, _ = _resolve_op_types(
            "add",
            ["%w"],
            ssa_map,
            shape_map,
            weights,
            {},
        )
        assert any("tensor<64x128xf32>" in t for t in input_types), f"Expected tensor<64x128xf32> in {input_types}"

    def test_ssa_map_resolution_still_works(self):
        """%w resolved via ssa_map → w lookup still yields correct shape."""
        from compiler.fx.utils import _resolve_op_types

        weights = {"w": torch.tensor(3.14)}
        ssa_map = {"w": "%w"}
        shape_map: dict[str, tuple[tuple, str]] = {}

        input_types, _ = _resolve_op_types(
            "add",
            ["%w"],
            ssa_map,
            shape_map,
            weights,
            {},
        )
        assert any("tensor<1xf32>" in t for t in input_types), f"Expected tensor<1xf32> in {input_types}"

    def test_mixed_bf16_weight_promotes_to_f32(self):
        """f32 SSA + bf16 weight follows PyTorch promotion to f32."""
        from compiler.fx.utils import _resolve_op_types

        weights = {"w": torch.randn(64, dtype=torch.bfloat16)}
        ssa_map = {"a": "%a", "w": "%w"}
        shape_map = {"a": ((64,), "f32"), "w": ((64,), "bf16")}

        input_types, _ = _resolve_op_types(
            "add",
            ["%a", "%w"],
            ssa_map,
            shape_map,
            weights,
            {},
        )
        assert weights["w"].dtype == torch.float32
        assert all("tensor<64xf32>" in t for t in input_types), input_types

    def test_mixed_nonweight_floats_record_promotion(self):
        """bf16 * f32 non-weight operands promote to f32 with a cast record."""
        from compiler.fx.utils import _resolve_op_types

        ssa_map = {"a": "%a", "b": "%b"}
        shape_map = {"a": ((64,), "bf16"), "b": ((64,), "f32")}
        promotions: list[tuple[int, str, str]] = []

        input_types, _ = _resolve_op_types(
            "mul",
            ["%a", "%b"],
            ssa_map,
            shape_map,
            {},
            {},
            promotions,
        )
        assert promotions == [(0, "tensor<64xbf16>", "f32")]
        assert all("tensor<64xf32>" in t for t in input_types), input_types

    def test_shape_map_takes_priority(self):
        """shape_map entries should take priority over weight lookup."""
        from compiler.fx.utils import _resolve_op_types

        weights = {"w": torch.tensor(3.14)}
        ssa_map = {"w": "%w"}
        shape_map = {"w": ((128,), "f32")}

        input_types, _ = _resolve_op_types(
            "add",
            ["%w"],
            ssa_map,
            shape_map,
            weights,
            {},
        )
        assert any("tensor<128xf32>" in t for t in input_types), (
            f"Expected tensor<128xf32> from shape_map in {input_types}"
        )
