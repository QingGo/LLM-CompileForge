"""Tests for compiler.mlir_dialect.lowering — sf→linalg pass."""

from __future__ import annotations

import pytest

from compiler.mlir_dialect.lowering import sf_to_linalg_pass


def _check_op_count(text: str, op_name: str, min_count: int = 1) -> None:
    count = text.count(op_name)
    assert count >= min_count, f"Expected >= {min_count} '{op_name}', got {count}"


def _check_absent(text: str, op_name: str) -> None:
    assert op_name not in text, f"'{op_name}' should not be in lowered output"


@pytest.mark.unit
class TestLoweringBasicOps:

    def test_add_to_arith(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>, %b: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.add\"(%a, %b) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "linalg.generic", 1)
        _check_op_count(r, "arith.addf", 1)
        _check_absent(r, "sf.add")

    def test_mul_to_arith(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>, %b: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.mul\"(%a, %b) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "arith.mulf", 1)
        _check_absent(r, "sf.mul")

    def test_relu_to_arith_max(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.relu\"(%a) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "arith.maxnumf", 1)
        _check_absent(r, "sf.relu")

    def test_silu_decomposes(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.silu\"(%a) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "arith.negf", 1)
        _check_op_count(r, "math.exp", 1)
        _check_op_count(r, "arith.divf", 1)
        _check_absent(r, "sf.silu")

    def test_exp_to_math(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.exp\"(%a) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "math.exp", 1)
        _check_absent(r, "sf.exp")


@pytest.mark.unit
class TestLoweringMatmul:

    def test_matmul_to_linalg(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x4xf32>, %b: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %0 = \"sf.matmul\"(%a, %b) : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}""")
        _check_op_count(r, "linalg.matmul", 1)
        _check_absent(r, "sf.matmul")

    def test_linear_to_matmul_with_transpose(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>, %w: tensor<128x64xf32>, %b: tensor<128xf32>) -> tensor<2x128xf32> {
    %0 = \"sf.linear\"(%a, %w, %b) : (tensor<2x64xf32>, tensor<128x64xf32>, tensor<128xf32>) -> tensor<2x128xf32>
    return %0 : tensor<2x128xf32>
  }
}""")
        _check_op_count(r, "linalg.matmul", 1)
        _check_absent(r, "sf.linear")


@pytest.mark.unit
class TestLoweringChain:

    def test_add_relu_chain(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>, %b: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.add\"(%a, %b) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xf32>
    %1 = \"sf.relu\"(%0) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %1 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "linalg.generic", 2)
        _check_op_count(r, "arith.addf", 1)
        _check_op_count(r, "arith.maxnumf", 1)
        _check_absent(r, "sf.add")
        _check_absent(r, "sf.relu")

    def test_matmul_silu_chain(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x4xf32>, %w: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %0 = \"sf.matmul\"(%a, %w) : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = \"sf.silu\"(%0) : (tensor<2x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}""")
        _check_op_count(r, "linalg.matmul", 1)
        _check_op_count(r, "linalg.generic", 1)
        _check_absent(r, "sf.matmul")
        _check_absent(r, "sf.silu")


@pytest.mark.unit
class TestLoweringEdgeCases:

    def test_identity_passthrough(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.identity\"(%a) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "linalg.generic", 1)
        _check_absent(r, "sf.identity")

    def test_weight_not_lowered(self):
        r = sf_to_linalg_pass("""module {
  func.func @test() -> tensor<128x64xf32> {
    %0 = \"sf.weight\"() {name = \"w\"} : () -> tensor<128x64xf32>
    return %0 : tensor<128x64xf32>
  }
}""")
        _check_op_count(r, "sf.weight", 1)
        _check_absent(r, "linalg")

    def test_unknown_sf_op_finalized(self):
        """Unknown sf op → linalg.generic passthrough (finalize)."""
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.unknown_future_op\"(%a) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "linalg.generic", 1)
        _check_absent(r, "sf.unknown_future_op")

    def test_empty_module_noop(self):
        r = sf_to_linalg_pass("""module {
  func.func @test() {
    return
  }
}""")
        assert "module" in r

    def test_transpose_to_linalg(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x4xf32>) -> tensor<4x2xf32> {
    %0 = \"sf.transpose\"(%a) {dim0 = 0 : i64, dim1 = 1 : i64} : (tensor<2x4xf32>) -> tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
  }
}""")
        _check_op_count(r, "linalg.generic", 1)
        _check_absent(r, "sf.transpose")

    def test_gelu_to_arith_math(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.gelu\"(%a) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "math.tanh", 1)
        _check_absent(r, "sf.gelu")


@pytest.mark.unit
class TestLoweringReductions:

    def test_mean_lowered(self):
        """sf.mean → linalg.generic + arith.addf reduction + arith.divf."""
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<4x32x64xf32>) -> tensor<4x64xf32> {
    %0 = \"sf.mean\"(%a) {dim = 1 : i64} : (tensor<4x32x64xf32>) -> tensor<4x64xf32>
    return %0 : tensor<4x64xf32>
  }
}""")
        _check_op_count(r, "linalg.generic", 2)
        _check_op_count(r, "arith.addf", 1)
        _check_op_count(r, "arith.divf", 1)
        _check_absent(r, "sf.mean")

    def test_sum_lowered(self):
        """sf.sum → linalg.generic with arith.addf reduction."""
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<4x32x64xf32>) -> tensor<4x64xf32> {
    %0 = \"sf.sum\"(%a) {dim = 1 : i64} : (tensor<4x32x64xf32>) -> tensor<4x64xf32>
    return %0 : tensor<4x64xf32>
  }
}""")
        _check_op_count(r, "linalg.generic", 1)
        _check_op_count(r, "arith.addf", 1)
        _check_absent(r, "sf.sum")


@pytest.mark.unit
class TestLoweringShapeOps:

    def test_view_preserved(self):
        """sf.view kept as sf — needs tensor.reshape (rank-changing)."""
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x128xf32>) -> tensor<4x64xf32> {
    %0 = \"sf.view\"(%a) {shape = [4 : i64, 64 : i64]} : (tensor<2x128xf32>) -> tensor<4x64xf32>
    return %0 : tensor<4x64xf32>
  }
}""")
        _check_op_count(r, "sf.view", 1)

    def test_slice_lowered(self):
        """sf.slice → tensor.extract_slice."""
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x100xf32>) -> tensor<2x50xf32> {
    %0 = \"sf.slice\"(%a) {dim = 1 : i64, start = 0 : i64, end = 50 : i64} : (tensor<2x100xf32>) -> tensor<2x50xf32>
    return %0 : tensor<2x50xf32>
  }
}""")
        _check_op_count(r, "tensor.extract_slice", 1)
        _check_absent(r, "sf.slice")

    def test_select_lowered(self):
        """sf.select → linalg.generic with constant-index affine map."""
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x100x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.select\"(%a) {dim = 1 : i64, index = 5 : i64} : (tensor<2x100x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "linalg.generic", 1)
        _check_absent(r, "sf.select")

    def test_copy_to_linalg(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.copy_\"(%a) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "linalg.generic", 1)
        _check_absent(r, "sf.copy_")


@pytest.mark.unit
class TestLoweringComparisonOps:

    def test_eq_to_arith_cmpf(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>, %b: tensor<2x64xf32>) -> tensor<2x64xi1> {
    %0 = \"sf.eq\"(%a, %b) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xi1>
    return %0 : tensor<2x64xi1>
  }
}""")
        _check_op_count(r, "arith.cmpf", 1)
        _check_absent(r, "sf.eq")

    def test_gt_to_arith_cmpf(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>, %b: tensor<2x64xf32>) -> tensor<2x64xi1> {
    %0 = \"sf.gt\"(%a, %b) : (tensor<2x64xf32>, tensor<2x64xf32>) -> tensor<2x64xi1>
    return %0 : tensor<2x64xi1>
  }
}""")
        _check_op_count(r, "arith.cmpf", 1)
        _check_absent(r, "sf.gt")


@pytest.mark.unit
class TestLoweringMiscOps:

    def test_softmax_lowered(self):
        """sf.softmax → exp/sum/div decomposition."""
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %0 = \"sf.softmax\"(%a) {dim = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }
}""")
        _check_op_count(r, "math.exp", 1)
        _check_op_count(r, "arith.addf", 1)
        _check_op_count(r, "arith.divf", 1)
        _check_absent(r, "sf.softmax")

    def test_zeros_lowered(self):
        r = sf_to_linalg_pass("""module {
  func.func @test() -> tensor<2x64xf32> {
    %0 = \"sf.zeros\"() : () -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "tensor.empty", 1)
        _check_op_count(r, "arith.constant", 1)
        _check_absent(r, "sf.zeros")

    def test_ones_like_lowered(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.ones_like\"(%a) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "arith.constant", 1)
        _check_absent(r, "sf.ones_like")

    def test_softplus_lowered(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.softplus\"(%a) : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "math.exp", 1)
        _check_op_count(r, "math.log", 1)
        _check_absent(r, "sf.softplus")

    def test_clamp_min_lowered(self):
        r = sf_to_linalg_pass("""module {
  func.func @test(%a: tensor<2x64xf32>) -> tensor<2x64xf32> {
    %0 = \"sf.clamp_min\"(%a) {min = 0 : i64} : (tensor<2x64xf32>) -> tensor<2x64xf32>
    return %0 : tensor<2x64xf32>
  }
}""")
        _check_op_count(r, "arith.maxnumf", 1)
        _check_absent(r, "sf.clamp_min")


@pytest.mark.unit
class TestCoverageAllOps:

    def test_all_sf_ops_in_lower_table(self):
        from compiler.mlir_dialect.lowering import _LOWER_TABLE
        from compiler.mlir_dialect.sf import _ALL_OPS
        for op_name in sorted(_ALL_OPS.keys()):
            if op_name in ("sf.weight", "sf.constant"):
                continue  # intentionally skipped by the walk callback
            assert op_name in _LOWER_TABLE, \
                f"sf op '{op_name}' not in _LOWER_TABLE — add a lowering entry"
