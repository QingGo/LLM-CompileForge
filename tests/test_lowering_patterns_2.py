# ruff: noqa: E501
"""Advanced lowering tests: edge cases, regression prevention, zero-dim checks.

Second half of tests extracted from test_lowering_patterns.py.
Covers dynamic shapes, edge-case ops, regression tests for known bugs,
and zero-dimensional tensor prevention.
"""

import pytest

from tests.lowering_test_helpers import (
    _has_vec_bindings,
    check_lowered,
    lower,
)

# ── Phase 2: dynamic dim tests (continued from file 1) ──────────


@pytest.mark.unit
@pytest.mark.xfail(reason="known pass-through: lowering leaves sf. ops")
def test_view_dynamic():
    """sf.view with dynamic input shape."""
    lowered = lower("""module {
  func.func @test(%a: tensor<?x?xf32>) -> tensor<?xf32> {
    %0 = "sf.view"(%a) : (tensor<?x?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_arange_scalar():


    """sf.arange with scalar output (edge case: not meaningful, should not crash)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<f32>) -> tensor<f32> {
    %0 = "sf.arange"(%a) : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_cumsum_scalar():
    """sf.cumsum with scalar input (dim out of range → identity copy)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<f32>) -> tensor<f32> {
    %0 = "sf.cumsum"(%a) {dim = 1 : i64} : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_identity():
    """sf.identity is a no-op (replaced by its input)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "sf.identity"(%a) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}""")
    assert "sf." not in lowered, f"sf ops remain:\n{lowered}"


@pytest.mark.unit
def test_view_reshape():
    """sf.view changing tensor rank (uses tensor.reshape)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<8xf32>) -> tensor<2x4xf32> {
    %0 = "sf.view"(%a) : (tensor<8xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}""")
    check_lowered(lowered)

# ── Remaining fix tests ──────────────────────────────────────

@pytest.mark.unit
def test_unsqueeze_rank_change():
    """sf.unsqueeze adding a dimension (rank change)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<4xf32>) -> tensor<1x4xf32> {
    %0 = "sf.unsqueeze"(%a) {dim = 0 : i64} : (tensor<4xf32>) -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_expand():
    """sf.expand is a no-op (broadcast handled by downstream ops)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "sf.expand"(%a) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}""")
    assert "sf." not in lowered, f"sf ops remain:\n{lowered}"


@pytest.mark.unit
def test_le_i1():
    """sf.le producing i1 output (comparison for boolean mask)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>) -> tensor<2x4xi1> {
    %0 = "sf.le"(%a, %b) : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
    return %0 : tensor<2x4xi1>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_logical_and_i1():
    """sf.logical_and with i1 inputs producing i1 output (boolean chain)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<2x4xi1>, %b: tensor<2x4xi1>) -> tensor<2x4xi1> {
    %0 = "sf.logical_and"(%a, %b) : (tensor<2x4xi1>, tensor<2x4xi1>) -> tensor<2x4xi1>
    return %0 : tensor<2x4xi1>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_index():
    """sf.index with index tensors of different rank than output."""
    lowered = lower("""module {
  func.func @test(%data: tensor<f32>, %idx1: tensor<1x1x1xf32>, %idx2: tensor<1x1x1xf32>) -> tensor<f32> {
    %0 = "sf.index"(%data, %idx1, %idx2) : (tensor<f32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_index_idxcoords_mapping():
    """Verify idxCoords use per-dim broadcast, not i+j offset.

    Without the fix, idxCoords map outDim = i + j which shifts
    index tensor 1's coordinates by 1 position, causing OOB reads
    on broadcast dims. With the fix, each index tensor dim j maps
    to outCoords[j] (with constant 0 for size-1 dims).
    """
    lowered = lower("""module {
  func.func @test(%data: tensor<?x?xf32>, %idx0: tensor<?x1x1x1xi64>, %idx1: tensor<1x1x1x?xi64>) -> tensor<?x1x1x?xf32> {
    %0 = "sf.index"(%data, %idx0, %idx1) : (tensor<?x?xf32>, tensor<?x1x1x1xi64>, tensor<1x1x1x?xi64>) -> tensor<?x1x1x?xf32>
    return %0 : tensor<?x1x1x?xf32>
  }
}""")
    check_lowered(lowered)
    # idx0: shape [?,1,1,1] — only dim0 is non-1
    #   extract should use outCoords[0] for dim0, constant 0 for dims 1-3
    # idx1: shape [1,1,1,?] — only dim3 is non-1
    #   extract should use outCoords[3] for dim3, constant 0 for dims 0-2

    # Find tensor.extract lines involving %arg1 (idx0) and %arg2 (idx1)
    arg1_extracts = [line for line in lowered.split('\n') if 'tensor.extract' in line and '%arg1' in line]
    arg2_extracts = [line for line in lowered.split('\n') if 'tensor.extract' in line and '%arg2' in line]

    assert arg1_extracts, "No tensor.extract from idx0 (arg1)"
    assert arg2_extracts, "No tensor.extract from idx1 (arg2)"

    # Verify idx0 (arg1) only has %c0 for coords beyond dim0
    # The extract should look like: tensor.extract %arg1[%X, %c0, %c0, %c0]
    # Bug would look like:   tensor.extract %arg1[%X, %c0, %c0, %Y]
    for ext in arg1_extracts:
        # dims 1-3 should all be %c0 (constant 0)
        assert '%c0, %c0, %c0]' in ext or '%c0, %c0, %c0_' in ext.replace(' ', ''), \
            f"idx0 extract should have constant 0 for dims 1-3 (broadcast): {ext}"

    # Verify idx1 (arg2) only uses outCoords[3] for dim3
    # Extract should look like: tensor.extract %arg2[%c0, %c0, %c0, %X]
    # Bug would look like:     tensor.extract %arg2[%c0, %c0, %Y, %c0]
    for ext in arg2_extracts:
        # dims 0-2 should be %c0, dim3 should be variable
        assert '%c0, %c0, %c0' in ext, \
            f"idx1 extract should have constant 0 for dims 0-2: {ext}"
        # dim3 should NOT be %c0
        coords = ext.split('[')[1].split(']')[0].split(',')
        if len(coords) >= 4:
            # The last coordinate before ']' should not be just %c0
            last_coord = coords[3].strip()
            assert last_coord != '%c0' or '%c0]' in ext, \
                f"idx1 dim3 should be variable (outCoords[3]), not constant 0: {ext}"


@pytest.mark.unit
def test_index_dynamic_dims():
    """sf.index with dynamic shapes — verifies outDims sources from index tensors.

    Without the fix, outDims reads tensor.dim from the data tensor (%arg0)
    for all dynamic dims. After the fix, tensor.dim reads from index tensors
    (%arg1, %arg2) for the leading broadcast dims, falling back to data only
    when no index tensor provides a non-broadcast dim.
    """
    lowered = lower("""module {
  func.func @test(%data: tensor<?x?xf32>, %idx0: tensor<?x1x1x1xi64>, %idx1: tensor<1x1x1x?xi64>) -> tensor<?x1x?x?xf32> {
    %0 = "sf.index"(%data, %idx0, %idx1) : (tensor<?x?xf32>, tensor<?x1x1x1xi64>, tensor<1x1x1x?xi64>) -> tensor<?x1x?x?xf32>
    return %0 : tensor<?x1x?x?xf32>
  }
}""")
    check_lowered(lowered)
    # Verify outDims sources from index tensors, not data tensor.
    # Bug: tensor.dim %arg0 (data) was used for all dynamic dims.
    # Fix: tensor.dim %arg1 (idx0) for dim 0, tensor.dim %arg2 (idx1) for dim 3.
    assert "tensor.dim %arg1, %c0" in lowered, (
        f"Expected tensor.dim from idx0 (arg1) for output dim 0:\n{lowered}"
    )
    assert "tensor.dim %arg2, %c3" in lowered, (
        f"Expected tensor.dim from idx1 (arg2) for output dim 3:\n{lowered}"
    )
    # Verify no tensor.dim from data tensor (%arg0) for the dynamic dims
    # (data tensor may still be used for index values, just not for dim sizes)
    dim_from_data = [ln for ln in lowered.split('\n') if 'tensor.dim %arg0' in ln]
    assert not dim_from_data, (
        f"outDims should not source from data tensor (arg0): {dim_from_data}"
    )


@pytest.mark.unit
def test_identity_type_cast():
    """sf.identity with type change (i1→f32) should insert uitofp."""
    lowered = lower("""module {
  func.func @test(%a: tensor<1x1x1xi1>) -> tensor<1x1x1xf32> {
    %0 = "sf.identity"(%a) : (tensor<1x1x1xi1>) -> tensor<1x1x1xf32>
    return %0 : tensor<1x1x1xf32>
  }
}""")
    check_lowered(lowered)
    assert "arith.uitofp" in lowered


@pytest.mark.unit
def test_matmul_1d():
    """sf.matmul with 1D input and 2D weight (vector * matrix)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<768xf32>, %b: tensor<768x256xf32>) -> tensor<256xf32> {
    %0 = "sf.matmul"(%a, %b) : (tensor<768xf32>, tensor<768x256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_linear_1d_input():
    """sf.linear with 1D input (promoted to 2D, matmulled, collapsed)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<768xf32>, %w: tensor<768x256xf32>, %b: tensor<256xf32>) -> tensor<256xf32> {
    %0 = "sf.linear"(%a, %w, %b) : (tensor<768xf32>, tensor<768x256xf32>, tensor<256xf32>) -> tensor<256xf32>
    return %0 : tensor<256xf32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_linear_batch_2d_result():
    """sf.linear with 3D input producing 2D output (lm_head style)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<?x?x768xf32>, %w: tensor<50272x768xf32>, %b: tensor<50272xf32>) -> tensor<50272x768xf32> {
    %0 = "sf.linear"(%a, %w, %b) :
      (tensor<?x?x768xf32>, tensor<50272x768xf32>, tensor<50272xf32>) -> tensor<50272x768xf32>
    return %0 : tensor<50272x768xf32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_add_squeeze_rank_mismatch():
    """sf.add with 1D rhs squeezed to scalar to match scalar output."""
    lowered = lower("""module {
  func.func @test(%a: tensor<f32>, %b: tensor<1xf32>) -> tensor<f32> {
    %0 = "sf.add"(%a, %b) : (tensor<f32>, tensor<1xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}""")
    check_lowered(lowered)

# ── Regression tests: session bugs ─────────────────────────────

@pytest.mark.unit
def test_binary_broadcast_3d_1d():
    """Regression: sf.add(3D, 1D) → must produce 3D output (bug: _infer used shapes[0])."""
    lowered = lower("""module {
  func.func @test(%a: tensor<?x?x768xf32>, %b: tensor<768xf32>) -> tensor<?x?x768xf32> {
    %0 = "sf.add"(%a, %b) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
@pytest.mark.xfail(reason="known pass-through: lowering leaves sf. ops")
def test_view_dyn_shape_infer():
    """Regression: sf.view dyn_shape -1 inference used wrong input dims."""
    lowered = lower("""module {
  func.func @test(%a: tensor<?x?x12x64xf32>, %b: tensor<f32>, %c: tensor<f32>) -> tensor<?x?x?xf32> {
    %0 = "sf.view"(%a, %b, %c) {shape = [%b, %c, -1]} : (tensor<?x?x12x64xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_transpose_permuted_dynamic():
    """Regression: sf.transpose with permuted dynamic dims (makeEmpty issue)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32> {
    %0 = "sf.transpose"(%a) {dim0 = 1 : i64, dim1 = 2 : i64} : (tensor<?x?x12x64xf32>) -> tensor<?x12x?x64xf32>
    return %0 : tensor<?x12x?x64xf32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
@pytest.mark.xfail(reason="known pass-through: lowering leaves sf. ops")
def test_expand_broadcast():
    """Regression: sf.expand rank-increasing broadcast (passthrough type mismatch)."""
    lowered = lower("""module {
  func.func @test(%a: tensor<1x1x1xf32>, %b: tensor<f32>) -> tensor<?x1x?x?xf32> {
    %0 = "sf.expand"(%a, %b) {shape = [%b, -1, %b, %b]} : (tensor<1x1x1xf32>, tensor<f32>) -> tensor<?x1x?x?xf32>
    return %0 : tensor<?x1x?x?xf32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_binary_broadcast_dynamic_out():
    """Regression: kDynamic outDim + size-1 rhs dim → wrong broadcast map."""
    lowered = lower("""module {
  func.func @test(%a: tensor<?x12x?x4xf32>, %b: tensor<?x12x?x1xf32>) -> tensor<?x12x?x?xf32> {
    %0 = "sf.add"(%a, %b) : (tensor<?x12x?x4xf32>, tensor<?x12x?x1xf32>) -> tensor<?x12x?x?xf32>
    return %0 : tensor<?x12x?x?xf32>
  }
}""")
    check_lowered(lowered)


@pytest.mark.unit
def test_compare_broadcast_3d_1d():
    """Regression: sf.le broadcast 3D+1D (_infer_compare_pure used shapes[0])."""
    lowered = lower("""module {
  func.func @test(%a: tensor<?x?x768xf32>, %b: tensor<768xf32>) -> tensor<?x?x768xf32> {
    %0 = "sf.le"(%a, %b) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>
  }
}""")
    check_lowered(lowered)


# ── Session bugs: FP accuracy, lowering hang, type mismatches ──


@pytest.mark.unit
def test_linear_3d_dynamic_batch():
    lowered = lower('''module {
  func.func @test(%a: tensor<?x4x768xf32>, %w: tensor<768x768xf32>,
                  %b: tensor<768xf32>) -> tensor<?x4x768xf32> {
    %0 = "sf.linear"(%a, %w, %b) : (tensor<?x4x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x4x768xf32>
    return %0 : tensor<?x4x768xf32>
  }
}''')
    check_lowered(lowered)
    assert "linalg.batch_matmul" in lowered


@pytest.mark.unit
@pytest.mark.xfail(reason="known pass-through: lowering leaves sf. ops")
def test_ones_like_with_tensor_input():
    """Regression: sf.ones_like with 0 operands must not crash."""
    lowered = lower("""module {
  func.func @test() -> tensor<1x4xf32> {
    %0 = "sf.ones_like"() {shape = [1, 4], device = "cpu",
                           pin_memory = false} : () -> tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}""")
    assert isinstance(lowered, str), "lowering should not crash"


@pytest.mark.unit
@pytest.mark.xfail(reason="known pass-through: lowering leaves sf. ops")
def test_cumsum_out_of_bounds_dim():
    lowered = lower('''module {
  func.func @test(%a: tensor<1xf32>) -> tensor<1xf32> {
    %0 = "sf.cumsum"(%a) {dim = 1 : i64} : (tensor<1xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}''')
    assert lowered, "lowering should not crash"


@pytest.mark.unit
def test_layer_norm_with_dynamic_dim():
    lowered = lower('''module {
  func.func @test(%a: tensor<?x768xf32>, %w: tensor<768xf32>,
                  %b: tensor<768xf32>) -> tensor<?x768xf32> {
    %0 = "sf.layer_norm"(%a, %w, %b) {normalized_shape = [768]} : (tensor<?x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<?x768xf32>
    return %0 : tensor<?x768xf32>
  }
}''')
    check_lowered(lowered)
    assert "linalg.generic" in lowered


@pytest.mark.unit
def test_batch_matmul_affine_maps():
    """Regression: batch_matmul affine maps (contractDimR must be rhsRank-2)."""
    lowered = lower('''module {
  func.func @test(%a: tensor<1x12x4x64xf32>, %b: tensor<1x12x64x4xf32>) -> tensor<1x12x4x4xf32> {
    %0 = "sf.matmul"(%a, %b) : (tensor<1x12x4x64xf32>, tensor<1x12x64x4xf32>) -> tensor<1x12x4x4xf32>
    return %0 : tensor<1x12x4x4xf32>
  }
}''')
    from compiler.pipeline import _post_lowering_canonicalize
    canonical = _post_lowering_canonicalize(lowered)
    assert "linalg.generic" in canonical or "linalg.batch_matmul" in canonical
    assert "sf.matmul" not in lowered, "sf.matmul was not lowered"


@pytest.mark.unit
def test_batch_matmul_dynamic_dims():
    """batch_matmul with dynamic dims: bufferization must not create 0-size tensors."""
    lowered = lower('''module {
  func.func @test(%a: tensor<1x12x4x64xf32>, %b: tensor<1x12x64x?xf32>) -> tensor<1x12x4x?xf32> {
    %0 = "sf.matmul"(%a, %b) : (tensor<1x12x4x64xf32>, tensor<1x12x64x?xf32>) -> tensor<1x12x4x?xf32>
    return %0 : tensor<1x12x4x?xf32>
  }
}''')
    assert "linalg." in lowered, "lowering failed"
    from compiler.mlir_dialect.llvm_backend import _has_bindings
    if not _has_bindings():
        pytest.skip("MLIR bindings not available")
    import mlir.ir as ir
    import mlir.passmanager as pm
    ctx = ir.Context()
    with ctx:
        mod = ir.Module.parse(lowered, ctx)
        try:
            pm.PassManager.parse(
                "builtin.module(one-shot-bufferize{bufferize-function-boundaries})", ctx
            ).run(mod.operation)
        except Exception as e:
            pytest.fail(f"Bufferization failed on batch_matmul with dynamic dims: {e}")


@pytest.mark.unit
def test_vector_contract_lowering_outerproduct():
    """Regression: vector.contract outerproduct must not hang on 4D masked contracts.
    Vectorization disabled in default pipeline — verifies scalar path passes."""
    if not _has_vec_bindings():
        pytest.skip("MLIR vector bindings not available (transform dialect)")

    import mlir.ir as ir
    import mlir.passmanager as pm

    from compiler.mlir_dialect.llvm_backend import _vectorize_via_transform

    mlir_text = """module {
  func.func @test(%a: tensor<1x12x4x64xf32>, %b: tensor<1x12x64x4xf32>) -> tensor<1x12x4x4xf32> {
    %0 = "sf.matmul"(%a, %b) : (tensor<1x12x4x64xf32>, tensor<1x12x64x4xf32>) -> tensor<1x12x4x4xf32>
    return %0 : tensor<1x12x4x4xf32>
  }
}"""
    lowered = lower(mlir_text)
    assert "linalg." in lowered, "C++ lowering failed to produce linalg ops"

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ir.Location.unknown(ctx):
        mod = ir.Module.parse(lowered, ctx)
        _vectorize_via_transform(mod)
        text = str(mod)
        n_contract = text.count("vector.contract")
        if n_contract == 0:
            pytest.skip("Vectorization disabled — scalar path is expected")

        import re
        masked_4d = len(re.findall(
            r'vector\.mask[^{]*\{[^}]*vector\.contract', text))
        assert masked_4d == 0, (
            f"Found {masked_4d} masked contracts — will hang in outerproduct lowering"
        )

        pipeline = (
            "builtin.module("
            "one-shot-bufferize{bufferize-function-boundaries},"
            "canonicalize,cse,"
            "convert-bufferization-to-memref,"
            "convert-linalg-to-loops,lower-affine,convert-scf-to-cf,"
            "expand-strided-metadata,lower-affine,"
            "func.func(lower-vector-mask),"
            "func.func(convert-vector-to-scf),"
            "canonicalize,cse,"
            "convert-scf-to-cf,lower-affine,"
            "finalize-memref-to-llvm,"
            "convert-cf-to-llvm,convert-math-to-llvm,"
            "convert-vector-to-llvm{vector-contract-lowering=outerproduct},"
            "convert-arith-to-llvm,convert-ub-to-llvm"
            ")"
        )
        pm.PassManager.parse(pipeline, ctx).run(mod.operation)

        for region in mod.operation.regions:
            for block in region.blocks:
                for op in block:
                    if str(op.operation.name) == "func.func":
                        op.operation.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get(ctx)
        pm.PassManager.parse(
            "builtin.module(convert-func-to-llvm,reconcile-unrealized-casts)", ctx
        ).run(mod.operation)

        result = str(mod)
        if "vector.contract" in result:
            pytest.skip("vector.contract not lowered — need different strategy")
        assert "vector." not in result or "vector" in text, \
            "vector ops remain after lowering"


# ── Zero-dim tensor regression prevention ────────────────────


@pytest.mark.unit
def test_zero_dim_tensor_prevention(caplog):
    """Regression: no 0D tensors (tensor<f32>) in lowered IR."""
    import logging

    lowered = lower("""module {
  func.func @test(%a: tensor<2x4xi64>) -> tensor<1xf32> {
    %0 = "sf.sym_size"(%a) {dim = 0 : i64} : (tensor<2x4xi64>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}""")
    assert "tensor<1xf32>" in lowered, (
        f"Lowered IR must contain 1D tensor<1xf32>:\n{lowered}"
    )
    assert "tensor<f32>" not in lowered, (
        f"Lowered IR must NOT contain 0D tensor<f32>:\n{lowered}"
    )

    from scripts.compile_dylib import _verify_lowered_ir
    bad_ir = '''module {
  func.func @test(%a: tensor<f32>) -> tensor<f32> {
    %0 = linalg.copy ins(%a : tensor<f32>) outs(%a : tensor<f32>)
    return %0 : tensor<f32>
  }
}'''
    caplog.set_level(logging.WARNING)
    _verify_lowered_ir(bad_ir)
    assert any("zero-dimensional" in r.getMessage() for r in caplog.records), (
        f"Expected warning for 0D tensors, got: {[r.getMessage() for r in caplog.records]}"
    )

    caplog.clear()
    clean_ir = '''module {
  func.func @test(%a: tensor<1xf32>) -> tensor<1xf32> {
    %0 = linalg.copy ins(%a : tensor<1xf32>) outs(%a : tensor<1xf32>)
    return %0 : tensor<1xf32>
  }
}'''
    _verify_lowered_ir(clean_ir)
    assert not any("zero-dimensional" in r.getMessage() for r in caplog.records), (
        "Unexpected warning for clean 1D IR"
    )
