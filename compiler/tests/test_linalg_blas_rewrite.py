"""Seam tests for the post-bufferization linalg→SFA BLAS rewrite."""

from __future__ import annotations

from typing import Any


def _count_op(module_text: str, op_name: str) -> int:
    return module_text.count(op_name)


def test_linalg_blas_rewrite_replaces_rank2_and_rank3(mlir_context: Any) -> None:
    from compiler.pipeline.stages import _lower_linalg_matmul_to_sfa_blas_action

    text = r"""
module {
  func.func @mm(%a: memref<?x768xf32>, %b: memref<768x3072xf32>,
                %c: memref<?x3072xf32>) {
    linalg.matmul ins(%a, %b : memref<?x768xf32>, memref<768x3072xf32>)
                  outs(%c : memref<?x3072xf32>)
    return
  }
  func.func @bmm(%a: memref<1x?x768xf32>, %b: memref<1x768x768xf32>,
                 %c: memref<1x?x768xf32>) {
    linalg.batch_matmul ins(%a, %b : memref<1x?x768xf32>, memref<1x768x768xf32>)
                        outs(%c : memref<1x?x768xf32>)
    return
  }
}
"""
    import mlir.ir as ir

    with mlir_context, ir.Location.unknown():
        module = ir.Module.parse(text)
        _lower_linalg_matmul_to_sfa_blas_action(module)
        out = str(module)

    assert _count_op(out, "linalg.matmul") == 0
    assert _count_op(out, "linalg.batch_matmul") == 0
    assert out.count("call @sfa_sgemm") == 1
    assert out.count("call @sfa_batch_sgemm(") == 1
    assert "func.func private @sfa_sgemm" in out
    assert "func.func private @sfa_batch_sgemm" in out


def _run_action(text: str, mlir_context: Any) -> str:
    import mlir.ir as ir

    from compiler.pipeline.stages import _lower_linalg_matmul_to_sfa_blas_action

    with mlir_context, ir.Location.unknown():
        module = ir.Module.parse(text)
        _lower_linalg_matmul_to_sfa_blas_action(module)
        return str(module)


def test_linalg_blas_rewrite_replaces_rank2_transb_chain(mlir_context: Any) -> None:
    """transpose(weight) -> linalg.matmul must become one sfa_sgemm_transb.

    The original weight is ``[N, K]`` (torch linear layout).  The lowered
    IR materializes ``W^T`` and calls ``sfa_sgemm`` with NoTrans; the rewrite
    must instead pass the original ``[N, K]`` memref to a bridge that calls
    cblas_sgemm with CblasTrans, eliminating the per-step copy.
    """
    text = r"""
module {
  func.func @mm(%a: memref<?x768xf32>, %w: memref<3072x768xf32>,
                %c: memref<?x3072xf32>) {
    %t = memref.alloc() {alignment = 64 : i64} : memref<768x3072xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                                      affine_map<(d0, d1) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel"]}
      ins(%w : memref<3072x768xf32>) outs(%t : memref<768x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.matmul ins(%a, %t : memref<?x768xf32>, memref<768x3072xf32>)
                  outs(%c : memref<?x3072xf32>)
    return
  }
}
"""
    out = _run_action(text, mlir_context)

    assert _count_op(out, "linalg.matmul") == 0
    assert _count_op(out, "linalg.generic") == 0
    assert out.count("call @sfa_sgemm_transb") == 1
    assert "func.func private @sfa_sgemm_transb" in out


def test_linalg_blas_rewrite_replaces_rank3_transb_broadcast_chain(mlir_context: Any) -> None:
    """transpose -> broadcast -> batch_matmul becomes sfa_batch_sgemm_transb."""
    text = r"""
module {
  func.func @bmm(%a: memref<1x?x768xf32>, %w: memref<3072x768xf32>,
                 %c: memref<1x?x3072xf32>) {
    %t = memref.alloc() {alignment = 64 : i64} : memref<768x3072xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                                      affine_map<(d0, d1) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel"]}
      ins(%w : memref<3072x768xf32>) outs(%t : memref<768x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %b = memref.alloc() {alignment = 64 : i64} : memref<1x768x3072xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                    iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%t : memref<768x3072xf32>) outs(%b : memref<1x768x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.batch_matmul ins(%a, %b : memref<1x?x768xf32>, memref<1x768x3072xf32>)
                        outs(%c : memref<1x?x3072xf32>)
    return
  }
}
"""
    out = _run_action(text, mlir_context)

    assert _count_op(out, "linalg.batch_matmul") == 0
    assert _count_op(out, "linalg.generic") == 0
    assert out.count("call @sfa_batch_sgemm_transb") == 1
    assert out.count("call @sfa_batch_sgemm(") == 0
    assert "func.func private @sfa_batch_sgemm_transb" in out


def test_linalg_blas_rewrite_rejects_non_identity_transpose_chain(mlir_context: Any) -> None:
    """A transpose-like generic with arithmetic body must fall back to the
    existing NoTrans bridge, not be misclassified as a pure transb weight."""
    text = r"""
module {
  func.func @mm(%a: memref<?x768xf32>, %w: memref<3072x768xf32>,
                %c: memref<?x3072xf32>) {
    %t = memref.alloc() {alignment = 64 : i64} : memref<768x3072xf32>
    %one = arith.constant 1.000000e+00 : f32
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                                      affine_map<(d0, d1) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel"]}
      ins(%w : memref<3072x768xf32>) outs(%t : memref<768x3072xf32>) {
    ^bb0(%in: f32, %out: f32):
      %x = arith.addf %in, %one : f32
      linalg.yield %x : f32
    }
    linalg.matmul ins(%a, %t : memref<?x768xf32>, memref<768x3072xf32>)
                  outs(%c : memref<?x3072xf32>)
    return
  }
}
"""
    out = _run_action(text, mlir_context)

    assert out.count("call @sfa_sgemm") == 1
    assert out.count("call @sfa_sgemm_transb") == 0


def test_linalg_blas_rewrite_rejects_broadcast_chain_without_transpose(mlir_context: Any) -> None:
    """A batch matmul whose B operand is not produced by a pure transpose
    keeps the regular NoTrans sfa_batch_sgemm rewrite."""
    text = r"""
module {
  func.func @bmm(%a: memref<1x?x768xf32>, %b: memref<1x768x768xf32>,
                 %c: memref<1x?x768xf32>) {
    linalg.batch_matmul ins(%a, %b : memref<1x?x768xf32>, memref<1x768x768xf32>)
                        outs(%c : memref<1x?x768xf32>)
    return
  }
}
"""
    out = _run_action(text, mlir_context)

    assert out.count("call @sfa_batch_sgemm(") == 1
    assert out.count("call @sfa_batch_sgemm_transb") == 0


def test_linalg_blas_rewrite_noop_without_matmuls(mlir_context: Any) -> None:
    from compiler.pipeline.stages import _lower_linalg_matmul_to_sfa_blas_action

    text = r"""
module {
  func.func @add(%a: memref<?x4xf32>, %b: memref<?x4xf32>,
                 %c: memref<?x4xf32>) {
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                      affine_map<(d0, d1) -> (d0, d1)>,
                                      affine_map<(d0, d1) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : memref<?x4xf32>, memref<?x4xf32>)
      outs(%c : memref<?x4xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %0 = arith.addf %x, %y : f32
      linalg.yield %0 : f32
    }
    return
  }
}
"""
    import mlir.ir as ir

    with mlir_context, ir.Location.unknown():
        module = ir.Module.parse(text)
        _lower_linalg_matmul_to_sfa_blas_action(module)
        out = str(module)

    assert "sfa_sgemm" not in out
    assert _count_op(out, "linalg.generic") == 1
