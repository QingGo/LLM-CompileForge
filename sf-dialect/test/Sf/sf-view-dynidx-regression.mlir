// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s
//
// Regression: dynIdx counter was NOT incremented for -2/-3 sentinels in
// SfLowerShape.cpp, causing a subsequent -1 to reuse an already-consumed
// dynShapeOperand instead of inferring from total element count.
//
// Bug: shape=[-2, -3, -1] with dynShapeOperands=[%batch=1, %seq=2]
//   -2 → operand[0], dynIdx stays 0
//   -3 → operand[1], dynIdx stays 0
//   -1 → dynIdx(0) < 2 → reuses operand[0]=batch → wrong output dim=1
//   Should: dynIdx=2 → infers total/(known_dims) = (1*2*12*64)/(2*12*64)=1
//
// Fix: ++dynIdx after consuming operand in -2/-3 branch.
//
// The lowered IR should contain a tensor.reshape with the correct
// dynamic dimensions from the RIGHT dynShapeOperands, not reusing
// operand[0] for the -1 sentinel.


module {
  // ── Test 1: model pattern [-2, -3, -1] ──
  // This is the exact pattern from opt-125m layer attention output reshape:
  //   sf.view %attn_out, %batch, %seq {shape=[-2, -3, -1]}
  //   4D [batch, seq, 12, 64] → 3D [batch, seq, 768]
  // The -1 should infer 768 (12*64), NOT reuse %batch or %seq.
  func.func @model_pattern(%input: tensor<?x?x12x64xf32>, %batch: tensor<1xf32>, %seq: tensor<1xf32>) -> tensor<?x?x768xf32> {
    %0 = "sf.view"(%input, %batch, %seq) {shape = [-2, -3, -1]} : (tensor<?x?x12x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>
  }

  // ── Test 2: explicit sentinels [-2, -3, -4] ──
  // Baseline: all dims use explicit sentinels, no -1 inference needed.
  // This should always work correctly (was never broken).
  func.func @all_explicit(%input: tensor<?x?x12x64xf32>, %batch: tensor<1xf32>, %seq: tensor<1xf32>, %hidden: tensor<1xf32>) -> tensor<?x?x768xf32> {
    %0 = "sf.view"(%input, %batch, %seq, %hidden) {shape = [-2, -3, -4]} : (tensor<?x?x12x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>
  }
}


// CHECK: func.func @model_pattern
// CHECK: arith.constant 768 : index
// CHECK-NOT: arith.constant 1 : index
// CHECK: tensor.reshape
// CHECK-SAME: tensor<3xindex>) -> tensor<?x?x768xf32>
// The output must be 3D [?, ?, 768] — NOT [?, ?, 1] which was the bug.
// The -1 sentinel correctly inferred 768 from total elements.

// CHECK: func.func @all_explicit
// CHECK: arith.constant 768 : index
// CHECK: tensor.reshape
// CHECK-SAME: tensor<3xindex>) -> tensor<?x?x768xf32>
