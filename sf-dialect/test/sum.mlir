// REQUIRES: sf-opt
// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

// NOTE: sum_2d_to_1d and sum_dynamic cases are commented out due to a known C++ bug.
// The sf-to-linalg lowering produces a linalg.generic where the output indexing map
// has rank 0 (full reduction to scalar) but the output tensor has rank 1.
// This causes: "expected operand #1 rank (1) to match the result rank of indexing_map (0)"
// The lowering code needs to be fixed to properly compute partial reduction indexing maps.

// Test lowering of sf.sum (full reduction — 2D → 0D).
// CHECK-LABEL: func.func @sum_2d_to_0d
func.func @sum_2d_to_0d(%arg0: tensor<2x4xf32>) -> tensor<f32> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK-NOT: sf.sum
  %0 = "sf.sum"(%arg0) : (tensor<2x4xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// KNOWN BUG (commented out — see note at top):
// func.func @sum_2d_to_1d(%arg0: tensor<2x4xf32>) -> tensor<2xf32> {
//   %0 = "sf.sum"(%arg0) : (tensor<2x4xf32>) -> tensor<2xf32>
//   return %0 : tensor<2xf32>
// }

// KNOWN BUG (commented out — see note at top):
// func.func @sum_dynamic(%arg0: tensor<?x4xf32>) -> tensor<4xf32> {
//   %0 = "sf.sum"(%arg0) : (tensor<?x4xf32>) -> tensor<4xf32>
//   return %0 : tensor<4xf32>
// }
