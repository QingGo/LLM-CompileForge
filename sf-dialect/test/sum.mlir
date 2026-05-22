// REQUIRES: sf-opt
// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

// Test lowering of sf.sum (full reduction — 2D → 0D).
// CHECK-LABEL: func.func @sum_2d_to_0d
func.func @sum_2d_to_0d(%arg0: tensor<2x4xf32>) -> tensor<f32> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK-NOT: sf.sum
  %0 = "sf.sum"(%arg0) : (tensor<2x4xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// Test lowering of sf.sum (2D → 1D partial reduction).
// CHECK-LABEL: func.func @sum_2d_to_1d
func.func @sum_2d_to_1d(%arg0: tensor<2x4xf32>) -> tensor<2xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK-NOT: sf.sum
  %0 = "sf.sum"(%arg0) : (tensor<2x4xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// Test lowering of sf.sum (dynamic dims).
// CHECK-LABEL: func.func @sum_dynamic
func.func @sum_dynamic(%arg0: tensor<?x4xf32>) -> tensor<4xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK-NOT: sf.sum
  %0 = "sf.sum"(%arg0) : (tensor<?x4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
