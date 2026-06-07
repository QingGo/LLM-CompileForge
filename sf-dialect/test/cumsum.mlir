// REQUIRES: sf-opt
// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

// Test lowering of sf.cumsum (2D, dim=1, static).
// CHECK-LABEL: func.func @cumsum_2d_dim1
func.func @cumsum_2d_dim1(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: linalg.copy
  // CHECK: arith.addf
  // CHECK-NOT: sf.cumsum
  %0 = "sf.cumsum"(%arg0) {dim = 1 : i64} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// Test lowering of sf.cumsum (2D, dim=0, static).
// CHECK-LABEL: func.func @cumsum_2d_dim0
func.func @cumsum_2d_dim0(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  // CHECK: linalg.copy
  // CHECK: arith.addf
  // CHECK-NOT: sf.cumsum
  %0 = "sf.cumsum"(%arg0) {dim = 0 : i64} : (tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// Test lowering of sf.cumsum (dynamic batch dim).
// CHECK-LABEL: func.func @cumsum_dynamic
func.func @cumsum_dynamic(%arg0: tensor<?x10xf32>) -> tensor<?x10xf32> {
  // CHECK: linalg.copy
  // CHECK: arith.addf
  // CHECK-NOT: sf.cumsum
  %0 = "sf.cumsum"(%arg0) {dim = 1 : i64} : (tensor<?x10xf32>) -> tensor<?x10xf32>
  return %0 : tensor<?x10xf32>
}
