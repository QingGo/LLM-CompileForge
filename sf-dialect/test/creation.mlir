// REQUIRES: sf-opt
// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

// Test lowering of sf.ones_like (static shape).
// CHECK-LABEL: func.func @ones_like_static
func.func @ones_like_static(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: arith.constant 1.0
  // CHECK: linalg.fill
  // CHECK-NOT: sf.ones_like
  %0 = "sf.ones_like"(%arg0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// Test lowering of sf.ones_like (dynamic shape).
// CHECK-LABEL: func.func @ones_like_dynamic
func.func @ones_like_dynamic(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: arith.constant 1.0
  // CHECK: linalg.fill
  // CHECK-NOT: sf.ones_like
  %0 = "sf.ones_like"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// Test lowering of sf.new_ones (static).
// CHECK-LABEL: func.func @new_ones_static
func.func @new_ones_static(%arg0: tensor<f32>) -> tensor<?xf32> {
  // CHECK: arith.constant 1.0
  // CHECK: linalg.fill
  // CHECK-NOT: sf.new_ones
  %0 = "sf.new_ones"(%arg0) : (tensor<f32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
