// REQUIRES: sf-opt
// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

// Test lowering of sf.add with static shapes.
// CHECK-LABEL: func.func @add_static
func.func @add_static(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>)
  // CHECK-NOT: sf.add
  %0 = "sf.add"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Test lowering of sf.add with dynamic shapes.
// CHECK-LABEL: func.func @add_dynamic
func.func @add_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: linalg.generic
  // CHECK-NOT: sf.add
  %0 = "sf.add"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// Test lowering of sf.mul.
// CHECK-LABEL: func.func @mul_static
func.func @mul_static(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: linalg.generic
  // CHECK-NOT: sf.mul
  %0 = "sf.mul"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// Test lowering of sf.sub.
// CHECK-LABEL: func.func @sub_static
func.func @sub_static(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: linalg.generic
  // CHECK-NOT: sf.sub
  %0 = "sf.sub"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
