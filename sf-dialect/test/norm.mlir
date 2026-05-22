// REQUIRES: sf-opt
// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

// Test lowering of sf.layer_norm (2D input, static).
// CHECK-LABEL: func.func @layer_norm_2d
func.func @layer_norm_2d(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<2x4xf32> {
  // CHECK: linalg.generic
  // CHECK: math.rsqrt
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK-NOT: sf.layer_norm
  %0 = "sf.layer_norm"(%arg0, %arg1, %arg2) : (tensor<2x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// Test lowering of sf.layer_norm (dynamic batch dim).
// CHECK-LABEL: func.func @layer_norm_dynamic
func.func @layer_norm_dynamic(%arg0: tensor<?x4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
  // CHECK: linalg.generic
  // CHECK: math.rsqrt
  // CHECK-NOT: sf.layer_norm
  %0 = "sf.layer_norm"(%arg0, %arg1, %arg2) : (tensor<?x4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// Test lowering of sf.rms_norm.
// CHECK-LABEL: func.func @rms_norm_static
func.func @rms_norm_static(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x4xf32> {
  // CHECK: linalg.generic
  // CHECK: math.rsqrt
  // CHECK: arith.mulf
  // CHECK-NOT: sf.rms_norm
  %0 = "sf.rms_norm"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
