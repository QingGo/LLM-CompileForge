// REQUIRES: sf-opt
// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

// Test lowering of sf.linear with weight and bias.
// CHECK-LABEL: func.func @linear_with_bias
func.func @linear_with_bias(%arg0: tensor<2x64xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<128xf32>) -> tensor<2x128xf32> {
  // CHECK: linalg.matmul
  // CHECK: arith.addf
  // CHECK-NOT: sf.linear
  %0 = "sf.linear"(%arg0, %arg1, %arg2) : (tensor<2x64xf32>, tensor<128x64xf32>, tensor<128xf32>) -> tensor<2x128xf32>
  return %0 : tensor<2x128xf32>
}

// Test lowering of sf.linear without bias.
// CHECK-LABEL: func.func @linear_no_bias
func.func @linear_no_bias(%arg0: tensor<2x64xf32>, %arg1: tensor<128x64xf32>) -> tensor<2x128xf32> {
  // CHECK: linalg.matmul
  // CHECK-NOT: sf.linear
  %0 = "sf.linear"(%arg0, %arg1) : (tensor<2x64xf32>, tensor<128x64xf32>) -> tensor<2x128xf32>
  return %0 : tensor<2x128xf32>
}

// Test lowering of sf.linear with 3D input (batch matmul).
// CHECK-LABEL: func.func @linear_3d
func.func @linear_3d(%arg0: tensor<?x4x768xf32>, %arg1: tensor<768x768xf32>, %arg2: tensor<768xf32>) -> tensor<?x4x768xf32> {
  // CHECK: linalg.batch_matmul
  // CHECK: arith.addf
  // CHECK-NOT: sf.linear
  %0 = "sf.linear"(%arg0, %arg1, %arg2) : (tensor<?x4x768xf32>, tensor<768x768xf32>, tensor<768xf32>) -> tensor<?x4x768xf32>
  return %0 : tensor<?x4x768xf32>
}
