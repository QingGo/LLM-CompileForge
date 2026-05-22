// REQUIRES: sf-opt
// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

// Test lowering of sf.matmul (2D, static).
// CHECK-LABEL: func.func @matmul_2d_static
func.func @matmul_2d_static(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>) -> tensor<4x4xf32> {
  // CHECK: linalg.matmul
  // CHECK-NOT: sf.matmul
  %0 = "sf.matmul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Test lowering of sf.matmul (1D x 2D).
// CHECK-LABEL: func.func @matmul_1d_by_2d
func.func @matmul_1d_by_2d(%arg0: tensor<768xf32>, %arg1: tensor<768x256xf32>) -> tensor<256xf32> {
  // CHECK: linalg.generic
  // CHECK-NOT: sf.matmul
  %0 = "sf.matmul"(%arg0, %arg1) : (tensor<768xf32>, tensor<768x256xf32>) -> tensor<256xf32>
  return %0 : tensor<256xf32>
}

// Test lowering of sf.matmul (batch, 4D).
// CHECK-LABEL: func.func @matmul_batch_4d
func.func @matmul_batch_4d(%arg0: tensor<1x12x4x64xf32>, %arg1: tensor<1x12x64x4xf32>) -> tensor<1x12x4x4xf32> {
  // CHECK: linalg.generic
  // CHECK-NOT: sf.matmul
  %0 = "sf.matmul"(%arg0, %arg1) : (tensor<1x12x4x64xf32>, tensor<1x12x64x4xf32>) -> tensor<1x12x4x4xf32>
  return %0 : tensor<1x12x4x4xf32>
}
