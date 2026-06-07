// REQUIRES: sf-opt
// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

// Test lowering of sf.relu.
// CHECK-LABEL: func.func @relu
func.func @relu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.maxnumf
  // CHECK-NOT: sf.relu
  %0 = "sf.relu"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Test lowering of sf.silu (sigmoid * x).
// CHECK-LABEL: func.func @silu
func.func @silu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: linalg.generic
  // CHECK: math.exp
  // CHECK: arith.divf
  // CHECK-NOT: sf.silu
  %0 = "sf.silu"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Test lowering of sf.gelu (tanh approximation).
// CHECK-LABEL: func.func @gelu
func.func @gelu(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.tanh
  // CHECK-NOT: sf.gelu
  %0 = "sf.gelu"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// Test lowering of sf.sigmoid.
// CHECK-LABEL: func.func @sigmoid
func.func @sigmoid(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: linalg.generic
  // CHECK: math.exp
  // CHECK: arith.divf
  // CHECK-NOT: sf.sigmoid
  %0 = "sf.sigmoid"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
