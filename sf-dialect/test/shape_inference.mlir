// Test: Shape inference and verification for sf binary ops.
//
// Verifies that ops with broadcast-compatible shapes pass parsing
// (verifier runs during op construction).  All ops in this file use
// valid broadcast patterns and should parse without errors.
//
// RUN: %sf-opt %s | FileCheck %s

// CHECK-LABEL: func.func @add_same_static
func.func @add_same_static(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: sf.add
  %0 = "sf.add"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @add_broadcast_1dim
func.func @add_broadcast_1dim(%arg0: tensor<1x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: sf.add
  %0 = "sf.add"(%arg0, %arg1) : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @add_broadcast_diff_rank
func.func @add_broadcast_diff_rank(%arg0: tensor<2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: sf.add
  %0 = "sf.add"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @add_dynamic_agree
func.func @add_dynamic_agree(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> tensor<?x4xf32> {
  // CHECK: sf.add
  %0 = "sf.add"(%arg0, %arg1) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// CHECK-LABEL: func.func @mul_same_3d
func.func @mul_same_3d(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: sf.mul
  %0 = "sf.mul"(%arg0, %arg1) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @mul_both_broadcast
func.func @mul_both_broadcast(%arg0: tensor<5x1x6xf32>, %arg1: tensor<1x3x6xf32>) -> tensor<5x3x6xf32> {
  // CHECK: sf.mul
  %0 = "sf.mul"(%arg0, %arg1) : (tensor<5x1x6xf32>, tensor<1x3x6xf32>) -> tensor<5x3x6xf32>
  return %0 : tensor<5x3x6xf32>
}

// CHECK-LABEL: func.func @sub_same
func.func @sub_same(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK: sf.sub
  %0 = "sf.sub"(%arg0, %arg1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func.func @div_broadcast
func.func @div_broadcast(%arg0: tensor<4x1xf32>, %arg1: tensor<1x5xf32>) -> tensor<4x5xf32> {
  // CHECK: sf.div
  %0 = "sf.div"(%arg0, %arg1) : (tensor<4x1xf32>, tensor<1x5xf32>) -> tensor<4x5xf32>
  return %0 : tensor<4x5xf32>
}

// CHECK-LABEL: func.func @pow_broadcast
func.func @pow_broadcast(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> tensor<2x3xf32> {
  // CHECK: sf.pow
  %0 = "sf.pow"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @max_broadcast
func.func @max_broadcast(%arg0: tensor<2x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<2x3xf32> {
  // CHECK: sf.max
  %0 = "sf.max"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @le_broadcast
func.func @le_broadcast(%arg0: tensor<1x1x4x1xf32>, %arg1: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
  // CHECK: sf.le
  %0 = "sf.le"(%arg0, %arg1) : (tensor<1x1x4x1xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// CHECK-LABEL: func.func @gt_broadcast
func.func @gt_broadcast(%arg0: tensor<3x4xf32>, %arg1: tensor<4xf32>) -> tensor<3x4xf32> {
  // CHECK: sf.gt
  %0 = "sf.gt"(%arg0, %arg1) : (tensor<3x4xf32>, tensor<4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @lt_broadcast
func.func @lt_broadcast(%arg0: tensor<2x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> {
  // CHECK: sf.lt
  %0 = "sf.lt"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @eq_broadcast
func.func @eq_broadcast(%arg0: tensor<4x4xf32>, %arg1: tensor<4xf32>) -> tensor<4x4xf32> {
  // CHECK: sf.eq
  %0 = "sf.eq"(%arg0, %arg1) : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @ne_broadcast
func.func @ne_broadcast(%arg0: tensor<1x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: sf.ne
  %0 = "sf.ne"(%arg0, %arg1) : (tensor<1x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: func.func @logical_and_broadcast
func.func @logical_and_broadcast(%arg0: tensor<1x1x4x1xf32>, %arg1: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
  // CHECK: sf.logical_and
  %0 = "sf.logical_and"(%arg0, %arg1) : (tensor<1x1x4x1xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}
