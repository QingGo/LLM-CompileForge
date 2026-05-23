// Test: one-shot-bufferize does NOT error on kDynamic dims that broadcast.
//
// The C++ lowering (sf-le, sf-logical_and, makeBinaryOp) now refines
// output types so that dims which all inputs broadcast on (constant 0
// in the index map) are 1, not kDynamic.  This test verifies that after
// sf→linalg lowering, the generated linalg.generic ops survive bufferize.
//
// REQUIRES: mlir-opt
// RUN: %sf-opt --sf-lower-to-linalg %s | %mlir-opt -one-shot-bufferize=bufferize-function-boundaries 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @le_broadcast
func.func @le_broadcast(%arg0: tensor<1x1x4x1xf32>, %arg1: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
  // CHECK: linalg.generic
  %0 = "sf.le"(%arg0, %arg1) : (tensor<1x1x4x1xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// CHECK-LABEL: func.func @logical_and_broadcast
func.func @logical_and_broadcast(%arg0: tensor<1x1x4x1xf32>, %arg1: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
  // CHECK: linalg.generic
  %0 = "sf.logical_and"(%arg0, %arg1) : (tensor<1x1x4x1xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}
