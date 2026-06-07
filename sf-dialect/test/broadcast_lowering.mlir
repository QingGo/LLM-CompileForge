// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s
// RUN: %sf-opt --sf-lower-to-linalg %s | %mlir-opt --one-shot-bufferize=bufferize-function-boundaries 2>&1 | FileCheck --check-prefix=BUF %s

// Test that sf.le lowering uses linalg.generic with broadcast affine maps
// (the implementation now fuses broadcast into the generic's indexing maps
// instead of using separate linalg.broadcast ops).
// CHECK-LABEL: func.func @le_broadcast
func.func @le_broadcast(%a: tensor<1x1x4x1xf32>, %b: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1, #map1]
  // BUF-NOT: error
  %0 = "sf.le"(%a, %b) : (tensor<1x1x4x1xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// Test that both operands get broadcast affine maps when needed (multi-broadcast dim case).
// CHECK-LABEL: func.func @le_broadcast_both
func.func @le_broadcast_both(%a: tensor<1x1x4x1xf32>, %b: tensor<1x4x1x4xf32>) -> tensor<1x4x4x4xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map2, #map3, #map1]
  // BUF-NOT: error
  %0 = "sf.le"(%a, %b) : (tensor<1x1x4x1xf32>, tensor<1x4x1x4xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// Test sf.logical_and with broadcast.
// CHECK-LABEL: func.func @logical_and_broadcast
func.func @logical_and_broadcast(%a: tensor<1x1x4x1xf32>, %b: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map1, #map1]
  // BUF-NOT: error
  %0 = "sf.logical_and"(%a, %b) : (tensor<1x1x4x1xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// Test with no broadcast needed (same shapes).
// CHECK-LABEL: func.func @le_no_broadcast
func.func @le_no_broadcast(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map4, #map4, #map4]
  // BUF-NOT: error
  %0 = "sf.le"(%a, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
