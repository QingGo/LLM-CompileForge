// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s
// RUN: %sf-opt --sf-lower-to-linalg %s | %mlir-opt --one-shot-bufferize=bufferize-function-boundaries 2>&1 | FileCheck --check-prefix=BUF %s

// Test that sf.le lowering uses linalg.broadcast + identity-map linalg.generic
// instead of broadcast affine maps.
// CHECK-LABEL: func.func @le_broadcast
func.func @le_broadcast(%a: tensor<1x1x4x1xf32>, %b: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
  // CHECK: linalg.broadcast
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map, #map]
  // BUF-NOT: error
  %0 = "sf.le"(%a, %b) : (tensor<1x1x4x1xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// Test that both operands get broadcast when needed (multi-broadcast dim case).
// CHECK-LABEL: func.func @le_broadcast_both
func.func @le_broadcast_both(%a: tensor<1x1x4x1xf32>, %b: tensor<1x4x1x4xf32>) -> tensor<1x4x4x4xf32> {
  // CHECK: linalg.broadcast
  // CHECK: linalg.broadcast
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map, #map]
  // BUF-NOT: error
  %0 = "sf.le"(%a, %b) : (tensor<1x1x4x1xf32>, tensor<1x4x1x4xf32>) -> tensor<1x4x4x4xf32>
  return %0 : tensor<1x4x4x4xf32>
}

// Test sf.logical_and with broadcast.
// CHECK-LABEL: func.func @logical_and_broadcast
func.func @logical_and_broadcast(%a: tensor<1x1x4x1xf32>, %b: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
  // CHECK: linalg.broadcast
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map, #map]
  // BUF-NOT: error
  %0 = "sf.logical_and"(%a, %b) : (tensor<1x1x4x1xf32>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}

// Test with no broadcast needed (same shapes).
// CHECK-LABEL: func.func @le_no_broadcast
func.func @le_no_broadcast(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NOT: linalg.broadcast
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map1, #map1, #map1]
  // BUF-NOT: error
  %0 = "sf.le"(%a, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
