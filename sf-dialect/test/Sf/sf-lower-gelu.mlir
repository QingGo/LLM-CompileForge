// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// sf.gelu lowering — GELU approximation → linalg.generic with math.tanh
//===----------------------------------------------------------------------===//

module {
  func.func @gelu(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %0 = sf.gelu %arg0 : tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}

// CHECK:      func.func @gelu
// CHECK:      linalg.generic
// CHECK:      math.tanh
// CHECK-NOT:  sf.gelu
