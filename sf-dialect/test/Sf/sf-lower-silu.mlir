// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// sf.silu lowering — x * sigmoid(x) → linalg.generic with math.exp
//===----------------------------------------------------------------------===//

module {
  func.func @silu(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %0 = sf.silu %arg0 : tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}

// CHECK:      func.func @silu
// CHECK:      linalg.generic
// CHECK:      math.exp
// CHECK-NOT:  sf.silu
