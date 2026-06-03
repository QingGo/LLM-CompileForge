// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// sf.sum lowering — full reduction → linalg.generic with reduction iterators
//===----------------------------------------------------------------------===//

module {
  func.func @sum(%arg0: tensor<2x8xf32>) -> tensor<f32> {
    %0 = "sf.sum"(%arg0) : (tensor<2x8xf32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}

// CHECK:      func.func @sum
// CHECK:      linalg.generic
// CHECK-SAME: iterator_types = ["reduction", "reduction"]
// CHECK:      arith.addf
// CHECK-NOT:  "sf.
