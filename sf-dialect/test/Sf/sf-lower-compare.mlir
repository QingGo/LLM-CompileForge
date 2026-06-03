// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// sf.le lowering — element-wise <= → linalg.generic with arith.cmpf
//===----------------------------------------------------------------------===//

module {
  func.func @le(%arg0: tensor<2x8xf32>, %arg1: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %0 = sf.le %arg0, %arg1 : tensor<2x8xf32>, tensor<2x8xf32> -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}

// CHECK:      func.func @le
// CHECK:      linalg.generic
// CHECK:      arith.cmpf
// CHECK-NOT:  sf.le
