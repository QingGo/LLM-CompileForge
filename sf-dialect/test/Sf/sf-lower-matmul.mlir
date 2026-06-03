// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// sf.matmul lowering — 2D matmul → linalg.matmul
//===----------------------------------------------------------------------===//

module {
  func.func @matmul_2d(%arg0: tensor<2x4xf32>, %arg1: tensor<4x2xf32>) -> tensor<2x2xf32> {
    %0 = "sf.matmul"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// CHECK:      func.func @matmul_2d
// CHECK:      linalg.matmul
// CHECK-NOT:  sf.matmul
// CHECK-NOT:  "sf.
