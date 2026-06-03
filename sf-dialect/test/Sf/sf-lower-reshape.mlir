// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Shape/reshape lowering: sf.transpose → linalg.transpose,
// sf.unsqueeze → tensor.reshape
//===----------------------------------------------------------------------===//

module {
  // Test 1: sf.transpose — swap last two dims → linalg.transpose
  func.func @transpose(%arg0: tensor<2x4xf32>) -> tensor<4x2xf32> {
    %0 = "sf.transpose"(%arg0) {dim0 = 0 : i64, dim1 = 1 : i64} : (tensor<2x4xf32>) -> tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
  }

  // Test 2: sf.unsqueeze — insert dim 1 at position 0 → tensor.reshape
  func.func @unsqueeze(%arg0: tensor<8xf32>) -> tensor<1x8xf32> {
    %0 = "sf.unsqueeze"(%arg0) {dim = 0 : i64} : (tensor<8xf32>) -> tensor<1x8xf32>
    return %0 : tensor<1x8xf32>
  }
}

// CHECK:      func.func @transpose
// CHECK:      linalg.transpose
// CHECK-NOT:  "sf.transpose"
// CHECK:      func.func @unsqueeze
// CHECK:      tensor.reshape
// CHECK-NOT:  "sf.unsqueeze"
