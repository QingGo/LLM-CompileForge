// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Element-wise op lowering: sf.add, sf.mul, sf.relu → linalg.generic
//===----------------------------------------------------------------------===//

module {
  // Test 1: sf.add
  func.func @add(%arg0: tensor<2x8xf32>, %arg1: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %0 = sf.add %arg0, %arg1 : tensor<2x8xf32>, tensor<2x8xf32> -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // Test 2: sf.mul
  func.func @mul(%arg0: tensor<2x8xf32>, %arg1: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %0 = sf.mul %arg0, %arg1 : tensor<2x8xf32>, tensor<2x8xf32> -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // Test 3: sf.relu
  func.func @relu(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %0 = sf.relu %arg0 : tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}

// CHECK:      func.func @add
// CHECK:      linalg.generic
// CHECK:      arith.addf
// CHECK:      func.func @mul
// CHECK:      linalg.generic
// CHECK:      arith.mulf
// CHECK:      func.func @relu
// CHECK:      linalg.generic
// CHECK:      arith.maxnumf
// CHECK-NOT:  sf.add
// CHECK-NOT:  sf.mul
// CHECK-NOT:  sf.relu
