// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

// TDD: sf.view with rank change and -2 sentinel must NOT produce
// arith.constant -2 in the lowered output. The sentinel should be
// replaced with the dyn_shape operand (extracted from input tensor).

module {
  func.func @view_rank3_to_rank4(%input: tensor<?x?x768xf32>, %batch: tensor<1xf32>) -> tensor<?x?x12x64xf32> {
    %0 = "sf.view"(%input, %batch) {shape = [-2, -1, 12, 64]} : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x12x64xf32>
    return %0 : tensor<?x?x12x64xf32>
  }
}

// CHECK: func.func @view_rank3_to_rank4
// CHECK-NOT: arith.constant -2
// CHECK: tensor.reshape
