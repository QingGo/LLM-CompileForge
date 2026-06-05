// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s
// Phase 5: sf-dialect lowering precision verification.
// Verifies that sf→linalg lowering produces numerically correct output
// by checking computed values in the lowered IR.

//===----------------------------------------------------------------------===//
// sf.matmul precision: [[1,2],[3,4]] @ [[0.5,0],[0,0.5]] = [[0.5,1],[1.5,2]]
//===----------------------------------------------------------------------===//

module {
  func.func @precision_matmul_2x2() -> tensor<2x2xf32> {
    %a = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %b = arith.constant dense<[[0.5, 0.0], [0.0, 0.5]]> : tensor<2x2xf32>
    %r = "sf.matmul"(%a, %b) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %r : tensor<2x2xf32>
  }
}

// CHECK:      func.func @precision_matmul_2x2
// CHECK:      linalg.matmul
// CHECK-NOT:  sf.matmul
// CHECK:      dense<{{\[\[}}5.000000e-01, 1.000000e+00], [1.500000e+00, 2.000000e+00]]>

//===----------------------------------------------------------------------===//
// sf.element_wise precision: [1,2,3,4] + [5,6,7,8] = [6,8,10,12]
//===----------------------------------------------------------------------===//

module {
  func.func @precision_add_4() -> tensor<4xf32> {
    %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
    %b = arith.constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
    %r = "sf.add"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}

// CHECK:      func.func @precision_add_4
// CHECK-NOT:  sf.add
// CHECK:      dense<[6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01]>

//===----------------------------------------------------------------------===//
// sf.matmul precision 2: [1,2] @ [[1,2],[3,4]] = [7,10]
// (matches the compiler precision contract fixture matmul_2x2_f32)
//===----------------------------------------------------------------------===//

module {
  func.func @precision_matmul_1x2() -> tensor<1x2xf32> {
    %a = arith.constant dense<[[1.0, 2.0]]> : tensor<1x2xf32>
    %b = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %r = "sf.matmul"(%a, %b) : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
    return %r : tensor<1x2xf32>
  }
}

// CHECK:      func.func @precision_matmul_1x2
// CHECK:      linalg.matmul
// CHECK-NOT:  sf.matmul
// CHECK:      dense<{{\[\[}}7.000000e+00, 1.000000e+01]]>
