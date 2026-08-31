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

//===----------------------------------------------------------------------===//
// sf.logical_and lowering — mixed i1/f32 operand conversion
//===----------------------------------------------------------------------===//

module {
  func.func @logical_and_i1_mask(%mask: tensor<1x1x4x4xi1>,
                                  %values: tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32> {
    %0 = "sf.logical_and"(%mask, %values)
         : (tensor<1x1x4x4xi1>, tensor<1x1x4x4xf32>) -> tensor<1x1x4x4xf32>
    return %0 : tensor<1x1x4x4xf32>
  }
}

// CHECK:      func.func @logical_and_i1_mask
// CHECK:      arith.uitofp
// CHECK:      linalg.generic
// CHECK:      arith.mulf
// CHECK-NOT:  sf.logical_and
