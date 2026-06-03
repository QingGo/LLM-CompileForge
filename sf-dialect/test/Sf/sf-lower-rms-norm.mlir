// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// sf.rms_norm lowering — x / sqrt(mean(x²) + eps) * weight
// Decomposes into: square → reduce_sum/mean → sqrt → divide → multiply
//===----------------------------------------------------------------------===//

module {
  func.func @rms_norm(%input: tensor<2x8xf32>, %weight: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = "sf.rms_norm"(%input, %weight) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}

// CHECK:      func.func @rms_norm
// CHECK-DAG:  linalg.generic
// CHECK-DAG:  math.sqrt
// CHECK-DAG:  arith.addf
// CHECK-DAG:  arith.mulf
// CHECK-DAG:  arith.divf
// CHECK-NOT:  "sf.
