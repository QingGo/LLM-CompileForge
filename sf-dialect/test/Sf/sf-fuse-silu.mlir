// RUN: %sf-opt --sf-fuse-silu %s | FileCheck %s

//===----------------------------------------------------------------------===//
// sf-fuse-silu fusion: sf.silu + sf.mul → sf.fused_silu_mul
//===----------------------------------------------------------------------===//

module {
  func.func @fuse_silu(%gate: tensor<2x8xf32>, %up: tensor<2x8xf32>) -> tensor<2x8xf32> {
    %silu = sf.silu %gate : tensor<2x8xf32>
    %0 = sf.mul %silu, %up : tensor<2x8xf32>, tensor<2x8xf32> -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}

// CHECK:      func.func @fuse_silu
// CHECK:      sf.fused_silu_mul
// CHECK-NOT:  sf.silu
// CHECK-NOT:  sf.mul
