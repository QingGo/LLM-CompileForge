// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// sf.scaled_dot_product_attention lowering
// Decomposes into: K^T transpose → Q*K^T matmul → scale → softmax → attn*V matmul
// Requires rank >= 3 with static d_k dimension.
//===----------------------------------------------------------------------===//

module {
  func.func @sdpa(%Q: tensor<1x4x8xf32>, %K: tensor<1x4x8xf32>, %V: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
    %0 = "sf.scaled_dot_product_attention"(%Q, %K, %V) : (tensor<1x4x8xf32>, tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
    return %0 : tensor<1x4x8xf32>
  }
}

// CHECK:      func.func @sdpa
// CHECK-DAG:  linalg.generic
// CHECK-DAG:  math.exp
// CHECK-DAG:  arith.addf
// CHECK-DAG:  arith.mulf
// CHECK-DAG:  arith.divf
// CHECK-DAG:  arith.subf
// CHECK-DAG:  arith.maxnumf
// CHECK-NOT:  "sf.scaled_dot_product_attention"
