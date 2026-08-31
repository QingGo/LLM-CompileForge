// RUN: %sf-opt --sf-lower-to-linalg %s | FileCheck %s

//===----------------------------------------------------------------------===//
// SDPA mask broadcast regression (2026-08-14).
//
// During KV-cache decode, Q has q_len=1 while K/V span k_len=7. The model
// computes the causal mask from the CURRENT input seq, so it is
// tensor<1x1x1x1xf32> at runtime while the scores are [1, H, 1, 7].
// linalg.generic affine maps are static, so reading mask[b,0,q,k] for
// k>0 reads out of bounds. The lowering must PAD the mask to the scores'
// [B,1,Q,K] shape with mask[0,0,0,0] (torch broadcast semantics) before
// applying it.
//===----------------------------------------------------------------------===//

module {
  func.func @sdpa_decode_mask(
      %Q: tensor<1x4x?x8xf32>,
      %K: tensor<1x4x?x8xf32>,
      %V: tensor<1x4x?x8xf32>,
      %mask: tensor<?x1x?x?xf32>)
      -> tensor<1x4x?x8xf32> {
    %0 = "sf.scaled_dot_product_attention"(%Q, %K, %V, %mask) :
      (tensor<1x4x?x8xf32>, tensor<1x4x?x8xf32>, tensor<1x4x?x8xf32>,
       tensor<?x1x?x?xf32>) -> tensor<1x4x?x8xf32>
    return %0 : tensor<1x4x?x8xf32>
  }
}

// CHECK:      func.func @sdpa_decode_mask
// CHECK:      tensor.extract
// CHECK:      tensor.pad
// CHECK:      linalg.generic
// CHECK-NOT:  "sf.scaled_dot_product_attention"
