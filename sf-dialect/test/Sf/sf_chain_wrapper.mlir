// RUN: %sf-opt --sf-chain-wrapper %s | FileCheck %s

module attributes {sf.chain_order = ["main_0", "main_1", "main_2"]} {
  func.func @main_0(%arg0: tensor<2x4xi64>, %arg1: tensor<8x64xf32>,
                      %arg2: tensor<64x64xf32>, %arg3: tensor<64xf32>)
      -> (tensor<2x4x64xf32>, tensor<64x64xf32>, tensor<64xf32>)
      attributes {sf.weight_names = ["tok_embed_weight", "q_proj_weight",
                                      "ln_weight"]} {
    %emb = "sf.embedding"(%arg1, %arg0) {num_buckets = 8 : i64}
        : (tensor<8x64xf32>, tensor<2x4xi64>) -> tensor<2x4x64xf32>
    return %emb, %arg2, %arg3
        : tensor<2x4x64xf32>, tensor<64x64xf32>, tensor<64xf32>
  }
  func.func @main_1(%arg0: tensor<2x4x64xf32>, %arg1: tensor<64x64xf32>)
      -> tensor<2x4x64xf32>
      attributes {sf.weight_names = ["q_proj_weight"]} {
    %0 = "sf.linear"(%arg0, %arg1) {use_bias = false}
        : (tensor<2x4x64xf32>, tensor<64x64xf32>) -> tensor<2x4x64xf32>
    return %0 : tensor<2x4x64xf32>
  }
  func.func @main_2(%arg0: tensor<2x4x64xf32>, %arg1: tensor<64xf32>)
      -> tensor<2x4x64xf32>
      attributes {sf.weight_names = ["ln_weight"]} {
    %0 = "sf.layer_norm"(%arg0, %arg1, %arg1) {axis = 2 : i64, eps = 1.000000e-05 : f64}
        : (tensor<2x4x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<2x4x64xf32>
    return %0 : tensor<2x4x64xf32>
  }
}

// CHECK:      func.func private @main
// CHECK:        call @main_0(
// CHECK:        call @main_1(
// CHECK-SAME:     %{{.+}}#0
// CHECK-SAME:     %{{.+}}#1
// CHECK:        call @main_2(
// CHECK-SAME:     %{{.+}}#2
// CHECK:        return
