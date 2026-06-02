// REQUIRES: sf-opt
// RUN: not %sf-opt --sfa-contract-verify %s --split-input-file 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Test 1: test_valid_lowered
// Fully lowered IR with emit_c_interface — pass succeeds.
// Verifies no warnings or errors on valid post-lowering IR.
//===----------------------------------------------------------------------===//

module {
  func.func @valid_lowered(%arg0: tensor<2x4xf32>, %arg1: tensor<4x2xf32>) -> tensor<2x2xf32> attributes {llvm.emit_c_interface} {
    %init = tensor.empty() : tensor<2x2xf32>
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x4xf32>, tensor<4x2xf32>)
                        outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// -----
//===----------------------------------------------------------------------===//
// Test 3: test_missing_emit_c_interface
// Lowered IR missing emit_c_interface attribute — pass warns, not errors.
//===----------------------------------------------------------------------===//

module {
  func.func @missing_emit_c_interface(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = arith.addf %arg0, %arg0 : tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}

// -----
//===----------------------------------------------------------------------===//
// Test 4: test_rank_out_of_range
// IR with rank-5 memref — pass warns about unsupported rank.
//===----------------------------------------------------------------------===//

module {
  func.func @rank_out_of_range(%arg0: memref<2x3x4x5x6xf32>) {
    return
  }
}

// -----
//===----------------------------------------------------------------------===//
// Test 2: test_sf_weight_residual
// IR with unpromoted sf.weight op — pass fails with hard error.
// This section triggers overall pass failure (non-zero exit, handled by `not`).
//===----------------------------------------------------------------------===//

module {
  func.func @sf_weight_residual(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "sf.weight"() {name = "w1", type = tensor<4x8xf32>} : () -> tensor<4x8xf32>
    return %arg0 : tensor<2x4xf32>
  }
}

//===----------------------------------------------------------------------===//
// FileCheck patterns — stderr diagnostics appear first (sequential order),
// then stdout IR output follows.
//===----------------------------------------------------------------------===//

// Test 1: valid_lowered — no warning or error in this section
// CHECK:      [sfa-contract-verify] checking valid_lowered
// CHECK-NOT:  warning:
// CHECK-NOT:  error:
// CHECK:      [sfa-contract-verify] checked

// Test 3: missing_emit_c_interface — warning emitted
// CHECK:      [sfa-contract-verify] checking missing_emit_c_interface
// CHECK:      warning: {{.*}}missing llvm.emit_c_interface
// CHECK:      [sfa-contract-verify] checked

// Test 4: rank_out_of_range — warning about rank 5
// CHECK:      [sfa-contract-verify] checking rank_out_of_range
// CHECK:      warning: {{.*}}rank 5
// CHECK:      [sfa-contract-verify] checked

// Test 2: sf_weight_residual — hard error
// CHECK:      [sfa-contract-verify] checking sf_weight_residual
// CHECK:      error: {{.*}}unpromoted sf.weight
// CHECK:      [sfa-contract-verify] checked

// Stdout IR checks — verify that successful sections produce IR with metadata.
// Section 4 (sf_weight_residual) does NOT print IR due to pass failure.
// CHECK:      func.func @valid_lowered
// CHECK:      emit_c_interface
// CHECK:      func.func @missing_emit_c_interface
// CHECK:      func.func @rank_out_of_range
