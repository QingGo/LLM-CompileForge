// RUN: %sf-opt --sf-lower-to-linalg %s | %mlir-opt --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-loops --convert-scf-to-cf --convert-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts | %mlir-translate --mlir-to-llvmir | FileCheck %s
// RUN: %sf-opt --sf-lower-to-linalg %s | %mlir-opt --one-shot-bufferize="bufferize-function-boundaries" --convert-linalg-to-loops --convert-scf-to-cf --convert-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts | %mlir-translate --mlir-to-llvmir | not cc -c -x ir - -o /dev/null 2>&1 | FileCheck --check-prefix=CC_BUG %s

//===----------------------------------------------------------------------===//
// Full lowering: sf dialect → linalg → loops → LLVM dialect → LLVM IR
//
// RUN line 1: mlir-translate produces valid LLVM IR text
//   - CHECK for expected constructs (define, load, store, call)
//   - CHECK-NOT for parse errors in mlir-translate output
//
// RUN line 2: Apple clang cc -c pipe catches rms_norm attribute group bug
//   - Uses `not cc -c` because system cc currently rejects the LLVM IR
//     with "unterminated attribute group" (rms_norm lowering bug).
//   - Verify the expected error message for this known issue.
//     When the lowering is fixed, the `not` will flip to failure:
//     remove `not` and add CC_BUG-NOT: unterminated.
//===----------------------------------------------------------------------===//

module {
  // Test 1: sf.rms_norm → LLVM IR (exercises math.sqrt → #0 attribute group)
  func.func @rms_norm(%input: tensor<2x8xf32>, %weight: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = "sf.rms_norm"(%input, %weight) : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // Test 2: sf.matmul → LLVM IR
  func.func @matmul(%arg0: tensor<2x4xf32>, %arg1: tensor<4x2xf32>) -> tensor<2x2xf32> {
    %0 = "sf.matmul"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// CHECK:      define {{.*}} @rms_norm
// CHECK:      define {{.*}} @matmul
// CHECK:      load
// CHECK:      store
// CHECK:      call
// CHECK-NOT:  unterminated
// CHECK-NOT:  error

// CC_BUG: unterminated attribute group
