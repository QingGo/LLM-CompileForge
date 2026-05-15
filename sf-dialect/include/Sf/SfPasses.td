#ifndef SF_PASSES
#define SF_PASSES

include "mlir/Pass/PassBase.td"

def SfPromoteWeights : Pass<"sf-promote-weights", "mlir::ModuleOp"> {
  let summary = "Promote sf.weight and sf.constant ops to func arguments";
  let description = [{
    Pre-processing pass that converts `sf.weight` and `sf.constant` ops into
    function arguments. Must run before `sf-lower-to-linalg`. The Rust runtime
    provides these tensors via the function call interface.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];
}

def SfLowerToLinalg : Pass<"sf-lower-to-linalg", "mlir::ModuleOp"> {
  let summary = "Lower sf dialect ops to linalg/arith/math/tensor/scf";
  let description = [{
    Converts all `sf.*` operations (excluding weight/constant which are
    pre-processed by sf-promote-weights) to `linalg`, `arith`, `math`,
    `tensor`, `scf`, and `func` dialect operations. Uses the DialectConversion
    framework for progressive lowering.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif // SF_PASSES
