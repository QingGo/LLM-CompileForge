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
    `tensor`, `scf`, and `func` dialect operations. Uses the GreedyPatternRewriteDriver (applyPatternsGreedily) for progressive lowering.
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

def SfStripGEPNuw : Pass<"sf-strip-gep-nuw", "mlir::ModuleOp"> {
  let summary = "Strip nuw flag from llvm.getelementptr operations";
  let description = [{
    LLVM 22's MLIR LLVM dialect may emit a `nuw` (no unsigned wrap) flag on
    `llvm.getelementptr` operations. Older LLVM versions (e.g. LLVM 20 used
    by mlir-translate in this project) cannot parse this flag. This pass
    strips the `nuw` flag while preserving all other no-wrap flags.

    This replaces a regex-based fixup in `_fixup_mlir_for_translate` that
    operated on the MLIR text after lowering.
  }];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
  ];
}

#endif // SF_PASSES
