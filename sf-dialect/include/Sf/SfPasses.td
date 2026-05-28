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

//===----------------------------------------------------------------------===//
// Fusion passes
//===----------------------------------------------------------------------===//

def SfFuseSiluPass : Pass<"sf-fuse-silu", "mlir::func::FuncOp"> {
  let summary = "Fuse sf.silu + sf.mul into sf.fused_silu_mul";
  let description = [{
    Fuses a sf.silu(gate) followed by sf.mul(silu_out, up) into a single
    sf.fused_silu_mul(gate, up) operation. This enables fusion of the
    SiLU activation with the element-wise multiplication in the gated MLP.
  }];
  let dependentDialects = ["mlir::sf::SfDialect"];
}

def SfFuseRmsNormPass : Pass<"sf-fuse-rms-norm", "mlir::func::FuncOp"> {
  let summary = "Fuse sf.rms_norm + sf.mul + sf.matmul into sf.fused_rms_norm_matmul";
  let description = [{
    Fuses a chain of sf.rms_norm(x, w1) → sf.mul(normed, w2) → sf.matmul(mul_out, w3)
    into a single sf.fused_rms_norm_matmul(x, w2, w3). This eliminates the
    intermediate norm normalization tensor materialization.
  }];
  let dependentDialects = ["mlir::sf::SfDialect"];
}

def SfFuseQKVPass : Pass<"sf-fuse-qkv", "mlir::func::FuncOp"> {
  let summary = "Fuse 3× sf.linear with same input into sf.fused_qkv";
  let description = [{
    Fuses three sf.linear (or sf.matmul) operations that share the same
    input tensor into a single sf.fused_qkv operation. This is the QKV
    projection fusion used in attention blocks.
  }];
  let dependentDialects = ["mlir::sf::SfDialect"];
}

def SfFuseAttentionPass : Pass<"sf-fuse-attention", "mlir::func::FuncOp"> {
  let summary = "Fuse SDPA + transpose + view + linear into sf.fused_attention_output";
  let description = [{
    Fuses the chain sf.scaled_dot_product_attention → sf.transpose → sf.view
    → sf.linear (output projection) into a single sf.fused_attention_output
    operation. This reduces memory traffic by keeping the attention output
    in-register.
  }];
  let dependentDialects = ["mlir::sf::SfDialect"];
}

#endif // SF_PASSES
