#ifndef SF_PASSES
#define SF_PASSES

include "mlir/Pass/PassBase.td"

def SfChainWrapper : Pass<"sf-chain-wrapper", "mlir::ModuleOp"> {
  let summary = "Generate a main() wrapper that chains main_0→main_1→...→main_N";
  let description = [{
    Runs after sf-lower-to-linalg (tensor level, before bufferize).
    Generates a `main` function that:
    1. Calls main_0 with all inputs → gets hidden + weight identity copies
    2. For each layer function main_{i} (i>0), calls it with the previous
       function's hidden output + the appropriate weight copies from main_0.
    3. Returns the final function's output.

    Weight partitioning: main_0's weight returns are partitioned across
    layer functions in order. Each layer function's weight arg count
    determines how many weights it consumes. The wrapper tracks a running
    offset into main_0's weight returns.

    This makes the compiled dylib self-contained: callers only need a
    single `_mlir_ciface_main` entry point.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];
}

def SfPromoteWeights : Pass<"sf-promote-weights", "mlir::ModuleOp"> {
  let summary = "Promote sf.weight and sf.constant ops to func arguments";
  let description = [{
    Pre-processing pass that converts `sf.weight` and `sf.constant` ops into
    function arguments. Must run before `sf-lower-to-linalg`. The Rust runtime
    provides these tensors via the function call interface.

    Sets `sf.weight_names` on main_0 with the ordered list of all promoted
    weight names.
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

//===----------------------------------------------------------------------===//
// Verification passes
//===----------------------------------------------------------------------===//

def SfaContractVerify : Pass<"sfa-contract-verify", "mlir::ModuleOp"> {
  let summary = "Verify lowered IR satisfies SFA ABI constraints";
  let description = [{
    Post-lowering verification pass that checks the IR satisfies SFA ABI
    constraints. Validates that all func.func inputs/outputs are MemRefType
    (or RankedTensorType) with rank 1-4 (SfaMemRef compatible), verifies
    no residual sf.weight/constant ops remain, and warns if
    llvm.emit_c_interface attribute is missing.

    Records per-function input semantics as module-level attributes
    (sfa.func_metadata) for downstream ABI generation.

    This pass serves as the CONTRACT boundary between sf-dialect lowering
    and the compiler ABI generation.
  }];
  let dependentDialects = [
    "mlir::func::FuncDialect",
  ];
}

#endif // SF_PASSES
