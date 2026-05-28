//===- SfFusionPasses.cpp - C++ fusion passes for sf dialect ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements 4 fusion passes that compose sf dialect ops into fused variants:
//
//   1. sf-fuse-silu:     sf.silu + sf.mul → sf.fused_silu_mul
//   2. sf-fuse-rms-norm: sf.rms_norm + sf.mul + sf.matmul → sf.fused_rms_norm_matmul
//   3. sf-fuse-qkv:      3× sf.linear with same input → sf.fused_qkv
//   4. sf-fuse-attention: SDPA + transpose + view + linear → sf.fused_attention_output
//
// Passes 1, 2, 4 use GreedyPatternRewriteDriver. Pass 3 uses a manual
// collect-then-replace strategy (3-op atomic replacement).
//
//===----------------------------------------------------------------------===//

#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"
#include "Sf/SfPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DEF_SFFUSESILUPASS
#define GEN_PASS_DEF_SFFUSERMSNORMPASS
#define GEN_PASS_DEF_SFFUSEQKVPASS
#define GEN_PASS_DEF_SFFUSEATTENTIONPASS
#include "Sf/SfPasses.h.inc"

#define DEBUG_TYPE "sf-fusion"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass 1: FuseSiLU — sf.silu(x) + sf.mul(y) → sf.fused_silu_mul(x, y)
//===----------------------------------------------------------------------===//
// Matches sf.mul ops where one operand is produced by sf.silu. Replaces
// both ops with a single sf.fused_silu_mul(gate, up) where gate is the
// silu's input and up is the other mul operand.
//===----------------------------------------------------------------------===//

namespace {

struct SiluMulFusePattern : public OpRewritePattern<sf::MulOp> {
  using OpRewritePattern<sf::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sf::MulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs(), rhs = op.getRhs();

    // Check if either operand comes from a sf.silu op
    auto siluOp = lhs.getDefiningOp<sf::SiluOp>();
    if (!siluOp)
      siluOp = rhs.getDefiningOp<sf::SiluOp>();
    if (!siluOp)
      return rewriter.notifyMatchFailure(op, "no silu operand found");

    // gate = silu's input, up = the other mul operand
    Value gate = siluOp.getInput();
    Value up = (siluOp.getResult() == lhs) ? rhs : lhs;

    auto fusedOp = rewriter.create<sf::FusedSiluMulOp>(
        op.getLoc(), op.getResult().getType(), gate, up);
    rewriter.replaceOp(op, fusedOp.getOutput());

    LLVM_DEBUG(llvm::dbgs() << "[sf-fuse-silu] fused silu+mul\n");
    return success();
  }
};

} // namespace

namespace {

struct SfFuseSiluPass
    : public ::impl::SfFuseSiluPassBase<SfFuseSiluPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfFuseSiluPass)

  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<SiluMulFusePattern>(&getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      func.emitError("sf-fuse-silu did not converge");
      return signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "[sf-fuse-silu] pass complete\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass 2: FuseRMSNorm — sf.rms_norm + sf.mul + sf.matmul →
//                       sf.fused_rms_norm_matmul
//===----------------------------------------------------------------------===//
// Matches sf.matmul (or sf.linear) whose first operand is produced by
// sf.mul whose first operand is produced by sf.rms_norm. Replaces the
// chain with sf.fused_rms_norm_matmul(rms_input, rms_weight, matmul_weight).
//===----------------------------------------------------------------------===//

namespace {

struct RmsNormMatmulFusePattern : public OpRewritePattern<sf::MatmulOp> {
  using OpRewritePattern<sf::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sf::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // matmul's first operand should be produced by sf.mul
    auto mulOp = op.getLhs().getDefiningOp<sf::MulOp>();
    if (!mulOp)
      return rewriter.notifyMatchFailure(op, "lhs not from mul");

    // mul's first operand should be produced by sf.rms_norm
    auto rmsOp = mulOp.getLhs().getDefiningOp<sf::RmsNormOp>();
    if (!rmsOp)
      return rewriter.notifyMatchFailure(op, "mul lhs not from rms_norm");

    // fused_rms_norm_matmul(rms_input, rms_weight, matmul_weight)
    auto fusedOp = rewriter.create<sf::FusedRmsNormMatmulOp>(
        op.getLoc(), op.getResult().getType(),
        rmsOp.getInput(),        // rms input (x)
        mulOp.getRhs(),          // rms weight (second operand of mul)
        op.getRhs());            // matmul weight

    rewriter.replaceOp(op, fusedOp.getOutput());

    LLVM_DEBUG(llvm::dbgs() << "[sf-fuse-rms-norm] fused rms_norm+mul+matmul\n");
    return success();
  }
};

} // namespace

// Also match on sf.linear for the same pattern (linear output projection
// after rms_norm in some model configurations).
namespace {

struct RmsNormLinearFusePattern : public OpRewritePattern<sf::LinearOp> {
  using OpRewritePattern<sf::LinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sf::LinearOp op,
                                PatternRewriter &rewriter) const override {
    auto mulOp = op.getInput().getDefiningOp<sf::MulOp>();
    if (!mulOp)
      return rewriter.notifyMatchFailure(op, "input not from mul");

    auto rmsOp = mulOp.getLhs().getDefiningOp<sf::RmsNormOp>();
    if (!rmsOp)
      return rewriter.notifyMatchFailure(op, "mul lhs not from rms_norm");

    auto fusedOp = rewriter.create<sf::FusedRmsNormMatmulOp>(
        op.getLoc(), op.getResult().getType(),
        rmsOp.getInput(),        // rms input
        mulOp.getRhs(),          // rms weight
        op.getWeight());         // linear weight

    rewriter.replaceOp(op, fusedOp.getOutput());

    LLVM_DEBUG(llvm::dbgs() << "[sf-fuse-rms-norm] fused rms_norm+mul+linear\n");
    return success();
  }
};

} // namespace

namespace {

struct SfFuseRmsNormPass
    : public ::impl::SfFuseRmsNormPassBase<SfFuseRmsNormPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfFuseRmsNormPass)

  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<RmsNormMatmulFusePattern, RmsNormLinearFusePattern>(
        &getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      func.emitError("sf-fuse-rms-norm did not converge");
      return signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "[sf-fuse-rms-norm] pass complete\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass 3: FuseQKV — 3× sf.linear with same input → sf.fused_qkv
//===----------------------------------------------------------------------===//
// Uses a manual collect-then-replace strategy. Scans all sf.linear and
// sf.matmul ops in the function, groups them by their input tensor, and
// replaces groups of 3+ into sf.fused_qkv ops.
//===----------------------------------------------------------------------===//

namespace {

struct SfFuseQKVPass : public ::impl::SfFuseQKVPassBase<SfFuseQKVPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfFuseQKVPass)

  void runOnOperation() override {
    auto func = getOperation();

    // Collect all linear/matmul ops grouped by their first operand (input).
    DenseMap<Value, SmallVector<Operation *>> inputToOps;
    func.walk([&](Operation *op) {
      Value input;
      if (auto lin = dyn_cast<sf::LinearOp>(op)) {
        input = lin.getInput();
      } else if (auto mat = dyn_cast<sf::MatmulOp>(op)) {
        input = mat.getLhs();
      } else {
        return;
      }
      inputToOps[input].push_back(op);
    });

    // Process each group. When 3+ ops share the same input, fuse the first 3.
    int fusedCount = 0;
    for (auto &[input, ops] : inputToOps) {
      if (ops.size() < 3)
        continue;

      // Weights for Q, K, V (first 3 ops in insertion order)
      auto getWeight = [](Operation *op) -> Value {
        if (auto lin = dyn_cast<sf::LinearOp>(op))
          return lin.getWeight();
        if (auto mat = dyn_cast<sf::MatmulOp>(op))
          return mat.getRhs();
        return Value();
      };

      Value qW = getWeight(ops[0]);
      Value kW = getWeight(ops[1]);
      Value vW = getWeight(ops[2]);
      if (!qW || !kW || !vW) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[sf-fuse-qkv] skipping group: missing weight\n");
        continue;
      }

      // Broadcast result types to the output projection types
      Type qType = ops[0]->getResult(0).getType();
      Type kType = ops[1]->getResult(0).getType();
      Type vType = ops[2]->getResult(0).getType();

      OpBuilder builder(func.getContext());
      builder.setInsertionPoint(ops[0]);

      auto fusedOp = builder.create<sf::FusedQKVOp>(
          ops[0]->getLoc(), qType, kType, vType, input, qW, kW, vW);

      // RAUW for all 3 ops
      ops[0]->getResult(0).replaceAllUsesWith(fusedOp.getQ());
      ops[1]->getResult(0).replaceAllUsesWith(fusedOp.getK());
      ops[2]->getResult(0).replaceAllUsesWith(fusedOp.getV());

      // Erase original ops
      ops[0]->erase();
      ops[1]->erase();
      ops[2]->erase();

      fusedCount++;
    }

    if (fusedCount > 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[sf-fuse-qkv] fused " << fusedCount << " QKV groups\n");
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass 4: FuseAttention — SDPA + transpose + view + linear →
//                         sf.fused_attention_output
//===----------------------------------------------------------------------===//
// Matches sf.linear (or sf.matmul) whose input flows backward through
// sf.view → sf.transpose/sf.permute → sf.scaled_dot_product_attention.
// Replaces the chain with sf.fused_attention_output.
//===----------------------------------------------------------------------===//

namespace {

struct AttentionFusePattern : public OpRewritePattern<sf::LinearOp> {
  using OpRewritePattern<sf::LinearOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sf::LinearOp op,
                                PatternRewriter &rewriter) const override {
    // Walk backward from the output projection linear op:
    // linear(input, weight) where input comes from view
    Value linearInput = op.getInput();
    auto viewOp = linearInput.getDefiningOp<sf::ViewOp>();
    if (!viewOp)
      return rewriter.notifyMatchFailure(op, "linear input not from view");

    // view(input) where input comes from transpose/permute
    Operation *viewInputOp = viewOp.getInput().getDefiningOp();
    if (!viewInputOp)
      return rewriter.notifyMatchFailure(op, "view input has no defining op");
    if (!isa<sf::TransposeOp, sf::PermuteOp>(viewInputOp))
      return rewriter.notifyMatchFailure(op,
                                         "view input is not transpose/permute");

    // transpose/permute(input) where input comes from SDPA
    Value transInput = viewInputOp->getOperand(0);
    auto sdpaOp = transInput.getDefiningOp<sf::ScaledDotProductAttentionOp>();
    if (!sdpaOp)
      return rewriter.notifyMatchFailure(op,
                                         "transpose input not from SDPA");

    // Build fused_attention_output operands:
    // query, key, value, [attn_mask], o_weight
    Value attnMask = sdpaOp.getAttnMask(); // may be null (Value())

    auto fusedOp = rewriter.create<sf::FusedAttentionOutputOp>(
        op.getLoc(), op.getResult().getType(),
        sdpaOp.getQuery(), sdpaOp.getKey(), sdpaOp.getValue(),
        attnMask, op.getWeight());

    rewriter.replaceOp(op, fusedOp.getOutput());

    LLVM_DEBUG(llvm::dbgs() << "[sf-fuse-attention] fused SDPA+output proj\n");
    return success();
  }
};

} // namespace

// Also match on sf.matmul for the attention output projection (some model
// configurations use matmul directly instead of linear).
namespace {

struct AttentionMatmulFusePattern : public OpRewritePattern<sf::MatmulOp> {
  using OpRewritePattern<sf::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sf::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value linearInput = op.getLhs();
    auto viewOp = linearInput.getDefiningOp<sf::ViewOp>();
    if (!viewOp)
      return rewriter.notifyMatchFailure(op, "matmul lhs not from view");

    Operation *viewInputOp = viewOp.getInput().getDefiningOp();
    if (!viewInputOp)
      return rewriter.notifyMatchFailure(op, "view input has no defining op");
    if (!isa<sf::TransposeOp, sf::PermuteOp>(viewInputOp))
      return rewriter.notifyMatchFailure(op,
                                         "view input is not transpose/permute");

    Value transInput = viewInputOp->getOperand(0);
    auto sdpaOp = transInput.getDefiningOp<sf::ScaledDotProductAttentionOp>();
    if (!sdpaOp)
      return rewriter.notifyMatchFailure(op,
                                         "transpose input not from SDPA");

    Value attnMask = sdpaOp.getAttnMask();

    auto fusedOp = rewriter.create<sf::FusedAttentionOutputOp>(
        op.getLoc(), op.getResult().getType(),
        sdpaOp.getQuery(), sdpaOp.getKey(), sdpaOp.getValue(),
        attnMask, op.getRhs());

    rewriter.replaceOp(op, fusedOp.getOutput());

    LLVM_DEBUG(llvm::dbgs() << "[sf-fuse-attention] fused SDPA+matmul out\n");
    return success();
  }
};

} // namespace

namespace {

struct SfFuseAttentionPass
    : public ::impl::SfFuseAttentionPassBase<SfFuseAttentionPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfFuseAttentionPass)

  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<AttentionFusePattern, AttentionMatmulFusePattern>(
        &getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      func.emitError("sf-fuse-attention did not converge");
      return signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "[sf-fuse-attention] pass complete\n");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass creation entry points
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> mlir::sf::createSfFuseSiluPass() {
  return std::make_unique<SfFuseSiluPass>();
}

std::unique_ptr<Pass> mlir::sf::createSfFuseRmsNormPass() {
  return std::make_unique<SfFuseRmsNormPass>();
}

std::unique_ptr<Pass> mlir::sf::createSfFuseQKVPass() {
  return std::make_unique<SfFuseQKVPass>();
}

std::unique_ptr<Pass> mlir::sf::createSfFuseAttentionPass() {
  return std::make_unique<SfFuseAttentionPass>();
}
