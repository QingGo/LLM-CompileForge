#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace mlir::sf {

//===----------------------------------------------------------------------===//
// Decompose sf.fused_silu_mul(gate, up) → sf.silu(gate) → sf.mul(silu, up)
//===----------------------------------------------------------------------===//

struct FusedSiluMulDecompose : public OpRewritePattern<sf::FusedSiluMulOp> {
  using OpRewritePattern<sf::FusedSiluMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::FusedSiluMulOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto gate = op.getGate();
    auto up = op.getUp();
    auto resultType = cast<RankedTensorType>(op.getOutput().getType());

    // sf.silu(gate)
    auto siluType = cast<RankedTensorType>(gate.getType());
    auto siluOp = rewriter.create<sf::SiluOp>(loc, siluType, gate);
    // sf.mul(silu, up)
    auto mulOp = rewriter.create<sf::MulOp>(loc, resultType,
                                            siluOp.getResult(), up);

    rewriter.replaceOp(op, mulOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Decompose sf.fused_rms_norm_matmul → sf.rms_norm + sf.matmul
//===----------------------------------------------------------------------===//

struct FusedRmsNormMatmulDecompose
    : public OpRewritePattern<sf::FusedRmsNormMatmulOp> {
  using OpRewritePattern<sf::FusedRmsNormMatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::FusedRmsNormMatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto rmsWeight = op.getRmsWeight();
    auto matmulWeight = op.getMatmulWeight();
    auto resultType = cast<RankedTensorType>(op.getOutput().getType());

    // sf.rms_norm(input, rms_weight) — normed shape matches input
    auto inputType = cast<RankedTensorType>(input.getType());
    auto normType = RankedTensorType::get(inputType.getShape(),
                                          inputType.getElementType());
    auto normOp = rewriter.create<sf::RmsNormOp>(loc, normType, input,
                                                  rmsWeight);
    // sf.matmul(normed, matmul_weight)
    auto matmulOp = rewriter.create<sf::MatmulOp>(loc, resultType,
                                                   normOp.getResult(),
                                                   matmulWeight);

    rewriter.replaceOp(op, matmulOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Decompose sf.fused_qkv → 3× sf.matmul
//===----------------------------------------------------------------------===//

struct FusedQKVDecompose : public OpRewritePattern<sf::FusedQKVOp> {
  using OpRewritePattern<sf::FusedQKVOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::FusedQKVOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Value qW = op.getQWeight(), kW = op.getKWeight(), vW = op.getVWeight();
    auto qType = cast<RankedTensorType>(op.getQ().getType());
    auto kType = cast<RankedTensorType>(op.getK().getType());
    auto vType = cast<RankedTensorType>(op.getV().getType());

    Value q = rewriter.create<sf::MatmulOp>(loc, qType, input, qW).getResult();
    Value k = rewriter.create<sf::MatmulOp>(loc, kType, input, kW).getResult();
    Value v = rewriter.create<sf::MatmulOp>(loc, vType, input, vW).getResult();

    rewriter.replaceOp(op, {q, k, v});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Decompose sf.fused_attention_output → sf.scaled_dot_product_attention
//                                         + sf.matmul
//===----------------------------------------------------------------------===//

struct FusedAttentionOutputDecompose
    : public OpRewritePattern<sf::FusedAttentionOutputOp> {
  using OpRewritePattern<sf::FusedAttentionOutputOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::FusedAttentionOutputOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto query = op.getQuery();
    auto key = op.getKey();
    auto value = op.getValue();
    Value attnMask = op.getAttnMask();
    auto oWeight = op.getOWeight();
    auto resultType = cast<RankedTensorType>(op.getOutput().getType());

    // SDPA result type from query shape
    auto sdpaType = cast<RankedTensorType>(query.getType());

    // sf.scaled_dot_product_attention(query, key, value, attn_mask)
    auto sdpaOp = rewriter.create<sf::ScaledDotProductAttentionOp>(
        loc, sdpaType, query, key, value, attnMask);

    // Output projection: sf.matmul
    auto matmulOp = rewriter.create<sf::MatmulOp>(loc, resultType,
                                                   sdpaOp.getResult(), oWeight);

    rewriter.replaceOp(op, matmulOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Decompose sf.fused_attention_block → rms_norm → 3× matmul → SDPA → matmul
//===----------------------------------------------------------------------===//

struct FusedAttentionBlockDecompose
    : public OpRewritePattern<sf::FusedAttentionBlockOp> {
  using OpRewritePattern<sf::FusedAttentionBlockOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::FusedAttentionBlockOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Value normWeight = op.getNormWeight();
    Value qW = op.getQWeight(), kW = op.getKWeight(), vW = op.getVWeight();
    Value oW = op.getOWeight();
    auto resultType = cast<RankedTensorType>(op.getOutput().getType());

    // Step 1: Pre-attention norm (RMS Norm)
    auto inputType = cast<RankedTensorType>(input.getType());
    auto normType = RankedTensorType::get(inputType.getShape(),
                                          inputType.getElementType());
    auto normOp = rewriter.create<sf::RmsNormOp>(loc, normType, input,
                                                  normWeight);
    Value normed = normOp.getResult();

    // Step 2: QKV projections (sf.matmul)
    auto qWType = cast<RankedTensorType>(qW.getType());
    auto kWType = cast<RankedTensorType>(kW.getType());
    auto vWType = cast<RankedTensorType>(vW.getType());

    SmallVector<int64_t> qShape(inputType.getShape().begin(),
                                inputType.getShape().end());
    if (!qShape.empty() && qWType.getRank() > 0)
      qShape.back() = qWType.getDimSize(qWType.getRank() - 1);
    auto qType = RankedTensorType::get(qShape, inputType.getElementType());

    SmallVector<int64_t> kShape(inputType.getShape().begin(),
                                inputType.getShape().end());
    if (!kShape.empty() && kWType.getRank() > 0)
      kShape.back() = kWType.getDimSize(kWType.getRank() - 1);
    auto kType = RankedTensorType::get(kShape, inputType.getElementType());

    SmallVector<int64_t> vShape(inputType.getShape().begin(),
                                inputType.getShape().end());
    if (!vShape.empty() && vWType.getRank() > 0)
      vShape.back() = vWType.getDimSize(vWType.getRank() - 1);
    auto vType = RankedTensorType::get(vShape, inputType.getElementType());

    Value q = rewriter.create<sf::MatmulOp>(loc, qType, normed, qW).getResult();
    Value k = rewriter.create<sf::MatmulOp>(loc, kType, normed, kW).getResult();
    Value v = rewriter.create<sf::MatmulOp>(loc, vType, normed, vW).getResult();

    // Step 3: Scaled dot-product attention
    auto sdpaType = RankedTensorType::get(qShape, inputType.getElementType());
    auto sdpaOp = rewriter.create<sf::ScaledDotProductAttentionOp>(
        loc, sdpaType, q, k, v, Value());

    // Step 4: Output projection
    auto matmulOp = rewriter.create<sf::MatmulOp>(loc, resultType,
                                                   sdpaOp.getResult(), oW);

    rewriter.replaceOp(op, matmulOp.getResult());
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerFusedPatterns(RewritePatternSet &patterns) {
  patterns.add<FusedSiluMulDecompose,
               FusedRmsNormMatmulDecompose,
               FusedQKVDecompose,
               FusedAttentionOutputDecompose,
               FusedAttentionBlockDecompose>(patterns.getContext());
}
} // namespace mlir::sf
