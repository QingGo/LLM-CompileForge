#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace mlir::sf {

//===----------------------------------------------------------------------===//
// Scaled Dot-Product Attention lowering:
//   softmax(Q * K^T / sqrt(d_k)) * V
//   Step 1: K^T = transpose(K)
//   Step 2: scores = matmul(Q, K^T)
//   Step 3: scores_scaled = scores * (1/sqrt(d_k))
//   Step 4: attn = softmax(scores_scaled) along last dim
//   Step 5: output = matmul(attn, V)
//===----------------------------------------------------------------------===//

struct SfScaledDotProductAttentionOpLowering
    : public OpRewritePattern<sf::ScaledDotProductAttentionOp> {
  using OpRewritePattern<sf::ScaledDotProductAttentionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::ScaledDotProductAttentionOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value Q = op.getQuery();
    Value K = op.getKey();
    Value V = op.getValue();

    auto qType = ::mlir::dyn_cast<::mlir::RankedTensorType>(Q.getType());
    if (!qType || qType.getRank() < 3) return failure();

    auto eltType = qType.getElementType();
    int64_t rank = qType.getRank();
  int64_t dk = qType.getDimSize(rank - 1);
  if (dk < 0) return failure();
  // Use explicit scale attribute if present; otherwise default to 1/sqrt(d_k)
  float scaleVal = 1.0f;
  if (auto scaleAttr = op->getAttrOfType<mlir::FloatAttr>("scale"))
    scaleVal = scaleAttr.getValueAsDouble();
  else
    scaleVal = 1.0f / std::sqrt(static_cast<float>(dk));
  auto ctx = rewriter.getContext();

  // Phase 1: tiled online softmax decision
  // For seq_kv > 64 or dynamic, use tiled attention to avoid O(seq²) memory.
  // (Implementation placeholder — currently falls through to standard path.)
  auto kType = ::mlir::dyn_cast<::mlir::RankedTensorType>(K.getType());
  int64_t seqKVDim = kType ? kType.getDimSize(rank - 2) : ShapedType::kDynamic;
  if (seqKVDim == ShapedType::kDynamic || seqKVDim > 64) {
    // Tiled attention needed — emit scf.for with online softmax.
    // For Phase 1 this falls back to standard attention (which creates
    // O(seq²) intermediate buffers but produces correct results).
    // The full tiled implementation will be added in a follow-up.
  }

  // Step 1: transpose K (last two dims)
    SmallVector<int64_t> ktShape(qType.getShape());
    std::swap(ktShape[rank - 1], ktShape[rank - 2]);
    auto ktType = RankedTensorType::get(ktShape, eltType);
    // Helper: dynamic sizes for ktType (last dim = seq from K dim rank-2)
    auto ktDyn = [&]() -> SmallVector<Value> {
      SmallVector<Value> dyns;
      for (int64_t i = 0; i < rank; ++i) {
        if (!ktType.isDynamicDim(i)) continue;
        if (i == rank - 1)
          dyns.push_back(tensor::DimOp::create(rewriter, loc, K, rank - 2));
        else
          dyns.push_back(tensor::DimOp::create(rewriter, loc, K, i));
      }
      return dyns;
    };
    Value ktEmpty = tensor::EmptyOp::create(rewriter, loc, ktType, ktDyn());
    SmallVector<unsigned> ktPerm(rank);
    for (int64_t i = 0; i < rank; ++i) ktPerm[i] = i;
    std::swap(ktPerm[rank - 1], ktPerm[rank - 2]);
    SmallVector<utils::IteratorType> ktIterTypes(rank, utils::IteratorType::parallel);
    auto ktOp = linalg::GenericOp::create(rewriter, loc, ktType, K, ktEmpty,
        {AffineMap::getPermutationMap(ktPerm, rewriter.getContext()),
         AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())}, ktIterTypes);
    populateBody(ktOp, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      linalg::YieldOp::create(b, loc, args[0]);
    });
    Value Kt = ktOp.getResult(0);

    // Helper: scores-type shapes (last dim = k_seq from K dim rank-2)
    auto scoresDyn = [&](RankedTensorType type) -> SmallVector<Value> {
      SmallVector<Value> dyns;
      for (int64_t i = 0; i < rank; ++i) {
        if (!type.isDynamicDim(i)) continue;
        if (i == rank - 1)
          dyns.push_back(tensor::DimOp::create(rewriter, loc, K, rank - 2));
        else
          dyns.push_back(tensor::DimOp::create(rewriter, loc, Q, i));
      }
      return dyns;
    };

    // Step 2: scores = matmul(Q, K^T) via linalg.generic (supports any rank)
    SmallVector<int64_t> scoresShape(qType.getShape());
    scoresShape[rank - 1] = qType.getDimSize(rank - 2);
    auto scoresType = RankedTensorType::get(scoresShape, eltType);
    Value scoresInit = tensor::EmptyOp::create(rewriter, loc, scoresType, scoresDyn(scoresType));
    Value scoresZero = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, 0.0f));
    Value scoresEmpty = linalg::FillOp::create(rewriter, loc,
        ValueRange{scoresZero}, ValueRange{scoresInit}).getResult(0);
    Value scores;
    {
      // Generic batch-matmul: loop [b..h, m, n, k]: parallel(b..h), parallel(m), parallel(n), reduction(k)
      // Maps: Q->(b..h,m,k) Kt->(b..h,k,n) Out->(b..h,m,n)
      int64_t mDim = rank - 2, kRed = rank; // loop has rank+1 dims
      SmallVector<AffineExpr> lhsMaps, rhsMaps, outMaps;
      for (int64_t i = 0; i < rank - 2; ++i)
        { lhsMaps.push_back(getAffineDimExpr(i, ctx)); rhsMaps.push_back(getAffineDimExpr(i, ctx)); outMaps.push_back(getAffineDimExpr(i, ctx)); }
      lhsMaps.push_back(getAffineDimExpr(mDim, ctx)); // M dim
      lhsMaps.push_back(getAffineDimExpr(kRed, ctx)); // K dim (contraction)
      rhsMaps.push_back(getAffineDimExpr(kRed, ctx)); // K dim
      rhsMaps.push_back(getAffineDimExpr(mDim + 1, ctx)); // N dim (= rank-1 in loop)
      outMaps.push_back(getAffineDimExpr(mDim, ctx)); // M dim
      outMaps.push_back(getAffineDimExpr(mDim + 1, ctx)); // N dim
      int64_t loopRank = rank + 1;
      SmallVector<utils::IteratorType> matIter(loopRank, utils::IteratorType::parallel);
      matIter[kRed] = utils::IteratorType::reduction;
      auto scoreGeneric = linalg::GenericOp::create(rewriter, loc, scoresType,
          ValueRange{Q, Kt}, ValueRange{scoresEmpty},
          {AffineMap::get(loopRank, 0, lhsMaps, ctx),
           AffineMap::get(loopRank, 0, rhsMaps, ctx),
           AffineMap::get(loopRank, 0, outMaps, ctx)}, matIter,
          [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
            Value _mul = arith::MulFOp::create(b, bodyLoc, args[0], args[1]);
            Value _add = arith::AddFOp::create(b, bodyLoc, args[2], _mul);
            linalg::YieldOp::create(b, bodyLoc, _add);
          });
      scores = scoreGeneric.getResult(0);
    }

    // Step 3: scores_scaled = scores * (1/sqrt(d_k))
    Value scaleConst = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, scaleVal));
    Value scaledEmpty = tensor::EmptyOp::create(rewriter, loc, scoresType, scoresDyn(scoresType));
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto scaleOp = linalg::GenericOp::create(rewriter, loc, scoresType,
        scores, scaledEmpty,
        identityMaps(rank, 2, rewriter.getContext()), iterTypes);
    populateBody(scaleOp, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value _scaled = arith::MulFOp::create(b, loc, args[0], scaleConst); linalg::YieldOp::create(b, loc, _scaled);
    });
    Value scoresScaled = scaleOp.getResult(0);

    // Step 3b: Apply attention mask if present
    // mask is boolean (0.0 = masked, 1.0 = attend) from sf.logical_and.
    // Causal condition: mask > 0.5 → attend (0.0 additive), else → -inf.
    if (Value mask = op.getAttnMask()) {
      Value zeroF32 = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, 0.0f));
      Value negLarge = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, -1.0e20f));

      // Step 3b-ii: additive[i,j] = (mask[i,j] > 0.5) ? 0.0 : -inf
      // mask is boolean (0.0 = masked, 1.0 = attend). No transpose comparison needed.
      Value additive = makeBinaryOp(rewriter, loc, mask, mask, scoresScaled.getType(), rewriter,
          [&](OpBuilder &b, Location loc, Value m, Value mT) {
            Value half = arith::ConstantOp::create(b, loc, b.getF32Type(),
                b.getFloatAttr(b.getF32Type(), 0.5f));
            Value cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, m, half);
            Value sel = arith::SelectOp::create(b, loc, cmp, zeroF32, negLarge);
            return sel;
          });
      // Step 3b-iii: scoresScaled = scoresScaled + additive
      scoresScaled = makeBinaryOp(rewriter, loc, scoresScaled, additive, scoresScaled.getType(), rewriter,
          [&](OpBuilder &b, Location loc, Value a, Value bVal) {
            return arith::AddFOp::create(b, loc, a, bVal);
          });
    }

    // Step 4: softmax along last dim
    // softmax(x) = exp(x - max) / sum(exp(x - max))
    int64_t lastDim = rank - 1;

    // Helper: max-type shapes (last dim = 1, static; leading dims from Q)
    auto maxDyn = [&](RankedTensorType type) -> SmallVector<Value> {
      SmallVector<Value> dyns;
      for (int64_t i = 0; i < rank; ++i) {
        if (!type.isDynamicDim(i)) continue;
        if (i == rank - 2)
          dyns.push_back(tensor::DimOp::create(rewriter, loc, Q, rank - 2));
        else
          dyns.push_back(tensor::DimOp::create(rewriter, loc, Q, i));
      }
      return dyns;
    };

    // 4a: max reduction along last dim
    SmallVector<int64_t> maxShape(scoresShape);
    maxShape[lastDim] = 1;
    auto maxType = RankedTensorType::get(maxShape, eltType);
    // Softmax max reduction: init to -inf
    Value negInf = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, -1.0e20f));
    Value maxEmpty = tensor::EmptyOp::create(rewriter, loc, maxType, maxDyn(maxType));
    auto fillMax = linalg::FillOp::create(rewriter, loc, ValueRange{negInf}, ValueRange{maxEmpty});
    maxEmpty = fillMax.getResult(0);
    SmallVector<utils::IteratorType> reduIters(rank);
    for (int64_t i = 0; i < rank; ++i)
      reduIters[i] = (i == lastDim) ? utils::IteratorType::reduction : utils::IteratorType::parallel;
    auto maxOp = linalg::GenericOp::create(rewriter, loc, maxType, scoresScaled, maxEmpty,
        {AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()),
         AffineMap::get(rank, 0,
             llvm::map_to_vector(llvm::seq<int64_t>(0, rank), [&](int64_t i) -> AffineExpr {
               return (i == lastDim) ? getAffineConstantExpr(0, rewriter.getContext())
                                     : getAffineDimExpr(i, rewriter.getContext());
             }), rewriter.getContext())}, reduIters);
    populateBody(maxOp, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value _mx = arith::MaxNumFOp::create(b, loc, args[0], args[1]); linalg::YieldOp::create(b, loc, _mx);
    });
    Value maxVal = maxOp.getResult(0);

    // 4b: sub = x - max (broadcast)
    Value sub = makeBinaryOp(rewriter, loc, scoresScaled, maxVal, scoresType, rewriter,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::SubFOp::create(b, loc, a, bVal);
        });

    // 4c: exp(x - max)
    Value expVal = makeUnaryOp(rewriter, loc, sub, scoresType, rewriter,
        [&](OpBuilder &b, Location loc, Value v) {
          return math::ExpOp::create(b, loc, v);
        });

    // 4d: sum reduction
    Value sumInit = tensor::EmptyOp::create(rewriter, loc, maxType, maxDyn(maxType));
    Value sumZero = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, 0.0f));
    Value sumEmpty = linalg::FillOp::create(rewriter, loc,
        ValueRange{sumZero}, ValueRange{sumInit}).getResult(0);
    auto sumOp = linalg::GenericOp::create(rewriter, loc, maxType, expVal, sumEmpty,
        {AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()),
         AffineMap::get(rank, 0,
             llvm::map_to_vector(llvm::seq<int64_t>(0, rank), [&](int64_t i) -> AffineExpr {
               return (i == lastDim) ? getAffineConstantExpr(0, rewriter.getContext())
                                     : getAffineDimExpr(i, rewriter.getContext());
             }), rewriter.getContext())}, reduIters);
    populateBody(sumOp, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value _ad = arith::AddFOp::create(b, loc, args[0], args[1]); linalg::YieldOp::create(b, loc, _ad);
    });
    Value sumVal = sumOp.getResult(0);

    // 4e: softmax = exp / sum (broadcast)
    Value attn = makeBinaryOp(rewriter, loc, expVal, sumVal, scoresType, rewriter,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::DivFOp::create(b, loc, a, bVal);
        });

    // Step 5: output = matmul(attn, V) via linalg.generic (supports any rank)
    SmallVector<int64_t> outShape(qType.getShape());
    outShape[rank - 1] = ::mlir::cast<::mlir::RankedTensorType>(V.getType()).getDimSize(rank - 1);
    auto outType = RankedTensorType::get(outShape, eltType);
    auto outEmptyType = RankedTensorType::get(outShape, eltType);
    SmallVector<Value> outDyn;
    for (int64_t i = 0; i < rank; ++i) {
      if (!outEmptyType.isDynamicDim(i)) continue;
      if (i == rank - 1)
        outDyn.push_back(tensor::DimOp::create(rewriter, loc, V, rank - 1));
      else if (i == rank - 2)
        outDyn.push_back(tensor::DimOp::create(rewriter, loc, Q, rank - 2));
      else
        outDyn.push_back(tensor::DimOp::create(rewriter, loc, Q, i));
    }
    Value outInit = tensor::EmptyOp::create(rewriter, loc, outEmptyType, outDyn);
    Value outZero = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, 0.0f));
    Value outEmpty = linalg::FillOp::create(rewriter, loc,
        ValueRange{outZero}, ValueRange{outInit}).getResult(0);
    Value attnVResult;
    {
      // Generic batch-matmul: loop [b..h, m, s, d]: parallel(b..h), parallel(m), reduction(s), parallel(d)
      // Maps: attn->(b..h,m,s) V->(b..h,s,d) Out->(b..h,m,d)
      int64_t redDim = rank - 1; // s dim in attn, rank-1 in V = contraction
      int64_t loopRank = rank + 1;
      SmallVector<AffineExpr> attnMaps, vMaps, outMaps;
      for (int64_t i = 0; i < rank - 2; ++i)
        { attnMaps.push_back(getAffineDimExpr(i, ctx)); vMaps.push_back(getAffineDimExpr(i, ctx)); outMaps.push_back(getAffineDimExpr(i, ctx)); }
      attnMaps.push_back(getAffineDimExpr(rank - 2, ctx)); // M dim
      attnMaps.push_back(getAffineDimExpr(redDim, ctx));   // S dim (contraction)
      vMaps.push_back(getAffineDimExpr(redDim, ctx));      // S dim (contraction, same loop dim as attn's S)
      vMaps.push_back(getAffineDimExpr(loopRank - 1, ctx)); // D dim (last in loop)
      outMaps.push_back(getAffineDimExpr(rank - 2, ctx));  // M dim
      outMaps.push_back(getAffineDimExpr(loopRank - 1, ctx)); // D dim
      SmallVector<utils::IteratorType> outMatIter(loopRank, utils::IteratorType::parallel);
      outMatIter[redDim] = utils::IteratorType::reduction;
      auto outGeneric = linalg::GenericOp::create(rewriter, loc, outType,
          ValueRange{attn, V}, ValueRange{outEmpty},
          {AffineMap::get(loopRank, 0, attnMaps, ctx),
           AffineMap::get(loopRank, 0, vMaps, ctx),
           AffineMap::get(loopRank, 0, outMaps, ctx)}, outMatIter,
          [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
            Value _mul = arith::MulFOp::create(b, bodyLoc, args[0], args[1]);
            Value _add = arith::AddFOp::create(b, bodyLoc, args[2], _mul);
            linalg::YieldOp::create(b, bodyLoc, _add);
          });
      attnVResult = outGeneric.getResult(0);
    }

    rewriter.replaceOp(op, attnVResult);
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerAttentionPatterns(RewritePatternSet &patterns) {
  patterns.add<SfScaledDotProductAttentionOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
