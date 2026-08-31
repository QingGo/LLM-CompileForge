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
  bool hasScale = op->hasAttr("scale");
  if (dk < 0 && !hasScale) return failure();
  // Use explicit scale attribute if present; otherwise default to 1/sqrt(d_k)
  float scaleVal = 1.0f;
  if (hasScale)
    scaleVal = op->getAttrOfType<mlir::FloatAttr>("scale").getValueAsDouble();
  else
    scaleVal = 1.0f / std::sqrt(static_cast<float>(dk));
  auto ctx = rewriter.getContext();

  // Phase 1: tiled online softmax decision
  // For seq_kv > 64 or dynamic, use tiled attention to avoid O(seq²) memory.
  // (Implementation placeholder — currently falls through to standard path.)
  auto kType = ::mlir::dyn_cast<::mlir::RankedTensorType>(K.getType());
  auto vType = ::mlir::dyn_cast<::mlir::RankedTensorType>(V.getType());
  if (!kType || !vType) return failure();
  int64_t seqKVDim = kType.getDimSize(rank - 2);
  if (seqKVDim == ShapedType::kDynamic || seqKVDim > 64) {
    // Tiled attention needed — emit scf.for with online softmax.
    // For Phase 1 this falls back to standard attention (which creates
    // O(seq²) intermediate buffers but produces correct results).
    // The full tiled implementation will be added in a follow-up.
  }

  // Qwen full-attention export normally arrives here already in the
  // standard [B, H, S, D] layout after the model's own transpose.  Do not
  // run an additional Qwen-layout normalization here; doing so was
  // mis-detecting standard dynamic-head Q tensors as Qwen [B,S,H,D] and
  // produced invalid GQA-shaped transposes.
  // GQA support: the compiler may lower repeat_kv into
  // ``enable_gqa=true`` with native kv-head K/V.  Expand K/V by the static
  // repeat factor before the attention math while the cache contract keeps
  // consuming the native kv-head outputs.
  Value effectiveK = K;
  Value effectiveV = V;
  auto resultShaped = cast<RankedTensorType>(op.getResult().getType());
  bool enableGqaAttr = false;
  if (auto gqaAttr = op->getAttrOfType<BoolAttr>("enable_gqa"))
    enableGqaAttr = gqaAttr.getValue();
  if (rank == 4 && enableGqaAttr) {
      // Derive the GQA repeat factor from the static head ratio when both
      // head dims are known.  Dynamic model graphs (q/k head dims both `?`)
      // fall back to the current LLaMA 8-KV-head/32-Q-head contract.
      int64_t group = 4;
      int64_t kvHeads = kType.getDimSize(1);
      int64_t qHeads = resultShaped.getDimSize(1);
      if (qHeads == ShapedType::kDynamic) qHeads = qType.getDimSize(1);
      if (kvHeads > 0 && qHeads > 0 && qHeads % kvHeads == 0)
        group = qHeads / kvHeads;
      SmallVector<int64_t> repShape = {
          kType.getDimSize(0),
          kType.getDimSize(1),
          group,
          kType.getDimSize(2),
          kType.getDimSize(3),
      };
      auto repType = RankedTensorType::get(repShape, eltType);
      SmallVector<Value> repDyn;
      for (int64_t i = 0; i < 5; ++i) {
        if (!repType.isDynamicDim(i))
          continue;
        int64_t srcDim = (i == 2) ? 1 : (i > 2 ? i - 1 : i);
        repDyn.push_back(tensor::DimOp::create(rewriter, loc, K, srcDim));
      }
      auto expandTo5D = [&](Value input) -> Value {
        Value empty = tensor::EmptyOp::create(rewriter, loc, repType, repDyn);
        auto inMap = AffineMap::get(
            5, 0,
            {getAffineDimExpr(0, rewriter.getContext()),
             getAffineDimExpr(1, rewriter.getContext()),
             getAffineDimExpr(3, rewriter.getContext()),
             getAffineDimExpr(4, rewriter.getContext())},
            rewriter.getContext());
        auto outMap = AffineMap::getMultiDimIdentityMap(5, rewriter.getContext());
        SmallVector<utils::IteratorType> iters(5, utils::IteratorType::parallel);
        auto g = linalg::GenericOp::create(rewriter, loc, repType, input, empty,
            {inMap, outMap}, iters);
        populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
          linalg::YieldOp::create(b, loc, args[0]);
        });
        auto kvExpandedType = qType;
        SmallVector<ReassociationIndices> reassoc = {
            ReassociationIndices{0}, ReassociationIndices{1, 2},
            ReassociationIndices{3}, ReassociationIndices{4}};
        return tensor::CollapseShapeOp::create(
            rewriter, loc, kvExpandedType, g.getResult(0), reassoc).getResult();
      };
      effectiveK = expandTo5D(K);
      effectiveV = expandTo5D(V);
  }

  // Step 1: transpose K (last two dims)
    auto effKType = cast<RankedTensorType>(effectiveK.getType());
    SmallVector<int64_t> ktShape(effKType.getShape());
    std::swap(ktShape[rank - 1], ktShape[rank - 2]);
    auto ktType = RankedTensorType::get(ktShape, eltType);
    // Helper: dynamic sizes for ktType (last dim = seq from K dim rank-2)
    auto ktDyn = [&]() -> SmallVector<Value> {
      SmallVector<Value> dyns;
      for (int64_t i = 0; i < rank; ++i) {
        if (!ktType.isDynamicDim(i)) continue;
        if (i == rank - 1)
          dyns.push_back(tensor::DimOp::create(rewriter, loc, effectiveK, rank - 2));
        else
          dyns.push_back(tensor::DimOp::create(rewriter, loc, effectiveK, i));
      }
      return dyns;
    };
    Value ktEmpty = tensor::EmptyOp::create(rewriter, loc, ktType, ktDyn());
    SmallVector<unsigned> ktPerm(rank);
    for (int64_t i = 0; i < rank; ++i) ktPerm[i] = i;
    std::swap(ktPerm[rank - 1], ktPerm[rank - 2]);
    SmallVector<utils::IteratorType> ktIterTypes(rank, utils::IteratorType::parallel);
    auto ktOp = linalg::GenericOp::create(rewriter, loc, ktType, effectiveK, ktEmpty,
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
          dyns.push_back(tensor::DimOp::create(rewriter, loc, effectiveK, rank - 2));
        else
          dyns.push_back(tensor::DimOp::create(rewriter, loc, Q, i));
      }
      return dyns;
    };

    // Step 2: scores = matmul(Q, K^T) via linalg.generic (supports any rank)
    // Prefer the declared SDPA result shape over Q's shape: in GQA graphs Q
    // may carry a dynamic head dim at the function boundary while the SDPA
    // result is statically head-expanded.
    SmallVector<int64_t> scoresShape(resultShaped.getShape());
    scoresShape[rank - 1] = resultShaped.getDimSize(rank - 2);
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

    // Step 3b: Apply the is_causal contract.  When ``is_causal=true`` and
    // there is no explicit boolean mask, synthesize a lower-triangular
    // additive mask directly from the score loop indices.  This is the
    // LLaMA no-mask path (HF SDPA passes ``is_causal`` positionally).
    auto hasBoolAttr = [&](StringRef name) -> bool {
      auto attr = op->getAttrOfType<BoolAttr>(name);
      return attr && attr.getValue();
    };
    if (hasBoolAttr("is_causal")) {
      auto causalType = cast<RankedTensorType>(scoresScaled.getType());
      Value causalEmpty =
          tensor::EmptyOp::create(rewriter, loc, causalType, scoresDyn(causalType));
      SmallVector<utils::IteratorType> causalIters(rank, utils::IteratorType::parallel);
      auto causalOp = linalg::GenericOp::create(
          rewriter, loc, causalType, ValueRange{}, ValueRange{causalEmpty},
          {AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())},
          causalIters);
      {
        auto guard = OpBuilder::InsertionGuard(rewriter);
        Block *causalBody = rewriter.createBlock(&causalOp.getRegion(), {});
        causalBody->addArgument(eltType, loc);
        rewriter.setInsertionPointToEnd(causalBody);
        Value qIdx = linalg::IndexOp::create(rewriter, loc, rank - 2);
        Value kvIdx = linalg::IndexOp::create(rewriter, loc, rank - 1);
        Value inCausal = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::sge, qIdx, kvIdx);
        Value causalKeep = arith::ConstantOp::create(
            rewriter, loc, eltType, rewriter.getFloatAttr(eltType, 0.0f));
        Value causalMasked = arith::ConstantOp::create(
            rewriter, loc, eltType, rewriter.getFloatAttr(eltType, -1.0e20f));
        Value causalVal = arith::SelectOp::create(
            rewriter, loc, inCausal, causalKeep, causalMasked);
        linalg::YieldOp::create(rewriter, loc, causalVal);
      }
      Value causalMask = causalOp.getResult(0);
      scoresScaled = makeBinaryOp(
          rewriter, loc, scoresScaled, causalMask, scoresScaled.getType(), rewriter,
          [&](OpBuilder &b, Location bodyLoc, Value a, Value bVal) {
            return arith::AddFOp::create(b, bodyLoc, a, bVal);
          });
    }

    // Step 3c: Apply attention mask if present
    if (Value mask = op.getAttnMask()) {
      auto rawMaskType = cast<RankedTensorType>(mask.getType());
      if (rawMaskType.getRank() >= 3) {
      Value zeroF32 = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, 0.0f));
      Value negLarge = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, -1.0e20f));

      auto scoresShaped = cast<RankedTensorType>(scoresScaled.getType());
      auto maskType = cast<RankedTensorType>(mask.getType());
      int64_t outRank = scoresShaped.getRank();
      int64_t maskRank = maskType.getRank();

      // ── Runtime-safe broadcast via tensor.pad ──────────────────────
      // linalg.generic affine maps are static, so a mask whose dynamic
      // seq dims are 1 at runtime (decode: [1,1,1,1] against scores
      // [1,H,1,K]) would be read out of bounds. Pad the mask to the
      // scores' [B,1,Q,K] shape with mask[0,0,0,0]: tensor.pad replicates
      // the extent-1 dims, which is exactly torch broadcast semantics.
      SmallVector<int64_t> padShape(scoresShaped.getShape().begin(),
                                    scoresShaped.getShape().end());
      padShape[1] = 1;  // head dim stays 1 (broadcast via static-1 map)
      auto maskEltType = cast<RankedTensorType>(mask.getType()).getElementType();
      auto paddedMaskType = RankedTensorType::get(padShape, maskEltType);

      Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto dimOf = [&](Value t, int64_t d) -> Value {
        return tensor::DimOp::create(rewriter, loc, t, d);
      };
      // high[i] = max(targetDim - sourceDim, 0)
      auto highFor = [&](Value target, int64_t td, Value source, int64_t sd) -> Value {
        Value hi = arith::SubIOp::create(rewriter, loc, dimOf(target, td), dimOf(source, sd));
        return arith::MaxSIOp::create(rewriter, loc, hi, c0);
      };
      SmallVector<OpFoldResult> lows(maskRank, OpFoldResult(rewriter.getIndexAttr(0)));
      SmallVector<OpFoldResult> highs(maskRank, OpFoldResult(rewriter.getIndexAttr(0)));
      highs[0] = highFor(Q, 0, mask, 0);
      highs[2] = highFor(Q, rank - 2, mask, 2);
      highs[3] = highFor(K, rank - 2, mask, 3);

      // Pad value = mask[0,0,0,0] (broadcast of the size-1 dims).  The pad
      // result keeps the mask element type; the broadcast generic below
      // performs the mask->score dtype conversion.
      SmallVector<Value> extractIdx(4, c0);
      Value padVal = rewriter.create<tensor::ExtractOp>(loc, mask, extractIdx);

      Value paddedMask = rewriter.create<tensor::PadOp>(
          loc, paddedMaskType, mask, lows, highs, padVal);

      // Broadcast the padded mask to scores shape using broadcastMap
      // which handles size-1 dimensions correctly via constant
      // expressions. After padding, every non-head mask dim matches the
      // scores dim, so the map never reads out of bounds.
      auto maskMap = broadcastMap(maskRank, maskRank, rewriter.getContext(),
                                   paddedMaskType.getShape());
      auto outMap = AffineMap::getMultiDimIdentityMap(maskRank, rewriter.getContext());

      SmallVector<Value> dynSizes;
      for (int64_t i = 0; i < outRank; ++i) {
        if (scoresShaped.isDynamicDim(i)) {
          if (i == rank - 1)
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, K, rank - 2));
          else if (i == rank - 2)
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, Q, rank - 2));
          else
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, Q, i));
        }
      }
      // Create 5D init with proper dyn sizes, then fill with broadcast mask
      Value maskInit5d = tensor::EmptyOp::create(rewriter, loc, scoresShaped, dynSizes);
      SmallVector<utils::IteratorType> addIters(maskRank, utils::IteratorType::parallel);
      auto maskBroadcast = linalg::GenericOp::create(rewriter, loc, scoresShaped,
          paddedMask, maskInit5d, {maskMap, outMap}, addIters,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value maskValue = args[0];
            if (maskValue.getType() != eltType && isa<FloatType>(maskValue.getType()) &&
                isa<FloatType>(eltType)) {
              auto inF = cast<FloatType>(maskValue.getType());
              auto outF = cast<FloatType>(eltType);
              if (inF.getWidth() < outF.getWidth())
                maskValue = arith::ExtFOp::create(b, loc, outF, maskValue);
              else if (inF.getWidth() > outF.getWidth())
                maskValue = arith::TruncFOp::create(b, loc, outF, maskValue);
            }
            b.create<linalg::YieldOp>(loc, maskValue);
          });
      Value mask5d = maskBroadcast.getResult(0);

      // Now apply comparison on the 5D mask
      Value additive = makeBinaryOp(rewriter, loc, mask5d, mask5d, scoresShaped, rewriter,
          [&](OpBuilder &b, Location loc, Value m, Value mT) {
            Value mF32 = m;
            if (isa<FloatType>(m.getType())) {
              auto mF = cast<FloatType>(m.getType());
              if (mF.getWidth() < 32)
                mF32 = arith::ExtFOp::create(b, loc, b.getF32Type(), m);
            }
            Value half = arith::ConstantOp::create(b, loc, b.getF32Type(),
                b.getFloatAttr(b.getF32Type(), 0.5f));
            Value cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, mF32, half);
            Value selected = arith::SelectOp::create(b, loc, cmp, zeroF32, negLarge);
            if (selected.getType() != eltType && isa<FloatType>(eltType) &&
                isa<FloatType>(selected.getType())) {
              auto outF = cast<FloatType>(eltType);
              auto selF = cast<FloatType>(selected.getType());
              if (selF.getWidth() > outF.getWidth())
                selected = arith::TruncFOp::create(b, loc, outF, selected);
            }
            return selected;
          });
      // Step 3b-iii: scoresScaled = scoresScaled + additive
      scoresScaled = makeBinaryOp(rewriter, loc, scoresScaled, additive, scoresScaled.getType(), rewriter,
          [&](OpBuilder &b, Location loc, Value a, Value bVal) {
            return arith::AddFOp::create(b, loc, a, bVal);
          });
      } else if (rawMaskType.getRank() <= 1) {
        // Scalar-mask contract: rank-0/rank-1 masks are additive scalars
        // (PyTorch additive mask / exported scalar dropout_p slot).  Extract
        // the scalar, cast it to the score element type, and add it to every
        // score.  This is a no-op for the LLaMA exported scalar 0.0 and must
        // NOT be interpreted as a boolean rank>=3 mask.
        Value scalarMask;
        if (rawMaskType.getRank() == 0) {
          scalarMask = rewriter.create<tensor::ExtractOp>(
              loc, rawMaskType.getElementType(), mask, ValueRange{});
        } else {
          Value maskC0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
          scalarMask = rewriter.create<tensor::ExtractOp>(
              loc, rawMaskType.getElementType(), mask, ValueRange{maskC0});
        }
        if (auto maskF = dyn_cast<FloatType>(scalarMask.getType())) {
          auto scoreF = cast<FloatType>(eltType);
          if (maskF.getWidth() < scoreF.getWidth())
            scalarMask = arith::ExtFOp::create(rewriter, loc, scoreF, scalarMask);
          else if (maskF.getWidth() > scoreF.getWidth())
            scalarMask = arith::TruncFOp::create(rewriter, loc, scoreF, scalarMask);
        }
        if (scalarMask.getType() != eltType) return failure();
        scoresScaled = makeUnaryOp(
            rewriter, loc, scoresScaled, scoresScaled.getType(), rewriter,
            [&](OpBuilder &b, Location bodyLoc, Value score) {
              return arith::AddFOp::create(b, bodyLoc, score, scalarMask);
            });
      }
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
    SmallVector<int64_t> outShape(resultShaped.getShape());
    outShape[rank - 1] = cast<RankedTensorType>(effectiveV.getType()).getDimSize(rank - 1);
    auto outType = RankedTensorType::get(outShape, eltType);
    auto outEmptyType = RankedTensorType::get(outShape, eltType);
    SmallVector<Value> outDyn;
    for (int64_t i = 0; i < rank; ++i) {
      if (!outEmptyType.isDynamicDim(i)) continue;
      if (i == rank - 1)
        outDyn.push_back(tensor::DimOp::create(rewriter, loc, effectiveV, rank - 1));
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
          ValueRange{attn, effectiveV}, ValueRange{outEmpty},
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
