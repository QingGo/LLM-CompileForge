#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include <algorithm>
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace mlir::sf {

// Identity → passthrough; handle type mismatches by inserting proper cast
struct IdentityLowering : public OpRewritePattern<sf::IdentityOp> {
  using OpRewritePattern<sf::IdentityOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::IdentityOp op, PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Type rt = op.getResult().getType();
    if (input.getType() == rt) {
      rewriter.replaceOp(op, input);
      return success();
    }
    // Type mismatch: insert linalg.generic with proper conversion
    auto loc = op.getLoc();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(rt);
    if (!inType || !outType) return failure();
    auto rank = outType.getRank();
    auto inElt = inType.getElementType();
    auto outElt = outType.getElementType();
    Value empty = makeEmpty(rewriter, loc, rt, {input});
    if (!empty) return failure();
    auto inType2 = cast<RankedTensorType>(input.getType());
    auto inRank = inType2.getRank();
    [[maybe_unused]] int64_t squeezeCount = (inRank > rank) ? (inRank - rank) : 0;
    auto inMap = broadcastMap(rank, inRank, rewriter.getContext(), inType2.getShape());
    auto outMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(rewriter, loc, rt, input, empty,
        {inMap, outMap}, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
          Value v;
          if (isa<IntegerType>(inElt) && isa<FloatType>(outElt))
            v = arith::UIToFPOp::create(b, bodyLoc, outElt, args[0]);
          else if (isa<FloatType>(inElt) && isa<IntegerType>(outElt))
            v = arith::FPToUIOp::create(b, bodyLoc, outElt, args[0]);
          else if (isa<FloatType>(inElt) && isa<FloatType>(outElt)) {
            auto inF = cast<FloatType>(inElt);
            auto outF = cast<FloatType>(outElt);
            if (inF.getWidth() < outF.getWidth())
              v = arith::ExtFOp::create(b, bodyLoc, outElt, args[0]);
            else if (inF.getWidth() > outF.getWidth())
              v = arith::TruncFOp::create(b, bodyLoc, outElt, args[0]);
            else
              v = args[0];
          } else if (isa<IntegerType>(inElt) && isa<IntegerType>(outElt)) {
            auto inI = cast<IntegerType>(inElt);
            auto outI = cast<IntegerType>(outElt);
            if (inI.getWidth() > outI.getWidth())
              v = arith::TruncIOp::create(b, bodyLoc, outElt, args[0]);
            else if (inI.getWidth() < outI.getWidth())
              v = arith::ExtUIOp::create(b, bodyLoc, outElt, args[0]);
            else
              v = args[0];
          } else {
            v = args[0];
          }
          linalg::YieldOp::create(b, bodyLoc, v);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

static SmallVector<int64_t> getSumReduceDims(sf::SumOp op, int64_t rank) {
  SmallVector<int64_t> dims;
  if (auto arr = op->getAttrOfType<ArrayAttr>("dim")) {
    for (Attribute a : arr) {
      if (auto i = dyn_cast<IntegerAttr>(a))
        dims.push_back(i.getInt());
    }
  } else if (auto i = op->getAttrOfType<IntegerAttr>("dim")) {
    dims.push_back(i.getInt());
  } else {
    // No explicit dim: ATen ``sum(dim=...)`` was not supplied; reduce all.
    for (int64_t i = 0; i < rank; ++i)
      dims.push_back(i);
  }
  SmallVector<int64_t> normalized;
  for (int64_t d : dims) {
    if (d < 0)
      d += rank;
    if (d >= 0 && d < rank && std::find(normalized.begin(), normalized.end(), d) == normalized.end())
      normalized.push_back(d);
  }
  return normalized;
}

struct SfSumOpLowering : public OpRewritePattern<sf::SumOp> {
  using OpRewritePattern<sf::SumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::SumOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();
    auto inType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outType = dyn_cast<RankedTensorType>(rt);
    if (!inType || !outType) return failure();

    int64_t inRank = inType.getRank();
    int64_t outRank = outType.getRank();
    bool keepdim = false;
    if (auto kd = op->getAttrOfType<BoolAttr>("keepdim"))
      keepdim = kd.getValue();
    SmallVector<int64_t> redDims = getSumReduceDims(op, inRank);
    if (redDims.empty())
      return failure();
    int64_t expectedRank = keepdim ? inRank : inRank - (int64_t)redDims.size();
    if (outRank != expectedRank) {
      return op.emitOpError("sf.sum result rank mismatch: expected ")
             << expectedRank << ", got " << outRank;
    }

    Value empty = makeZeroedEmpty(rewriter, loc, rt, {op.getInput()});
    if (!empty) return failure();

    // input identity map; output map keeps non-reduced dims, and uses constant
    // 0 for keepdim singleton positions.
    SmallVector<bool> isRed(inRank, false);
    for (int64_t d : redDims)
      isRed[d] = true;
    SmallVector<AffineExpr> outExprs;
    for (int64_t i = 0; i < inRank; ++i) {
      if (!isRed[i])
        outExprs.push_back(rewriter.getAffineDimExpr(i));
      else if (keepdim)
        outExprs.push_back(rewriter.getAffineConstantExpr(0));
    }
    auto inMap = AffineMap::getMultiDimIdentityMap(inRank, rewriter.getContext());
    auto outMap = AffineMap::get(inRank, 0, outExprs, rewriter.getContext());

    SmallVector<utils::IteratorType> iterTypes;
    iterTypes.reserve(inRank);
    for (int64_t i = 0; i < inRank; ++i)
      iterTypes.push_back(isRed[i] ? utils::IteratorType::reduction
                                   : utils::IteratorType::parallel);

    auto g = linalg::GenericOp::create(rewriter, loc, rt, op.getInput(), empty,
        {inMap, outMap}, iterTypes);
    populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      auto elt = cast<ShapedType>(rt).getElementType();
      Value add;
      if (isa<FloatType>(elt))
        add = arith::AddFOp::create(b, loc, args[0], args[1]);
      else
        add = arith::AddIOp::create(b, loc, args[0], args[1]);
      linalg::YieldOp::create(b, loc, add);
    });
    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

struct SfMeanOpLowering : public OpRewritePattern<sf::MeanOp> {
  using OpRewritePattern<sf::MeanOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::MeanOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto rt = op.getResult().getType();
    auto outType = dyn_cast<RankedTensorType>(rt);
    auto inType = dyn_cast<RankedTensorType>(op.getInput().getType());
    if (!outType || !inType) return failure();
    int64_t inRank = inType.getRank();
    int64_t outRank = outType.getRank();
    if (inRank < 1) return failure();
    // Default: reduce along last dimension (matching aten.mean.dim with dim=-1)
    int64_t redDim = inRank - 1;
    bool keepdim = false;
    if (auto kdAttr = op->getAttrOfType<BoolAttr>("keepdim"))
      keepdim = kdAttr.getValue();
    // Validate: if keepdim, outRank == inRank; otherwise outRank == inRank - 1
    if (keepdim && outRank != inRank) return failure();
    if (!keepdim && outRank != inRank - 1) return failure();
    // Build init (zeroed) and input as linalg.generic reduce
    Value init = makeZeroedEmpty(rewriter, loc, rt, {op.getInput()});
    if (!init) return failure();
    // Build maps: input identity, output projects the reduced dim away.
    // With keepdim the output rank equals the input rank and the reduced
    // dim is a size-1 singleton; its output index must be constant 0.
    // Pushing the affine dim expr for the reduction dim would make the
    // lowered loop write 1..dim-1 into a size-1 memref (heap corruption).
    auto inMap = AffineMap::getMultiDimIdentityMap(inRank, rewriter.getContext());
    SmallVector<AffineExpr> outExprs;
    for (int64_t i = 0; i < inRank; ++i) {
      if (i != redDim)
        outExprs.push_back(rewriter.getAffineDimExpr(i));
      else if (keepdim)
        outExprs.push_back(rewriter.getAffineConstantExpr(0));
    }
    auto outMap = AffineMap::get(inRank, 0, outExprs, rewriter.getContext());
    SmallVector<utils::IteratorType> iterTypes(inRank, utils::IteratorType::parallel);
    iterTypes[redDim] = utils::IteratorType::reduction;
    auto sumOp = linalg::GenericOp::create(rewriter, loc, rt, op.getInput(), init,
        {inMap, outMap}, iterTypes);
    populateBody(sumOp, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value add = arith::AddFOp::create(b, loc, args[0], args[1]);
      linalg::YieldOp::create(b, loc, add);
    });
    Value sum = sumOp.getResult(0);
    int64_t redSize = inType.getDimSize(redDim);
    if (ShapedType::isDynamic(redSize) || redSize <= 1) {
      rewriter.replaceOp(op, sum);
      return success();
    }
    auto eltType = inType.getElementType();
    Value count = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, static_cast<double>(redSize)));
    Value divEmpty = makeEmpty(rewriter, loc, rt, {sum});
    if (!divEmpty) return failure();
    SmallVector<utils::IteratorType> parTypes(outRank, utils::IteratorType::parallel);
    auto divMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());
    auto divOp = linalg::GenericOp::create(rewriter, loc, rt, sum, divEmpty,
        {divMap, divMap}, parTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
          Value d = arith::DivFOp::create(b, bodyLoc, args[0], count);
          linalg::YieldOp::create(b, bodyLoc, d);
        });
    rewriter.replaceOp(op, divOp.getResult(0));
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerReducePatterns(RewritePatternSet &patterns) {
  patterns.add<IdentityLowering, SfSumOpLowering, SfMeanOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
