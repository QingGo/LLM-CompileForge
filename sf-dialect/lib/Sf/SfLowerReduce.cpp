#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace {

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
    int64_t squeezeCount = (inRank > rank) ? (inRank - rank) : 0;
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
          else
            v = args[0];
          linalg::YieldOp::create(b, bodyLoc, v);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

struct SfSumOpLowering : public OpRewritePattern<sf::SumOp> {
  using OpRewritePattern<sf::SumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::SumOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); auto rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();
    Value empty = makeZeroedEmpty(rewriter, loc, rt, {op.getInput()});
    if (!empty) return failure();
    auto rank = cast<ShapedType>(rt).getRank();
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::reduction);
    auto g = linalg::GenericOp::create(rewriter, loc, rt, op.getInput(), empty,
        identityMaps(rank, 2, rewriter.getContext()), iterTypes);
    populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value add = arith::AddFOp::create(b, loc, args[0], args[1]);
      linalg::YieldOp::create(b, loc, add);
    });
    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerReducePatterns(RewritePatternSet &patterns) {
  patterns.add<IdentityLowering, SfSumOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
