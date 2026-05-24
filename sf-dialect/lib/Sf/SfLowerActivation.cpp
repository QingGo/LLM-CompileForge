#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace mlir::sf {

// Relu lowering
struct ReluLowering : public OpRewritePattern<sf::ReluOp> {
  using OpRewritePattern<sf::ReluOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::ReluOp op, PatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    if (!isa<ShapedType>(resultType)) return failure();
    auto loc = op.getLoc();
    Value empty = makeEmpty(rewriter, loc, resultType, {op.getInput()});
    if (!empty) return failure();
    auto rank = cast<ShapedType>(resultType).getRank();
    auto eltType = getElementTypeOrSelf(resultType);
    if (!isa<FloatType>(eltType)) return failure();

    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultType, op.getInput(), empty,
        identityMaps(rank, 2, rewriter.getContext()), iterTypes);
    populateBody(generic, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value zero = arith::ConstantOp::create(b, loc, eltType,
          b.getFloatAttr(eltType, 0.0));
      Value v = arith::MaxNumFOp::create(b, loc, args[0], zero);
      linalg::YieldOp::create(b, loc, v);
    });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerActivationPatterns(RewritePatternSet &patterns) {
  patterns.add<SfBinaryLowering<sf::AddOp, arith::AddFOp>,
               SfBinaryLowering<sf::MulOp, arith::MulFOp>,
               SfBinaryLowering<sf::SubOp, arith::SubFOp>,
               SfBinaryLowering<sf::PowOp, math::PowFOp>,
               SfBinaryLowering<sf::DivOp, arith::DivFOp>,
               SfBinaryLowering<sf::MaxOp, arith::MaxNumFOp>,
               ReluLowering,
               SfActivationOpLowering<sf::GeluOp>,
               SfActivationOpLowering<sf::SiluOp>,
               SfActivationOpLowering<sf::SigmoidOp>,
               SfActivationOpLowering<sf::ExpOp>,
               SfActivationOpLowering<sf::NegOp>,
               SfActivationOpLowering<sf::TanhOp>>(patterns.getContext());
}
} // namespace mlir::sf
