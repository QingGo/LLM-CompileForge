#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace {

// Le comparison → arith.cmpf in generic
// Computes lhs <= rhs element-wise. Output is f32 (0.0/1.0) to avoid
// i1→f32 unrealized_conversion_cast downstream.
struct SfLeOpLowering : public OpRewritePattern<sf::LeOp> {
  using OpRewritePattern<sf::LeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::LeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();
    auto eltType = cast<ShapedType>(rt).getElementType();
    // Always use f32 output to avoid i1→f32 unrealized_conversion_cast
    auto f32OutTypeRaw = cast<ShapedType>(rt).cloneWith(
        cast<ShapedType>(rt).getShape(), rewriter.getF32Type());
    auto refinedType = refineBroadcastType(
        cast<RankedTensorType>(f32OutTypeRaw),
        ValueRange{op.getLhs(), op.getRhs()});
    Value empty = makeEmpty(rewriter, loc, refinedType, {op.getLhs(), op.getRhs()});
    if (!empty) return failure();
    auto rank = cast<ShapedType>(rt).getRank();
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto lhsType2 = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType2 = cast<RankedTensorType>(op.getRhs().getType());
    auto lhsMap = broadcastMap(rank, lhsType2.getRank(), rewriter.getContext(), lhsType2.getShape());
    auto rhsMap = broadcastMap(rank, rhsType2.getRank(), rewriter.getContext(), rhsType2.getShape());
    auto outMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    auto g = linalg::GenericOp::create(rewriter, loc, refinedType,
        ValueRange{op.getLhs(), op.getRhs()}, empty,
        {lhsMap, rhsMap, outMap}, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      Value cmp;
      if (isa<IntegerType>(args[0].getType()) || isa<IntegerType>(args[1].getType())) {
        // Integer comparison: use CmpIOp
        auto lhsInt = arith::IndexCastOp::create(b, loc, b.getIndexType(), args[0]);
        auto rhsInt = arith::IndexCastOp::create(b, loc, b.getIndexType(), args[1]);
        cmp = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, lhsInt, rhsInt);
      } else {
        // Float comparison
        cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLE, args[0], args[1]);
      }
      Value result = arith::UIToFPOp::create(b, loc, rewriter.getF32Type(), cmp);
      linalg::YieldOp::create(b, loc, result);
    });
    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

// LogicalAnd → linalg.generic with f32 operands
//   bool_a = cmp UGT(a, 0.0), bool_b = cmp UGT(b, 0.0)
//   and = andi(bool_a, bool_b)
//   result = uitofp(and) → f32
struct SfLogicalAndOpLowering : public OpRewritePattern<sf::LogicalAndOp> {
  using OpRewritePattern<sf::LogicalAndOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::LogicalAndOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();
    // Always use f32 output type to avoid i1→f32 unrealized_conversion_cast
    auto outType = cast<ShapedType>(rt);
    auto f32OutTypeRaw = outType.cloneWith(outType.getShape(), rewriter.getF32Type());
    auto refinedType = refineBroadcastType(
        cast<RankedTensorType>(f32OutTypeRaw),
        ValueRange{op.getLhs(), op.getRhs()});
    Value empty = makeEmpty(rewriter, loc, refinedType, {op.getLhs(), op.getRhs()});
    if (!empty) return failure();
    auto rank = outType.getRank();
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto lhsType2 = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType2 = cast<RankedTensorType>(op.getRhs().getType());
    auto lhsMap = broadcastMap(rank, lhsType2.getRank(), rewriter.getContext(), lhsType2.getShape());
    auto rhsMap = broadcastMap(rank, rhsType2.getRank(), rewriter.getContext(), rhsType2.getShape());
    auto outMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    auto g = linalg::GenericOp::create(rewriter, loc, refinedType,
        ValueRange{op.getLhs(), op.getRhs()}, empty,
        {lhsMap, rhsMap, outMap}, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      // Both args are 0.0 (false) or 1.0 (true). Multiply gives AND.
      Value result = arith::MulFOp::create(b, bodyLoc, args[0], args[1]);
      linalg::YieldOp::create(b, bodyLoc, result);
    });
    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerComparePatterns(RewritePatternSet &patterns) {
  patterns.add<SfLeOpLowering, SfLogicalAndOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
