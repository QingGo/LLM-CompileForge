#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace mlir::sf {

// Le comparison -> arith.cmpf in generic with explicit linalg.broadcast
// Computes lhs <= rhs element-wise. Output is f32 (0.0/1.0) to avoid
// i1->f32 unrealized_conversion_cast downstream.
// Uses explicit linalg.broadcast + identity-map linalg.generic instead of
// broadcast affine maps to avoid kDynamic leaks from InferStaticShapeOfOperands.
struct SfLeOpLowering : public OpRewritePattern<sf::LeOp> {
  using OpRewritePattern<sf::LeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::LeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();

    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());
    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    int64_t outRank = std::max(lhsRank, rhsRank);
    MLIRContext *ctx = rewriter.getContext();

    // 1. Compute broadcast output shape (numpy rules)
    SmallVector<int64_t> outShape(outRank, ShapedType::kDynamic);
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < outRank; ++i) {
      int64_t lhsDim = (i >= outRank - lhsRank)
          ? lhsType.getDimSize(i - (outRank - lhsRank)) : 1;
      int64_t rhsDim = (i >= outRank - rhsRank)
          ? rhsType.getDimSize(i - (outRank - rhsRank)) : 1;

      bool lhsDynamic = ShapedType::isDynamic(lhsDim);
      bool rhsDynamic = ShapedType::isDynamic(rhsDim);

      if (lhsDynamic || rhsDynamic) {
        outShape[i] = ShapedType::kDynamic;
        if (i >= outRank - lhsRank) {
          int64_t li = i - (outRank - lhsRank);
          if (lhsType.isDynamicDim(li)) {
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, op.getLhs(), li));
            continue;
          }
        }
        if (i >= outRank - rhsRank) {
          int64_t ri = i - (outRank - rhsRank);
          if (rhsType.isDynamicDim(ri)) {
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, op.getRhs(), ri));
          }
        }
      } else {
        outShape[i] = std::max(lhsDim, rhsDim);
      }
    }

    auto f32Type = rewriter.getF32Type();
    auto outTensorType = RankedTensorType::get(outShape, f32Type);
    Value genericInit = tensor::EmptyOp::create(rewriter, loc, outTensorType, dynSizes);

    // 2. Build per-operand broadcast affine maps.
    auto buildBroadcastMap = [&](RankedTensorType opType, int64_t opRank)
        -> AffineMap {
      SmallVector<AffineExpr> exprs;
      for (int64_t i = 0; i < outRank; ++i) {
        int64_t opIdx = i - (outRank - opRank);
        if (opIdx < 0) continue;
        int64_t opSize = opType.getDimSize(opIdx);
        bool needsBroadcast = (opSize == 1) &&
            (ShapedType::isDynamic(outShape[i]) || outShape[i] > 1);
        exprs.push_back(needsBroadcast
            ? getAffineConstantExpr(0, ctx)
            : getAffineDimExpr(i, ctx));
      }
      return AffineMap::get(outRank, 0, exprs, ctx);
    };
    auto lhsMap = buildBroadcastMap(lhsType, lhsRank);
    auto rhsMap = buildBroadcastMap(rhsType, rhsRank);
    auto outMap = AffineMap::getMultiDimIdentityMap(outRank, ctx);

    // 3. linalg.generic with broadcast maps
    SmallVector<AffineMap> genericMaps = {lhsMap, rhsMap, outMap};
    SmallVector<utils::IteratorType> iterTypes(outRank, utils::IteratorType::parallel);

    auto g = linalg::GenericOp::create(rewriter, loc, outTensorType,
        ValueRange{op.getLhs(), op.getRhs()}, genericInit,
        genericMaps, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      Value cmp;
      if (isa<IntegerType>(args[0].getType()) || isa<IntegerType>(args[1].getType())) {
        auto lhsInt = arith::IndexCastOp::create(b, loc, b.getIndexType(), args[0]);
        auto rhsInt = arith::IndexCastOp::create(b, loc, b.getIndexType(), args[1]);
        cmp = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, lhsInt, rhsInt);
      } else {
        cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLE, args[0], args[1]);
      }
      Value result = arith::UIToFPOp::create(b, loc, rewriter.getF32Type(), cmp);
      linalg::YieldOp::create(b, loc, result);
    });

    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

// LogicalAnd -> linalg.generic with explicit linalg.broadcast
//   bool_a = cmp UGT(a, 0.0), bool_b = cmp UGT(b, 0.0)
//   and = andi(bool_a, bool_b)
//   result = uitofp(and) -> f32
// Uses explicit linalg.broadcast + identity-map linalg.generic to avoid
// kDynamic leaks from InferStaticShapeOfOperands.
struct SfLogicalAndOpLowering : public OpRewritePattern<sf::LogicalAndOp> {
  using OpRewritePattern<sf::LogicalAndOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::LogicalAndOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();

    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());
    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    int64_t outRank = std::max(lhsRank, rhsRank);
    MLIRContext *ctx = rewriter.getContext();

    // 1. Compute broadcast output shape (numpy rules)
    SmallVector<int64_t> outShape(outRank, ShapedType::kDynamic);
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < outRank; ++i) {
      int64_t lhsDim = (i >= outRank - lhsRank)
          ? lhsType.getDimSize(i - (outRank - lhsRank)) : 1;
      int64_t rhsDim = (i >= outRank - rhsRank)
          ? rhsType.getDimSize(i - (outRank - rhsRank)) : 1;

      bool lhsDynamic = ShapedType::isDynamic(lhsDim);
      bool rhsDynamic = ShapedType::isDynamic(rhsDim);

      if (lhsDynamic || rhsDynamic) {
        outShape[i] = ShapedType::kDynamic;
        if (i >= outRank - lhsRank) {
          int64_t li = i - (outRank - lhsRank);
          if (lhsType.isDynamicDim(li)) {
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, op.getLhs(), li));
            continue;
          }
        }
        if (i >= outRank - rhsRank) {
          int64_t ri = i - (outRank - rhsRank);
          if (rhsType.isDynamicDim(ri)) {
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, op.getRhs(), ri));
          }
        }
      } else {
        outShape[i] = std::max(lhsDim, rhsDim);
      }
    }

    auto f32Type = rewriter.getF32Type();
    auto outTensorType = RankedTensorType::get(outShape, f32Type);
    Value genericInit = tensor::EmptyOp::create(rewriter, loc, outTensorType, dynSizes);

    // 2. Build per-operand broadcast affine maps.
    // Maps each operand's dims to output dims, using affine constant 0
    // for size-1 dims that need broadcast (static >1 or dynamic output).
    auto buildBroadcastMap = [&](RankedTensorType opType, int64_t opRank)
        -> AffineMap {
      SmallVector<AffineExpr> exprs;
      for (int64_t i = 0; i < outRank; ++i) {
        int64_t opIdx = i - (outRank - opRank);
        if (opIdx < 0) continue;  // leading output dims not in operand
        int64_t opSize = opType.getDimSize(opIdx);
        bool needsBroadcast = (opSize == 1) &&
            (ShapedType::isDynamic(outShape[i]) || outShape[i] > 1);
        exprs.push_back(needsBroadcast
            ? getAffineConstantExpr(0, ctx)
            : getAffineDimExpr(i, ctx));
      }
      return AffineMap::get(outRank, 0, exprs, ctx);
    };
    auto lhsMap = buildBroadcastMap(lhsType, lhsRank);
    auto rhsMap = buildBroadcastMap(rhsType, rhsRank);
    auto outMap = AffineMap::getMultiDimIdentityMap(outRank, ctx);

    // 3. linalg.generic with broadcast maps
    SmallVector<AffineMap> genericMaps = {lhsMap, rhsMap, outMap};
    SmallVector<utils::IteratorType> iterTypes(outRank, utils::IteratorType::parallel);

    auto g = linalg::GenericOp::create(rewriter, loc, outTensorType,
        ValueRange{op.getLhs(), op.getRhs()}, genericInit,
        genericMaps, iterTypes,
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
