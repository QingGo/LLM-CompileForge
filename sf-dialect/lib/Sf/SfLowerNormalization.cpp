#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace mlir::sf {

struct SfLayerNormOpLowering : public OpRewritePattern<sf::LayerNormOp> {
  using OpRewritePattern<sf::LayerNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::LayerNormOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Value bias = op.getBias();
    Type rt = op.getResult().getType();
    auto inputType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    if (!inputType) return failure();

    int64_t rank = inputType.getRank();
    int64_t lastDim = rank - 1;
    int64_t dimSize = inputType.getDimSize(lastDim);
    bool dynNormDim = (dimSize < 0);  // dynamic: compute at runtime
    if (dynNormDim) dimSize = 1;      // placeholder, actual value used via invDim

    auto eltType = getElementTypeOrSelf(rt);
    float eps = 1e-5f;

    // Compute 1.0 / dimSize — either constant or runtime
    Value invDim;
    if (dynNormDim) {
      Value dimVal = tensor::DimOp::create(rewriter, loc, input, lastDim);
      Value dimI64 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), dimVal);
      Value dimF32 = arith::UIToFPOp::create(rewriter, loc, eltType, dimI64);
      Value one = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, 1.0));
      invDim = arith::DivFOp::create(rewriter, loc, one, dimF32);
    } else {
      invDim = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, 1.0f / dimSize));
    }

    // Output type for reductions (same as input but last dim = 1, then broadcast)
    SmallVector<int64_t> reducedShape(inputType.getShape());
    reducedShape[lastDim] = 1;
    auto reducedType = RankedTensorType::get(reducedShape, eltType);

    // Helper: create element-wise binary generic with broadcast maps
    auto makeBinary = [&](Value lhs, Value rhs, Type outType,
                           function_ref<Value(OpBuilder &, Location, Value, Value)> fn) -> Value {
      Value empty = makeEmpty(rewriter, loc, outType, {lhs, rhs});
      if (!empty) return Value();
      auto outRank = ::mlir::cast<::mlir::ShapedType>(outType).getRank();
      auto lhsType = ::mlir::cast<::mlir::RankedTensorType>(lhs.getType());
      auto rhsType = ::mlir::cast<::mlir::RankedTensorType>(rhs.getType());

      // Compute broadcast maps for lhs and rhs
      SmallVector<AffineExpr> lhsExprs, rhsExprs;
      int64_t lhsRank = lhsType.getRank();
      int64_t rhsRank = rhsType.getRank();
      for (int64_t i = 0; i < outRank; ++i) {
        int64_t outDim = ::mlir::cast<::mlir::ShapedType>(outType).getDimSize(i);
        int64_t lhsI = i - (outRank - lhsRank);
        int64_t rhsI = i - (outRank - rhsRank);
        if (lhsI >= 0) {
          int64_t lhsDim = lhsType.getDimSize(lhsI);
          lhsExprs.push_back((lhsDim == 1 && (outDim == ShapedType::kDynamic || outDim > 1))
              ? getAffineConstantExpr(0, rewriter.getContext())
              : getAffineDimExpr(i, rewriter.getContext()));
        }
        if (rhsI >= 0) {
          int64_t rhsDim = rhsType.getDimSize(rhsI);
          rhsExprs.push_back((rhsDim == 1 && outDim > 1)
              ? getAffineConstantExpr(0, rewriter.getContext())
              : getAffineDimExpr(i, rewriter.getContext()));
        }
      }
      auto lhsMap = AffineMap::get(outRank, 0, lhsExprs, rewriter.getContext());
      auto rhsMap = AffineMap::get(outRank, 0, rhsExprs, rewriter.getContext());
      auto outMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());

      SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
      auto g = linalg::GenericOp::create(rewriter, loc, outType,
          ValueRange{lhs, rhs}, empty, {lhsMap, rhsMap, outMap}, iters);
      populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _v = fn(b, loc, args[0], args[1]);
        linalg::YieldOp::create(b, loc, _v);
      });
      return g.getResult(0);
    };

    // Helper: unary generic with broadcast
    auto makeUnary = [&](Value in, Type outType,
                          function_ref<Value(OpBuilder &, Location, Value)> fn) -> Value {
      Value empty = makeEmpty(rewriter, loc, outType, {in});
      if (!empty) return Value();
      auto outRank = ::mlir::cast<::mlir::ShapedType>(outType).getRank();
      auto inType = ::mlir::cast<::mlir::RankedTensorType>(in.getType());
      int64_t inRank = inType.getRank();

      SmallVector<AffineExpr> inExprs;
      for (int64_t i = 0; i < outRank; ++i) {
        int64_t inI = i - (outRank - inRank);
        if (inI >= 0) {
          int64_t outDim = ::mlir::cast<::mlir::ShapedType>(outType).getDimSize(i);
          int64_t inDim = inType.getDimSize(inI);
          inExprs.push_back((inDim == 1 && (outDim == ShapedType::kDynamic || outDim > 1))
              ? getAffineConstantExpr(0, rewriter.getContext())
              : getAffineDimExpr(i, rewriter.getContext()));
        }
      }
      auto inMap = AffineMap::get(outRank, 0, inExprs, rewriter.getContext());
      auto outMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());

      SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
      auto g = linalg::GenericOp::create(rewriter, loc, outType,
          in, empty, {inMap, outMap}, iters);
      populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _v = fn(b, loc, args[0]);
        linalg::YieldOp::create(b, loc, _v);
      });
      return g.getResult(0);
    };

    // Helper: reduce along last dim
    auto makeReduce = [&](Value in, Type reduType) -> Value {
      Value empty = makeEmpty(rewriter, loc, reduType, {in});
      if (!empty) return Value();
      // Initialize reduction output to 0 before reduction
      Value zero = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, 0.0f));
      auto fill = linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{empty});
      Value filled = fill.getResult(0);
      SmallVector<utils::IteratorType> iters(rank);
      for (int64_t i = 0; i < rank; ++i)
        iters[i] = (i == lastDim) ? utils::IteratorType::reduction : utils::IteratorType::parallel;
      // Maps: input identity, output projects out the last dim
      auto inMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
      SmallVector<AffineExpr> outExprs;
      for (int64_t i = 0; i < rank; ++i) {
        if (i == lastDim)
          outExprs.push_back(getAffineConstantExpr(0, rewriter.getContext()));
        else
          outExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
      }
      auto outMap = AffineMap::get(rank, 0, outExprs, rewriter.getContext());
      auto g = linalg::GenericOp::create(rewriter, loc, reduType, in, filled,
          {inMap, outMap}, iters);
      populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _ad = arith::AddFOp::create(b, loc, args[0], args[1]); linalg::YieldOp::create(b, loc, _ad);
      });
      return g.getResult(0);
    };

    // Step 1: mean = reduce_sum(x) / dimSize
    Value sumVal = makeReduce(input, reducedType);
    Value meanVal = makeUnary(sumVal, reducedType, [&](OpBuilder &b, Location loc, Value v) {
      return arith::MulFOp::create(b, loc, v, invDim);
    });

    // Step 2: diff = x - mean (broadcast)
    Value diff = makeBinary(input, meanVal, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::SubFOp::create(b, loc, a, bVal);
        });

    // Step 3: diff_sq = diff * diff
    Value diffSq = makeBinary(diff, diff, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::MulFOp::create(b, loc, a, bVal);
        });

    // Step 4: var = reduce_sum(diff_sq) / dimSize
    Value varSum = makeReduce(diffSq, reducedType);
    Value varVal = makeUnary(varSum, reducedType, [&](OpBuilder &b, Location loc, Value v) {
      return arith::MulFOp::create(b, loc, v, invDim);
    });

    // Step 5: std = sqrt(var + eps)
    Value stdVal = makeUnary(varVal, reducedType, [&](OpBuilder &b, Location loc, Value v) {
      Value epsVal = arith::ConstantOp::create(b, loc, eltType,
          b.getFloatAttr(eltType, eps));
      Value add = arith::AddFOp::create(b, loc, v, epsVal);
      return math::SqrtOp::create(b, loc, add);
    });

    // Step 6: norm = diff / std (broadcast)
    Value norm = makeBinary(diff, stdVal, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::DivFOp::create(b, loc, a, bVal);
        });

    // Step 7: result = norm * weight + bias
    Value weighted = makeBinary(norm, weight, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::MulFOp::create(b, loc, a, bVal);
        });
    Value result = makeBinary(weighted, bias, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::AddFOp::create(b, loc, a, bVal);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

// RMSNorm lowering: x / sqrt(mean(x²) + eps) * weight
// Step 1: x² = x * x
// Step 2: mean_x2 = reduce_sum(x², last_dim) / dimSize
// Step 3: rms = sqrt(mean_x2 + eps)
// Step 4: normed = x / rms (broadcast)
// Step 5: out = normed * weight
struct SfRmsNormOpLowering : public OpRewritePattern<sf::RmsNormOp> {
  using OpRewritePattern<sf::RmsNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::RmsNormOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Value weight = op.getWeight();
    Type rt = op.getResult().getType();
    auto inputType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    if (!inputType) return failure();

    int64_t rank = inputType.getRank();
    int64_t lastDim = rank - 1;
    int64_t dimSize = inputType.getDimSize(lastDim);
    if (dimSize < 0) return failure();

    auto eltType = getElementTypeOrSelf(rt);
    float eps = 1e-5f;

    // reduced type: last dim = 1 (for broadcasting)
    SmallVector<int64_t> reducedShape(inputType.getShape());
    reducedShape[lastDim] = 1;
    auto reducedType = RankedTensorType::get(reducedShape, eltType);

    // Helper: reduce along last dim (sum)
    auto makeReduce = [&](Value in, Type reduType) -> Value {
      Value empty = makeEmpty(rewriter, loc, reduType, {in});
      if (!empty) return Value();
      // Initialize to 0 before reduction (same as LayerNorm makeReduce)
      Value zero = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, 0.0f));
      auto fill = linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{empty});
      Value filled = fill.getResult(0);
      SmallVector<utils::IteratorType> iters(rank);
      for (int64_t i = 0; i < rank; ++i)
        iters[i] = (i == lastDim) ? utils::IteratorType::reduction : utils::IteratorType::parallel;
      auto inMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
      SmallVector<AffineExpr> outExprs;
      for (int64_t i = 0; i < rank; ++i) {
        if (i == lastDim) outExprs.push_back(getAffineConstantExpr(0, rewriter.getContext()));
        else outExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
      }
      auto outMap = AffineMap::get(rank, 0, outExprs, rewriter.getContext());
      auto g = linalg::GenericOp::create(rewriter, loc, reduType, in, filled,
          {inMap, outMap}, iters);
      populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _ad = arith::AddFOp::create(b, loc, args[0], args[1]);
        linalg::YieldOp::create(b, loc, _ad);
      });
      return g.getResult(0);
    };

    // Helper: unary generic with broadcast
    auto makeUnary = [&](Value in, Type outType,
                          function_ref<Value(OpBuilder &, Location, Value)> fn) -> Value {
      Value empty = makeEmpty(rewriter, loc, outType, {in});
      if (!empty) return Value();
      auto outRank = ::mlir::cast<::mlir::ShapedType>(outType).getRank();
      auto inType = ::mlir::cast<::mlir::RankedTensorType>(in.getType());
      int64_t inRank = inType.getRank();
      SmallVector<AffineExpr> inExprs;
      for (int64_t i = 0; i < outRank; ++i) {
        int64_t inI = i - (outRank - inRank);
        if (inI >= 0) {
          int64_t outDim = ::mlir::cast<::mlir::ShapedType>(outType).getDimSize(i);
          int64_t inDim = inType.getDimSize(inI);
          inExprs.push_back((inDim == 1 && (outDim == ShapedType::kDynamic || outDim > 1))
              ? getAffineConstantExpr(0, rewriter.getContext())
              : getAffineDimExpr(i, rewriter.getContext()));
        }
      }
      auto inMap = AffineMap::get(outRank, 0, inExprs, rewriter.getContext());
      auto outMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());
      SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
      auto g = linalg::GenericOp::create(rewriter, loc, outType, in, empty,
          {inMap, outMap}, iters);
      populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _v = fn(b, loc, args[0]);
        linalg::YieldOp::create(b, loc, _v);
      });
      return g.getResult(0);
    };

    // Helper: binary generic with broadcast
    auto makeBinary = [&](Value lhs, Value rhs, Type outType,
                           function_ref<Value(OpBuilder &, Location, Value, Value)> fn) -> Value {
      Value empty = makeEmpty(rewriter, loc, outType, {lhs, rhs});
      if (!empty) return Value();
      auto outRank = ::mlir::cast<::mlir::ShapedType>(outType).getRank();
      auto lhsType = ::mlir::cast<::mlir::RankedTensorType>(lhs.getType());
      auto rhsType = ::mlir::cast<::mlir::RankedTensorType>(rhs.getType());
      int64_t lhsRank = lhsType.getRank(), rhsRank = rhsType.getRank();
      SmallVector<AffineExpr> lhsExprs, rhsExprs;
      for (int64_t i = 0; i < outRank; ++i) {
        int64_t outDim = ::mlir::cast<::mlir::ShapedType>(outType).getDimSize(i);
        int64_t lhsI = i - (outRank - lhsRank);
        int64_t rhsI = i - (outRank - rhsRank);
        if (lhsI >= 0) {
          int64_t lhsDim = lhsType.getDimSize(lhsI);
          lhsExprs.push_back((lhsDim == 1 && (outDim == ShapedType::kDynamic || outDim > 1))
              ? getAffineConstantExpr(0, rewriter.getContext())
              : getAffineDimExpr(i, rewriter.getContext()));
        }
        if (rhsI >= 0) {
          int64_t rhsDim = rhsType.getDimSize(rhsI);
          rhsExprs.push_back((rhsDim == 1 && outDim > 1)
              ? getAffineConstantExpr(0, rewriter.getContext())
              : getAffineDimExpr(i, rewriter.getContext()));
        }
      }
      auto lhsMap = AffineMap::get(outRank, 0, lhsExprs, rewriter.getContext());
      auto rhsMap = AffineMap::get(outRank, 0, rhsExprs, rewriter.getContext());
      auto outMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());
      SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
      auto g = linalg::GenericOp::create(rewriter, loc, outType,
          ValueRange{lhs, rhs}, empty, {lhsMap, rhsMap, outMap}, iters);
      populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _v = fn(b, loc, args[0], args[1]);
        linalg::YieldOp::create(b, loc, _v);
      });
      return g.getResult(0);
    };

    // Step 1: x² = x * x
    Value sq = makeBinary(input, input, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::MulFOp::create(b, loc, a, bVal);
        });

    // Step 2: mean_x2 = reduce_sum(x²) / dimSize
    Value sumSq = makeReduce(sq, reducedType);
    Value meanSq = makeUnary(sumSq, reducedType, [&](OpBuilder &b, Location loc, Value v) {
      Value scale = arith::ConstantOp::create(b, loc, eltType,
          b.getFloatAttr(eltType, 1.0f / dimSize));
      return arith::MulFOp::create(b, loc, v, scale);
    });

    // Step 3: rms = sqrt(mean_x2 + eps)
    Value rmsVal = makeUnary(meanSq, reducedType, [&](OpBuilder &b, Location loc, Value v) {
      Value epsVal = arith::ConstantOp::create(b, loc, eltType,
          b.getFloatAttr(eltType, eps));
      Value add = arith::AddFOp::create(b, loc, v, epsVal);
      return math::SqrtOp::create(b, loc, add);
    });

    // Step 4: normed = x / rms (broadcast)
    Value normed = makeBinary(input, rmsVal, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::DivFOp::create(b, loc, a, bVal);
        });

    // Step 5: out = normed * weight
    Value result = makeBinary(normed, weight, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::MulFOp::create(b, loc, a, bVal);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerNormalizationPatterns(RewritePatternSet &patterns) {
  patterns.add<SfLayerNormOpLowering, SfRmsNormOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
