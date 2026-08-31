#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace mlir::sf {

/// Convert a tensor from any integer type to f32 via elementwise uitofp.
/// Handles i1 from mask operations that need to participate in float ops.
static Value convertToF32(PatternRewriter &rewriter, Location loc, Value input,
                          RankedTensorType inputType) {
  auto f32 = rewriter.getF32Type();
  auto outShape = inputType.getShape();
  SmallVector<Value> dynSizes;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    if (inputType.isDynamicDim(i)) {
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, input, i));
    }
  }
  auto outType = RankedTensorType::get(outShape, f32);
  Value init = tensor::EmptyOp::create(rewriter, loc, outType, dynSizes);
  auto identityMap = AffineMap::getMultiDimIdentityMap(inputType.getRank(),
                                                       rewriter.getContext());
  SmallVector<utils::IteratorType> iterTypes(inputType.getRank(),
                                             utils::IteratorType::parallel);
  SmallVector<AffineMap> maps = {identityMap, identityMap};
  return linalg::GenericOp::create(rewriter, loc, outType, ValueRange{input},
      ValueRange{init}, maps, iterTypes,
      [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
        Value cast = b.create<arith::UIToFPOp>(bodyLoc, f32, args[0]);
        linalg::YieldOp::create(b, bodyLoc, cast);
      })->getResult(0);
}

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

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = cast<RankedTensorType>(op.getRhs().getType());
    int64_t lhsRank = lhsType.getRank();
    int64_t rhsRank = rhsType.getRank();
    int64_t outRank = std::max(lhsRank, rhsRank);
    MLIRContext *ctx = rewriter.getContext();

    // 1. Compute broadcast output shape
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
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, lhs, li));
            continue;
          }
        }
        if (i >= outRank - rhsRank) {
          int64_t ri = i - (outRank - rhsRank);
          if (rhsType.isDynamicDim(ri)) {
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, rhs, ri));
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

    // Ensure both operands are f32 (inputs may be i1 from mask ops).
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto f32Type = rewriter.getF32Type();
    if (getElementTypeOrSelf(lhs.getType()) != f32Type) {
      lhs = convertToF32(rewriter, loc, lhs, lhsType);
      lhsType = cast<RankedTensorType>(lhs.getType());
      lhsRank = lhsType.getRank();
      outRank = std::max({lhsRank, rhsRank});
    }
    if (getElementTypeOrSelf(rhs.getType()) != f32Type) {
      rhs = convertToF32(rewriter, loc, rhs, rhsType);
      rhsType = cast<RankedTensorType>(rhs.getType());
      rhsRank = rhsType.getRank();
      outRank = std::max({lhsRank, rhsRank});
    }

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
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, lhs, li));
            continue;
          }
        }
        if (i >= outRank - rhsRank) {
          int64_t ri = i - (outRank - rhsRank);
          if (rhsType.isDynamicDim(ri)) {
            dynSizes.push_back(tensor::DimOp::create(rewriter, loc, rhs, ri));
          }
        }
      } else {
        outShape[i] = std::max(lhsDim, rhsDim);
      }
    }


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
        ValueRange{lhs, rhs}, genericInit,
        genericMaps, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      // Both args should now be f32 after convertToF32.
      Value result = arith::MulFOp::create(b, bodyLoc, args[0], args[1]);
      linalg::YieldOp::create(b, bodyLoc, result);
    });

    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

// Triangular masks (triu/tril) preserve the input element type.  For each
// element of the trailing two dimensions, keep the value when the column is
// on the kept side of the diagonal, otherwise replace it with the zero value
// of the element type (false for i1, 0 for integers, 0.0 for floats).
template <bool Upper>
static Value lowerTriangular(
    Value input, RankedTensorType outType, int64_t diagonal,
    PatternRewriter &rewriter) {
  auto loc = rewriter.getUnknownLoc();
  int64_t rank = outType.getRank();
  if (rank < 2) return Value();
  auto eltType = outType.getElementType();
  if (!isa<FloatType>(eltType) && !isa<IntegerType>(eltType)) return Value();
  Value empty = makeEmpty(rewriter, loc, outType, {input});
  if (!empty) return Value();

  Value zero;
  if (isa<FloatType>(eltType)) {
    zero = arith::ConstantOp::create(
        rewriter, loc, eltType, rewriter.getFloatAttr(eltType, 0.0f));
  } else {
    zero = arith::ConstantOp::create(
        rewriter, loc, eltType, rewriter.getIntegerAttr(eltType, 0));
  }

  auto idMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      rewriter, loc, outType, ValueRange{input}, ValueRange{empty},
      {idMap, idMap}, iters);
  populateBody(generic, rewriter, [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
    Value row = linalg::IndexOp::create(b, bodyLoc, rank - 2);
    Value col = linalg::IndexOp::create(b, bodyLoc, rank - 1);
    Value diag = arith::ConstantIndexOp::create(b, bodyLoc, diagonal);
    Value sum = arith::AddIOp::create(b, bodyLoc, row, diag);
    Value keep;
    if (Upper) {
      keep = arith::CmpIOp::create(
          b, bodyLoc, arith::CmpIPredicate::sge, col, sum);
    } else {
      keep = arith::CmpIOp::create(
          b, bodyLoc, arith::CmpIPredicate::sle, col, sum);
    }
    Value kept = arith::SelectOp::create(b, bodyLoc, keep, args[0], zero);
    linalg::YieldOp::create(b, bodyLoc, kept);
  });
  return generic.getResult(0);
}

struct SfTriuOpLowering : public OpRewritePattern<sf::TriuOp> {
  using OpRewritePattern<sf::TriuOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::TriuOp op, PatternRewriter &rewriter) const override {
    auto inType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    llvm::errs() << "  [SfTriu] lowering " << inType << " -> " << outType << "\n";
    if (!inType || !outType) return failure();
    int64_t diag = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("diagonal"))
      diag = attr.getInt();
    Value result = lowerTriangular<true>(op.getInput(), outType, diag, rewriter);
    if (!result) return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SfTrilOpLowering : public OpRewritePattern<sf::TrilOp> {
  using OpRewritePattern<sf::TrilOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::TrilOp op, PatternRewriter &rewriter) const override {
    auto inType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    llvm::errs() << "  [SfTril] lowering " << inType << " -> " << outType << "\n";
    if (!inType || !outType) return failure();
    int64_t diag = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("diagonal"))
      diag = attr.getInt();
    Value result = lowerTriangular<false>(op.getInput(), outType, diag, rewriter);
    if (!result) return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

// MaskedFill → linalg.generic: output = mask ? value : input.
// The value is a scalar tensor (rank 0 or rank 1 of size 1); it is extracted
// and cast to the input element type before the generic body.
struct SfMaskedFillOpLowering : public OpRewritePattern<sf::MaskedFillOp> {
  using OpRewritePattern<sf::MaskedFillOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::MaskedFillOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Value mask = op.getMask();
    Value value = op.getValue();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    llvm::errs() << "  [SfMaskedFill] lowering " << inType << " mask " << maskType << " -> " << outType << "\n";
    if (!inType || !maskType || !outType) return failure();
    int64_t outRank = outType.getRank();
    int64_t maskRank = maskType.getRank();

    // Extract the scalar fill value.
    auto valueType = dyn_cast<RankedTensorType>(value.getType());
    if (!valueType) return failure();
    SmallVector<Value> valueIdx(valueType.getRank(),
        arith::ConstantIndexOp::create(rewriter, loc, 0));
    Value scalar = tensor::ExtractOp::create(
        rewriter, loc, valueType.getElementType(), value, valueIdx);
    auto outElt = outType.getElementType();
    if (scalar.getType() != outElt) {
      if (isa<FloatType>(outElt) && isa<IntegerType>(scalar.getType())) {
        scalar = arith::SIToFPOp::create(rewriter, loc, outElt, scalar);
      } else if (isa<FloatType>(outElt) && isa<FloatType>(scalar.getType())) {
        auto src = cast<FloatType>(scalar.getType());
        auto dst = cast<FloatType>(outElt);
        if (src.getWidth() < dst.getWidth())
          scalar = arith::ExtFOp::create(rewriter, loc, outElt, scalar);
        else if (src.getWidth() > dst.getWidth())
          scalar = arith::TruncFOp::create(rewriter, loc, outElt, scalar);
        else
          return failure();
      } else if (isa<IntegerType>(outElt) && isa<FloatType>(scalar.getType())) {
        scalar = arith::FPToSIOp::create(rewriter, loc, outElt, scalar);
      } else {
        return failure();
      }
    }

    Value empty = makeEmpty(rewriter, loc, outType, {input});
    if (!empty) return failure();

    auto inputMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());
    auto maskMap = broadcastMap(outRank, maskRank, rewriter.getContext(),
                                maskType.getShape());
    auto outMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());
    SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, outType,
        ValueRange{input, mask}, ValueRange{empty},
        {inputMap, maskMap, outMap}, iters);
    populateBody(generic, rewriter, [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      Value maskVal = args[1];
      Value cond;
      auto maskElt = maskVal.getType();
      if (maskElt.isInteger(1)) {
        cond = maskVal;
      } else if (isa<FloatType>(maskElt)) {
        Value zero = arith::ConstantOp::create(
            b, bodyLoc, maskElt, b.getFloatAttr(maskElt, 0.0));
        cond = arith::CmpFOp::create(
            b, bodyLoc, arith::CmpFPredicate::ONE, maskVal, zero);
      } else if (isa<IntegerType>(maskElt)) {
        Value zero = arith::ConstantOp::create(
            b, bodyLoc, maskElt, b.getIntegerAttr(maskElt, 0));
        cond = arith::CmpIOp::create(
            b, bodyLoc, arith::CmpIPredicate::ne, maskVal, zero);
      } else {
        return;
      }
      Value selected = arith::SelectOp::create(b, bodyLoc, cond, scalar, args[0]);
      linalg::YieldOp::create(b, bodyLoc, selected);
    });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

// Eq/Ne compare operations: same-shape elementwise compare, output f32.
static Value lowerEqNe(
    Value lhs, Value rhs, RankedTensorType outType,
    bool isEq, PatternRewriter &rewriter) {
  auto loc = rewriter.getUnknownLoc();
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType) return Value();
  int64_t outRank = outType.getRank();
  auto f32 = rewriter.getF32Type();
  Value empty = makeEmpty(rewriter, loc, outType, {lhs, rhs});
  if (!empty) return Value();
  auto ctx = rewriter.getContext();
  auto lhsMap = broadcastMap(outRank, lhsType.getRank(), ctx, lhsType.getShape());
  auto rhsMap = broadcastMap(outRank, rhsType.getRank(), ctx, rhsType.getShape());
  auto outMap = AffineMap::getMultiDimIdentityMap(outRank, ctx);
  SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
  auto generic = linalg::GenericOp::create(
      rewriter, loc, outType, ValueRange{lhs, rhs}, ValueRange{empty},
      {lhsMap, rhsMap, outMap}, iters);
  populateBody(generic, rewriter, [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
    Value cmp;
    auto lt = args[0].getType();
    auto rt = args[1].getType();
    if (isa<FloatType>(lt) && isa<FloatType>(rt)) {
      cmp = arith::CmpFOp::create(
          b, bodyLoc,
          isEq ? arith::CmpFPredicate::OEQ : arith::CmpFPredicate::ONE,
          args[0], args[1]);
    } else {
      cmp = arith::CmpIOp::create(
          b, bodyLoc,
          isEq ? arith::CmpIPredicate::eq : arith::CmpIPredicate::ne,
          args[0], args[1]);
    }
    Value result = arith::UIToFPOp::create(b, bodyLoc, f32, cmp);
    linalg::YieldOp::create(b, bodyLoc, result);
  });
  return generic.getResult(0);
}

struct SfEqOpLowering : public OpRewritePattern<sf::EqOp> {
  using OpRewritePattern<sf::EqOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::EqOp op, PatternRewriter &rewriter) const override {
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    llvm::errs() << "  [SfEq] lowering -> " << outType << "\n";
    if (!outType) return failure();
    Value result = lowerEqNe(op.getLhs(), op.getRhs(), outType, true, rewriter);
    if (!result) return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SfNeOpLowering : public OpRewritePattern<sf::NeOp> {
  using OpRewritePattern<sf::NeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::NeOp op, PatternRewriter &rewriter) const override {
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    llvm::errs() << "  [SfNe] lowering -> " << outType << "\n";
    if (!outType) return failure();
    Value result = lowerEqNe(op.getLhs(), op.getRhs(), outType, false, rewriter);
    if (!result) return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerComparePatterns(RewritePatternSet &patterns) {
  patterns.add<SfLeOpLowering, SfLogicalAndOpLowering,
               SfTriuOpLowering, SfTrilOpLowering,
               SfMaskedFillOpLowering, SfEqOpLowering,
               SfNeOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
