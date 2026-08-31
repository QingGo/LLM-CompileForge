#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace mlir::sf {

// OnesLike → linalg.fill(1.0)
// When dyn_shape operands are present (e.g., from aten.ones with symbolic
// shapes), each operand is a scalar tensor<f32> providing one dynamic
// dimension.  Extract their values and build a correctly-shaped tensor.
// When dyn_shape is empty, fall back to copying the input tensor's shape.
struct SfOnesLikeOpLowering : public OpRewritePattern<sf::OnesLikeOp> {
  using OpRewritePattern<sf::OnesLikeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::OnesLikeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type rt = op.getResult().getType();
    llvm::errs() << "  [SfOnesLike] lowering " << rt << "\n";
    if (!isa<ShapedType>(rt)) return failure();
    auto shapedType = cast<ShapedType>(rt);
    auto eltType = shapedType.getElementType();
    if (!isa<FloatType>(eltType) && !isa<IntegerType>(eltType)) return failure();

    auto operands = op.getOperands();

    // Builder for the one-value constant in the output element type.
    Value oneVal;
    if (isa<FloatType>(eltType)) {
      oneVal = arith::ConstantOp::create(
          rewriter, loc, eltType, rewriter.getFloatAttr(eltType, 1.0));
    } else {
      oneVal = arith::ConstantOp::create(
          rewriter, loc, eltType, rewriter.getIntegerAttr(eltType, 1));
    }

    if (!operands.empty() && operands.size() > 1) {
      // Dynamic shape from scalar tensor operands.  Each operand is a
      // scalar tensor providing one dimension size.
      size_t numDims = operands.size();
      SmallVector<int64_t> shape(numDims, ShapedType::kDynamic);
      auto tensorType = RankedTensorType::get(shape, eltType);

      SmallVector<Value> dynSizes;
      auto idxType = rewriter.getIndexType();
      for (auto operand : operands) {
        auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
        if (!operandTy) return failure();
        SmallVector<Value> indices(operandTy.getRank(),
            arith::ConstantIndexOp::create(rewriter, loc, 0));
        Value extracted = tensor::ExtractOp::create(
            rewriter, loc, operandTy.getElementType(), operand, indices);
        Value i64Val;
        if (isa<FloatType>(extracted.getType())) {
          i64Val = arith::FPToUIOp::create(
              rewriter, loc, rewriter.getI64Type(), extracted);
        } else if (isa<IntegerType>(extracted.getType())) {
          i64Val = arith::IndexCastOp::create(
              rewriter, loc, rewriter.getI64Type(), extracted);
        } else {
          return failure();
        }
        Value idx = arith::IndexCastOp::create(rewriter, loc, idxType, i64Val);
        dynSizes.push_back(idx);
      }

      Value empty = tensor::EmptyOp::create(rewriter, loc, tensorType, dynSizes);
      if (!empty) return failure();
      rewriter.replaceOpWithNewOp<linalg::FillOp>(
          op, ValueRange{oneVal}, ValueRange{empty});
      return success();
    }

    // Zero-operand form (`torch.ones` with a static shape): fill the
    // declared result type with ones.  One-operand form (`ones_like`):
    // copy the input tensor shape.
    Value empty;
    if (operands.size() == 1) {
      empty = makeEmpty(rewriter, loc, rt, {operands[0]});
    } else {
      for (int64_t i = 0; i < shapedType.getRank(); ++i) {
        if (shapedType.isDynamicDim(i)) {
          return failure();
        }
      }
      empty = tensor::EmptyOp::create(rewriter, loc, shapedType, ValueRange{});
    }
    if (!empty) return failure();
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        op, ValueRange{oneVal}, ValueRange{empty});
    return success();
  }
};

// NewOnes → tensor.empty + linalg.fill(1.0)
struct SfNewOnesOpLowering : public OpRewritePattern<sf::NewOnesOp> {
  using OpRewritePattern<sf::NewOnesOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::NewOnesOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();
    Value empty = makeEmpty(rewriter, loc, rt, {op.getInput()});
    if (!empty) return failure();
    auto elt = getElementTypeOrSelf(rt);
    Value oneVal;
    if (isa<FloatType>(elt))
      oneVal = arith::ConstantOp::create(rewriter, loc, elt,
          rewriter.getFloatAttr(elt, 1.0));
    else if (auto iTy = dyn_cast<IntegerType>(elt))
      oneVal = arith::ConstantOp::create(rewriter, loc, elt,
          rewriter.getIntegerAttr(elt, 1));
    else
      return failure();
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{oneVal}, ValueRange{empty});
    return success();
  }
};

// Zeros / ZerosLike → linalg.fill with zero.
struct SfZerosOpLowering : public OpRewritePattern<sf::ZerosOp> {
  using OpRewritePattern<sf::ZerosOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::ZerosOp op, PatternRewriter &rewriter) const override {
    auto rt = op.getResult().getType();
    auto shaped = dyn_cast<ShapedType>(rt);
    llvm::errs() << "  [SfZeros] lowering " << rt << "\n";
    if (!shaped) return failure();
    auto elt = shaped.getElementType();
    if (!isa<FloatType>(elt) && !isa<IntegerType>(elt)) return failure();
    auto loc = op.getLoc();
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < shaped.getRank(); ++i) {
      if (shaped.isDynamicDim(i)) return failure();
    }
    Value empty = tensor::EmptyOp::create(rewriter, loc, shaped, dynSizes);
    if (!empty) return failure();
    Value zero;
    if (isa<FloatType>(elt)) {
      zero = arith::ConstantOp::create(rewriter, loc, elt,
          rewriter.getFloatAttr(elt, 0.0f));
    } else {
      zero = arith::ConstantOp::create(rewriter, loc, elt,
          rewriter.getIntegerAttr(elt, 0));
    }
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{zero}, ValueRange{empty});
    return success();
  }
};

struct SfZerosLikeOpLowering : public OpRewritePattern<sf::ZerosLikeOp> {
  using OpRewritePattern<sf::ZerosLikeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::ZerosLikeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto rt = op.getResult().getType();
    llvm::errs() << "  [SfZerosLike] lowering " << rt << "\n";
    if (!isa<ShapedType>(rt)) return failure();
    auto shaped = cast<ShapedType>(rt);
    auto elt = shaped.getElementType();
    if (!isa<FloatType>(elt) && !isa<IntegerType>(elt)) return failure();
    Value empty = makeEmpty(rewriter, loc, rt, {op.getInput()});
    if (!empty) return failure();
    Value zero;
    if (isa<FloatType>(elt)) {
      zero = arith::ConstantOp::create(rewriter, loc, elt,
          rewriter.getFloatAttr(elt, 0.0f));
    } else {
      zero = arith::ConstantOp::create(rewriter, loc, elt,
          rewriter.getIntegerAttr(elt, 0));
    }
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{zero}, ValueRange{empty});
    return success();
  }
};

// Eye → linalg.generic: 1.0 on the diagonal, 0.0 elsewhere.
struct SfEyeOpLowering : public OpRewritePattern<sf::EyeOp> {
  using OpRewritePattern<sf::EyeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::EyeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    llvm::errs() << "  [SfEye] lowering " << outType << "\n";
    if (!outType || outType.getRank() != 2) return failure();
    auto elt = outType.getElementType();
    if (!isa<FloatType>(elt) && !isa<IntegerType>(elt)) return failure();
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < 2; ++i)
      if (outType.isDynamicDim(i)) return failure();
    Value empty = tensor::EmptyOp::create(rewriter, loc, outType, dynSizes);
    if (!empty) return failure();
    Value zero;
    Value one;
    if (isa<FloatType>(elt)) {
      zero = arith::ConstantOp::create(rewriter, loc, elt,
          rewriter.getFloatAttr(elt, 0.0f));
      one = arith::ConstantOp::create(rewriter, loc, elt,
          rewriter.getFloatAttr(elt, 1.0f));
    } else {
      zero = arith::ConstantOp::create(rewriter, loc, elt,
          rewriter.getIntegerAttr(elt, 0));
      one = arith::ConstantOp::create(rewriter, loc, elt,
          rewriter.getIntegerAttr(elt, 1));
    }
    auto idMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
    SmallVector<utils::IteratorType> iters(2, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, outType, ValueRange{}, ValueRange{empty},
        {idMap}, iters);
    populateBody(generic, rewriter, [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      Value row = linalg::IndexOp::create(b, bodyLoc, 0);
      Value col = linalg::IndexOp::create(b, bodyLoc, 1);
      Value isDiag = arith::CmpIOp::create(
          b, bodyLoc, arith::CmpIPredicate::eq, row, col);
      Value val = arith::SelectOp::create(b, bodyLoc, isDiag, one, zero);
      linalg::YieldOp::create(b, bodyLoc, val);
    });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Arange → tensor.empty + scf.for fill
//
// Contract (see compiler/tests/test_arange_fix.py):
//   sf.arange(start, [size]) produces [start, start+1, ..., start+size-1]
//   in the OUTPUT element type.  The input operand is the START value
//   (scalar tensor<1xT>; torch.arange(start, end) normalizes to
//   start + size = end - start on the compiler side).  The size comes from:
//     1. the static output dim (tensor<Nx...>), else
//     2. dyn_shape[0], else
//     3. legacy fallback (pre-contract artifacts): input value IS the
//        size and start is 0 — matches torch.arange(end).
//===----------------------------------------------------------------------===//

struct SfArangeOpLowering : public OpRewritePattern<sf::ArangeOp> {
  using OpRewritePattern<sf::ArangeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::ArangeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Type rt = op.getResult().getType();
    auto outType = ::mlir::dyn_cast<::mlir::RankedTensorType>(rt);
    if (!outType) return failure();
    if (outType.getRank() == 0) {
      // Scalar arange: not meaningful; just return zero.
      auto eltType = getElementTypeOrSelf(rt);
      Value zero = createSafeConst(rewriter, loc, eltType, 0.0, 0);
      if (!zero) return failure();
      auto empty = tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{}, eltType, ValueRange{});
      rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, zero, empty, ValueRange{});
      return success();
    }
    if (outType.getRank() != 1) return failure();
    auto eltType = getElementTypeOrSelf(rt);

    // Build zero-index vector for scalar extraction
    auto inType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    SmallVector<Value> zeroIdx;
    if (inType) for (int64_t _i = 0; _i < inType.getRank(); ++_i)
      zeroIdx.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));

    // Determine loop bound (output size) and dynamic size for EmptyOp.
    // Priority: static output dim > dyn_shape[0] > legacy input-as-size.
    int64_t staticOutDim = outType.getDimSize(0);
    bool legacyLimit = staticOutDim == ShapedType::kDynamic && op.getDynShape().empty();

    Value inputScalar = tensor::ExtractOp::create(rewriter, loc,
        getElementTypeOrSelf(input.getType()), input, zeroIdx);

    Value loopBound;
    SmallVector<int64_t> outShape;
    SmallVector<Value> dynSizes;
    if (staticOutDim != ShapedType::kDynamic) {
      // Static output: use declared size
      loopBound = arith::ConstantIndexOp::create(rewriter, loc, staticOutDim);
      outShape = {staticOutDim};
    } else if (!op.getDynShape().empty()) {
      // dyn_shape[0] = size (end - start on the compiler side)
      Value shapeOp = op.getDynShape()[0];
      Value extracted = tensor::ExtractOp::create(rewriter, loc,
          getElementTypeOrSelf(shapeOp.getType()), shapeOp, zeroIdx);
      Value sizeI64 = extracted;
      if (isa<FloatType>(extracted.getType()))
        sizeI64 = arith::FPToSIOp::create(rewriter, loc, rewriter.getI64Type(), extracted);
      else if (!extracted.getType().isInteger(64))
        sizeI64 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), extracted);
      loopBound = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), sizeI64);
      outShape = {ShapedType::kDynamic};
      dynSizes.push_back(loopBound);
    } else {
      // Legacy fallback: input value is the size (torch.arange(end)).
      Value sizeVal = inputScalar;
      auto sizeType = sizeVal.getType();
      Value sizeI64;
      if (isa<FloatType>(sizeType)) {
        sizeI64 = arith::FPToSIOp::create(rewriter, loc, rewriter.getI64Type(), sizeVal);
      } else {
        sizeI64 = sizeVal;
      }
      loopBound = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), sizeI64);
      outShape = {ShapedType::kDynamic};
      dynSizes.push_back(loopBound);
    }

    // Start value, converted to the OUTPUT element type.
    // Legacy fallback: start is 0 (input was the limit).  Otherwise the
    // input operand carries the start (i64 or f32 scalar).
    Value startVal;
    if (legacyLimit) {
      startVal = createSafeConst(rewriter, loc, eltType, 0.0, 0);
      if (!startVal) return failure();
    } else if (isa<FloatType>(eltType)) {
      if (isa<FloatType>(inputScalar.getType())) {
        startVal = inputScalar;
      } else {
        startVal = arith::SIToFPOp::create(rewriter, loc, eltType, inputScalar);
      }
    } else if (eltType.isInteger(64)) {
      if (isa<FloatType>(inputScalar.getType())) {
        startVal = arith::FPToSIOp::create(rewriter, loc, rewriter.getI64Type(), inputScalar);
      } else {
        startVal = inputScalar;
      }
    } else {
      return failure();
    }

    Value rawEmpty = tensor::EmptyOp::create(rewriter, loc, outShape, eltType, dynSizes);

    // Initialize with zeros
    Value zeroInit;
    if (isa<FloatType>(eltType)) {
      zeroInit = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, 0.0));
    } else {
      zeroInit = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getIntegerAttr(eltType, 0));
    }
    auto fillOp = linalg::FillOp::create(rewriter, loc, ValueRange{zeroInit}, ValueRange{rawEmpty});
    Value empty = fillOp.getResult(0);

    // scf.for %i = 0 to outputSize: insert startVal + i at position i
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto forOp = scf::ForOp::create(rewriter, loc, c0, loopBound, c1, empty);
    Value iv = forOp.getInductionVar();

    rewriter.setInsertionPointToStart(forOp.getBody());
    Value iterArg = forOp.getBody()->getArgument(1);
    Value ivI64 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), iv);
    Value outVal;
    if (eltType.isInteger(64)) {
      Value valI64 = arith::AddIOp::create(rewriter, loc, startVal, ivI64);
      outVal = tensor::InsertOp::create(rewriter, loc, iterArg.getType(), valI64, iterArg, iv);
    } else if (isa<FloatType>(eltType)) {
      Value ivF32 = arith::SIToFPOp::create(rewriter, loc, eltType, ivI64);
      Value valF32 = arith::AddFOp::create(rewriter, loc, startVal, ivF32);
      outVal = tensor::InsertOp::create(rewriter, loc, iterArg.getType(), valF32, iterArg, iv);
    } else {
      return failure();
    }
    scf::YieldOp::create(rewriter, loc, outVal);

    rewriter.setInsertionPointAfter(forOp);
    rewriter.replaceOp(op, forOp.getResult(0));
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerGenOpsPatterns(RewritePatternSet &patterns) {
  patterns.add<SfOnesLikeOpLowering, SfNewOnesOpLowering,
               SfZerosOpLowering, SfZerosLikeOpLowering,
               SfEyeOpLowering, SfArangeOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
