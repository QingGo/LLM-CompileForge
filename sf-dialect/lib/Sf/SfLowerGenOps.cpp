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
    if (!isa<ShapedType>(rt)) return failure();
    auto shapedType = cast<ShapedType>(rt);
    auto eltType = shapedType.getElementType();
    if (!isa<FloatType>(eltType)) return failure();

    auto dynShape = op.getDynShape();
    if (!dynShape.empty()) {
      // Dynamic shape from scalar tensor operands.
      // Collect all: input + dyn_shape together define the output rank.
      SmallVector<Value> allInputs;
      allInputs.push_back(op.getInput());
      allInputs.append(dynShape.begin(), dynShape.end());

      size_t numDims = allInputs.size();
      SmallVector<int64_t> shape(numDims, ShapedType::kDynamic);
      auto tensorType = RankedTensorType::get(shape, eltType);

      // Extract scalar f32 from each operand → i64 → index for tensor.empty
      // Operands can be 0D (tensor<f32>) or 1D (tensor<1xf32>).
      SmallVector<Value> dynSizes;
      auto idxType = rewriter.getIndexType();
      for (auto operand : allInputs) {
        Value extracted;
        auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
        if (operandTy && operandTy.getRank() == 0) {
          extracted = tensor::ExtractOp::create(rewriter, loc,
              operandTy.getElementType(), operand, ValueRange{});
        } else if (operandTy && operandTy.getRank() > 0) {
          SmallVector<Value> indices(operandTy.getRank(),
              arith::ConstantIndexOp::create(rewriter, loc, 0));
          extracted = tensor::ExtractOp::create(rewriter, loc,
              operandTy.getElementType(), operand, indices);
        } else {
          return failure();
        }
        Value i64Val;
        if (isa<FloatType>(extracted.getType())) {
          i64Val = arith::FPToUIOp::create(rewriter, loc, rewriter.getI64Type(), extracted);
        } else if (isa<IntegerType>(extracted.getType())) {
          i64Val = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), extracted);
        } else {
          return failure();
        }
        Value idx = arith::IndexCastOp::create(rewriter, loc, idxType, i64Val);
        dynSizes.push_back(idx);
      }

      Value empty = tensor::EmptyOp::create(rewriter, loc, tensorType, dynSizes);
      if (!empty) return failure();

      Value oneVal = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, 1.0));
      rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{oneVal}, ValueRange{empty});
      return success();
    }

    // Default path: no dyn_shape, copy shape from input tensor
    Value empty = makeEmpty(rewriter, loc, rt, {op.getInput()});
    if (!empty) return failure();
    Value oneVal = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, 1.0));
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{oneVal}, ValueRange{empty});
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

//===----------------------------------------------------------------------===//
// Arange → tensor.empty + scf.for fill
//===----------------------------------------------------------------------===//

struct SfArangeOpLowering : public OpRewritePattern<sf::ArangeOp> {
  using OpRewritePattern<sf::ArangeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::ArangeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Type rt = op.getResult().getType();
    llvm::errs() << "  [sf.arange] rt=" << rt << " operands=" << op.getOperation()->getNumOperands() << "\n";
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

    // Extract first element from input — this is the START value
    auto inType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    SmallVector<Value> zeroIdx;
    if (inType) for (int64_t _i = 0; _i < inType.getRank(); ++_i)
      zeroIdx.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
    Value startVal = tensor::ExtractOp::create(rewriter, loc,
        getElementTypeOrSelf(input.getType()), input, zeroIdx);
    auto startType = startVal.getType();

    // Determine loop bound (output size) and dynamic size for EmptyOp
    Value loopBound;
    SmallVector<int64_t> outShape;
    SmallVector<Value> dynSizes;
    int64_t staticOutDim = outType.getDimSize(0);
    if (staticOutDim != ShapedType::kDynamic) {
      // Static output: use declared size
      loopBound = arith::ConstantIndexOp::create(rewriter, loc, staticOutDim);
      outShape = {staticOutDim};
    } else {
      // Dynamic output: use dyn_shape[0] if available; otherwise query tensor.dim
      if (!op.getDynShape().empty()) {
        Value shapeOp = op.getDynShape()[0];
        // Extract scalar from tensor<1xi64> shape operand
        Value extracted = tensor::ExtractOp::create(rewriter, loc,
            getElementTypeOrSelf(shapeOp.getType()), shapeOp, zeroIdx);
        loopBound = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), extracted);
      } else {
        // No dyn_shape — create empty tensor first, then query dim at runtime
        outShape = {ShapedType::kDynamic};
        // Use a placeholder size of 1 for EmptyOp; actual loop bound comes from tensor.dim
        dynSizes.push_back(arith::ConstantIndexOp::create(rewriter, loc, 1));
        Value placeholder = tensor::EmptyOp::create(rewriter, loc, outShape, eltType, dynSizes);
        loopBound = rewriter.create<tensor::DimOp>(loc, placeholder, 0);
        // Reset dynSizes for the actual output tensor
        dynSizes.clear();
      }
      outShape = {ShapedType::kDynamic};
      if (dynSizes.empty()) {
        dynSizes.push_back(loopBound);
      }
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
      // startVal (i64 or f32) + ivI64 (i64) → insert at position iv
      Value startI64;
      if (startType.isInteger(64)) {
        startI64 = startVal;
      } else if (isa<FloatType>(startType)) {
        startI64 = arith::FPToSIOp::create(rewriter, loc, rewriter.getI64Type(), startVal);
      } else {
        return failure();
      }
      Value valI64 = arith::AddIOp::create(rewriter, loc, startI64, ivI64);
      outVal = tensor::InsertOp::create(rewriter, loc, iterArg.getType(), valI64, iterArg, iv);
    } else if (isa<FloatType>(eltType)) {
      // startVal (f32) + ivF32 (f32) → insert at position iv
      Value startF32;
      if (isa<FloatType>(startType)) {
        startF32 = startVal;
      } else if (startType.isInteger(64)) {
        startF32 = arith::SIToFPOp::create(rewriter, loc, eltType, startVal);
      } else {
        return failure();
      }
      Value ivF32 = arith::SIToFPOp::create(rewriter, loc, eltType, ivI64);
      Value valF32 = arith::AddFOp::create(rewriter, loc, startF32, ivF32);
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
               SfArangeOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
