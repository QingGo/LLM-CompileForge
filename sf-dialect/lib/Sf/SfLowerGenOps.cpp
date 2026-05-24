#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace {

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
          extracted = tensor::ExtractOp::create(rewriter, loc, operand, ValueRange{});
        } else if (operandTy && operandTy.getRank() > 0) {
          SmallVector<Value> indices(operandTy.getRank(),
              arith::ConstantIndexOp::create(rewriter, loc, 0));
          extracted = tensor::ExtractOp::create(rewriter, loc, operand, indices);
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
    if (!isa<FloatType>(elt)) return failure();
    Value oneVal = arith::ConstantOp::create(rewriter, loc, elt,
        rewriter.getFloatAttr(elt, 1.0));
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
    // Override non-float output to f32 — arange is used for positional
    // encodings which expect float tensor values.
    [[maybe_unused]] bool outputWasPromoted = false;
    if (!isa<FloatType>(eltType)) {
      eltType = rewriter.getF32Type();
      outType = RankedTensorType::get(outType.getShape(), eltType);
      outputWasPromoted = true;
    }

    // Extract first element from input and cast to index type
    auto inType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    SmallVector<Value> zeroIdx;
    if (inType) for (int64_t _i = 0; _i < inType.getRank(); ++_i)
      zeroIdx.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
    Value scalarVal = tensor::ExtractOp::create(rewriter, loc, input, zeroIdx);
    auto scalarType = scalarVal.getType();
    Value nIdx;
    if (scalarType.isInteger(64)) {
      // Input already i64 → direct index cast
      nIdx = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), scalarVal);
    } else if (isa<FloatType>(scalarType)) {
      // Input is f32 → fptoui + index cast
      Value nI64 = arith::FPToUIOp::create(rewriter, loc, rewriter.getI64Type(), scalarVal);
      nIdx = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), nI64);
    } else {
      llvm::errs() << "  [sf.arange] unsupported input type: " << scalarType << "\n";
      return failure();
    }

    // Create empty tensor with dynamic output type.
    // Always use tensor<?xf32> even when outType is tensor<1xf32>, because
    // the arange length depends on the input VALUE at runtime (not its type).
    // Using the declared static type (e.g. tensor<1xf32>) causes canonicalize
    // to specialize to the wrong concrete size, creating shape mismatches.
    SmallVector<int64_t> dynShape = {ShapedType::kDynamic};
    Value rawEmpty = tensor::EmptyOp::create(rewriter, loc, dynShape, eltType, ValueRange{nIdx});

    // Initialize with zeros — tensor::EmptyOp produces uninitialized memory,
    // and the scf.for loop only fills positions [0, nIdx), leaving gaps
    // if the loop doesn't converge or if downstream uses uninitialized elements
    // before the loop completes. Fill with 0.0 to prevent NaN propagation.
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

    // scf.for %i = 0 to N
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto forOp = scf::ForOp::create(rewriter, loc, c0, nIdx, c1, empty);
    Value iv = forOp.getInductionVar();

    rewriter.setInsertionPointToStart(forOp.getBody());
    // Region iter arg (not init value) — required for bufferization correctness
    Value iterArg = forOp.getBody()->getArgument(1);
    Value ivI64 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), iv);
    Value outVal;
    if (eltType.isInteger(64)) {
      outVal = tensor::InsertOp::create(rewriter, loc, iterArg.getType(), ivI64, iterArg, iv);
    } else if (isa<FloatType>(eltType)) {
      Value ivF32 = arith::UIToFPOp::create(rewriter, loc, eltType, ivI64);
      outVal = tensor::InsertOp::create(rewriter, loc, iterArg.getType(), ivF32, iterArg, iv);
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
