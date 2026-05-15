#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"
#include "Sf/SfPasses.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

static Value makeEmpty(OpBuilder &b, Location loc, Type t, ValueRange inputs) {
  auto shaped = dyn_cast<ShapedType>(t);
  if (!shaped) return Value();
  SmallVector<Value> dynSizes;
  SmallVector<bool> filled(shaped.getRank(), false);
  if (!inputs.empty()) {
    auto idxType = b.getIndexType();
    for (auto input : inputs) {
      if (auto inType = dyn_cast<RankedTensorType>(input.getType())) {
        for (int64_t i = 0; i < std::min((int64_t)shaped.getRank(), inType.getRank()); ++i) {
          if (shaped.isDynamicDim(i) && inType.isDynamicDim(i) && !filled[i]) {
            dynSizes.push_back(b.create<tensor::DimOp>(loc, input,
                b.create<arith::ConstantOp>(loc, idxType, b.getIndexAttr(i))));
            filled[i] = true;
          }
        }
      }
    }
    // For dynamic dims not filled from inputs, use 0 (will be replaced)
    for (int64_t i = 0; i < (int64_t)shaped.getRank(); ++i)
      if (shaped.isDynamicDim(i) && !filled[i])
        dynSizes.push_back(b.create<arith::ConstantIndexOp>(loc, 0));
  }
  return b.create<tensor::EmptyOp>(loc, shaped, dynSizes);
}

// Create tensor.empty with support for dynamic dims (kDynamic in shape).
// Falls back to makeEmpty when input is available.
static Value makeEmptyFromShape(OpBuilder &b, Location loc,
                                 ArrayRef<int64_t> shape, Type eltType) {
  auto tensorType = RankedTensorType::get(shape, eltType);
  SmallVector<Value> dynSizes;
  auto idxType = b.getIndexType();
  for (int64_t i = 0; i < (int64_t)shape.size(); ++i)
    if (ShapedType::isDynamic(shape[i]))
      dynSizes.push_back(b.create<arith::ConstantOp>(loc, idxType,
          b.getIndexAttr(shape[i])));  // will be replaced by proper dim op
  // For now, all dynamic dims get 0 as placeholder (must be overridden)
  return b.create<tensor::EmptyOp>(loc, tensorType, dynSizes);
}

static SmallVector<AffineMap> identityMaps(unsigned rank, unsigned count, MLIRContext *ctx) {
  SmallVector<AffineMap> maps;
  for (unsigned i = 0; i < count; ++i)
    maps.push_back(AffineMap::getMultiDimIdentityMap(rank, ctx));
  return maps;
}

// Build an affine map aligning trailing dims: given an operand of rank `operandRank`
// and an output of rank `outRank`, project away leading dims so the operand's
// dimensions align with the trailing `operandRank` dimensions of the output.
// E.g. outRank=3, operandRank=tensor<f32>(0) → empty map (broadcast scalar)
//      outRank=3, operandRank=tensor<768xf32>(1) → affine_map<(d0,d1,d2) -> (d2)>
//      outRank=3, operandRank=tensor<?x768xf32>(2) → affine_map<(d0,d1,d2) -> (d1,d2)>
//      outRank=2, operandRank=tensor<1xf32>(1) → affine_map<(d0,d1) -> (d1)>
static AffineMap broadcastMap(unsigned loopRank, unsigned operandRank, MLIRContext *ctx,
                               ArrayRef<int64_t> operandShape = {}) {
  if (operandRank == 0)
    return AffineMap::get(loopRank, 0, {}, ctx);
  // The map has 'operandRank' results, each referencing loop dims (d0..d{loopRank-1})
  // or constants. Trailing dims align: operand dim i maps to loop dim i (or constant 0
  // if size 1). Leading excess dims (if operandRank > loopRank) map to constants.
  SmallVector<AffineExpr> exprs;
  int64_t leadingExcess = (operandRank > loopRank) ? (operandRank - loopRank) : 0;
  // Leading squeezed dims: output dimension 0..leadingExcess-1 are constants
  for (int64_t i = 0; i < leadingExcess; ++i)
    exprs.push_back(getAffineConstantExpr(0, ctx));
  // Trailing dims: align with loop dims
  int64_t loopStart = (loopRank > operandRank) ? (loopRank - operandRank) : 0;
  for (int64_t i = leadingExcess; i < (int64_t)operandRank; ++i) {
    int64_t loopIdx = loopStart + (i - leadingExcess);
    int64_t shapeIdx = i;
    bool isSizeOne = (shapeIdx < (int64_t)operandShape.size() && operandShape[shapeIdx] == 1);
    if (isSizeOne)
      exprs.push_back(getAffineConstantExpr(0, ctx));
    else
      exprs.push_back(getAffineDimExpr(loopIdx, ctx));
  }
  return AffineMap::get(loopRank, 0, exprs, ctx);
}

static void populateBody(linalg::GenericOp op, function_ref<void(OpBuilder &, Location, ValueRange)> f) {
  OpBuilder b(op.getContext());
  Block *body = b.createBlock(&op.getRegion(), {});
  auto shaped = cast<ShapedType>(op->getResult(0).getType());
  auto eltTy = shaped.getElementType();
  unsigned numInputs = op.getNumDpsInputs();
  unsigned numOutputs = op.getNumDpsInits();
  for (unsigned i = 0; i < numInputs; ++i)
    body->addArgument(eltTy, op.getLoc());
  for (unsigned i = 0; i < numOutputs; ++i)
    body->addArgument(eltTy, op.getLoc());
  b.setInsertionPointToEnd(body);
  f(b, op.getLoc(), body->getArguments());
}

// Binary lowering with broadcast support via affine maps
template <typename SfOpTy, typename ArithOpTy>
struct SfBinaryLowering : public OpConversionPattern<SfOpTy> {
  using OpConversionPattern<SfOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(SfOpTy op, typename SfOpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    if (!isa<ShapedType>(resultType)) return failure();
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    auto rank = cast<ShapedType>(resultType).getRank();
    auto lhsRank = lhsType.getRank();
    auto rhsRank = rhsType.getRank();
    llvm::errs() << "  [SfBinary] type=" << SfOpTy::getOperationName() << " lhs=" << lhsType << " rhs=" << rhsType << " out=" << resultType << "\n";
    Value empty = makeEmpty(rewriter, loc, resultType, {lhs});
    if (!empty) { llvm::errs() << "  [SfBinary] makeEmpty failed\n"; return failure(); }

    // Handle rank mismatch: squeeze leading size-1 dims of higher-rank operands
    // so both operands match the output rank.
    auto squeezeToRank = [&](Value val, int64_t valRank) -> Value {
      if (valRank <= rank) return val;
      auto valType = cast<RankedTensorType>(val.getType());
      int64_t squeeze = valRank - rank;
      for (int64_t i = 0; i < squeeze; ++i)
        if (valType.getDimSize(i) != 1) { llvm::errs() << "  [squeeze] cannot squeeze dim " << i << " size " << valType.getDimSize(i) << "\n"; return Value(); }
      auto outShape = cast<RankedTensorType>(resultType).getShape();
      auto squeezedType = RankedTensorType::get(outShape, valType.getElementType());
      SmallVector<Value> shapeVals;
      for (int64_t i = 0; i < (int64_t)outShape.size(); ++i) {
        if (ShapedType::isDynamic(outShape[i]))
          shapeVals.push_back(rewriter.create<tensor::DimOp>(loc, val, squeeze + i));
        else
          shapeVals.push_back(rewriter.create<arith::ConstantIndexOp>(loc, outShape[i]));
      }
      auto shapeType = RankedTensorType::get({(int64_t)shapeVals.size()}, rewriter.getIndexType());
      Value shape;
      if (shapeVals.empty())
        shape = rewriter.create<tensor::EmptyOp>(loc, shapeType, ValueRange{});
      else
        shape = rewriter.create<tensor::FromElementsOp>(loc, shapeType, shapeVals);
      auto reshaped = rewriter.create<tensor::ReshapeOp>(loc, squeezedType, val, shape);
      llvm::errs() << "  [squeeze] " << valRank << "->" << rank << " OK\n";
      return reshaped.getResult();
    };
    Value newLhs = squeezeToRank(lhs, lhsRank);
    Value newRhs = squeezeToRank(rhs, rhsRank);
    if (!newLhs || !newRhs) { llvm::errs() << "  [squeeze] FAILED for sf binary\n"; return failure(); }
    lhs = newLhs; rhs = newRhs;
    lhsRank = cast<RankedTensorType>(lhs.getType()).getRank();
    rhsRank = cast<RankedTensorType>(rhs.getType()).getRank();

    // Build broadcast-aware affine maps for each operand
    auto lhsShaped = cast<RankedTensorType>(lhs.getType());
    auto rhsShaped = cast<RankedTensorType>(rhs.getType());
    auto lhsMap = broadcastMap(rank, lhsRank, rewriter.getContext(), lhsShaped.getShape());
    auto rhsMap = broadcastMap(rank, rhsRank, rewriter.getContext(), rhsShaped.getShape());
    auto outMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());

    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultType, ValueRange{lhs, rhs}, empty,
        {lhsMap, rhsMap, outMap}, iterTypes);
    populateBody(generic, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value v = b.create<ArithOpTy>(loc, args[0], args[1]);
      b.create<linalg::YieldOp>(loc, v);
    });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

// Relu lowering
struct ReluLowering : public OpConversionPattern<sf::ReluOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    if (!isa<ShapedType>(resultType)) return failure();
    auto loc = op.getLoc();
    Value empty = makeEmpty(rewriter, loc, resultType, {adaptor.getInput()});
    if (!empty) return failure();
    auto rank = cast<ShapedType>(resultType).getRank();
    auto eltType = getElementTypeOrSelf(resultType);

    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultType, adaptor.getInput(), empty,
        identityMaps(rank, 2, rewriter.getContext()), iterTypes);
    populateBody(generic, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value zero = b.create<arith::ConstantOp>(loc, eltType,
          b.getFloatAttr(eltType, 0.0));
      Value v = b.create<arith::MaxNumFOp>(loc, args[0], zero);
      b.create<linalg::YieldOp>(loc, v);
    });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

// Identity → passthrough; handle type mismatches by inserting proper cast
struct IdentityLowering : public OpConversionPattern<sf::IdentityOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::IdentityOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getInput();
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
            v = b.create<arith::UIToFPOp>(bodyLoc, outElt, args[0]);
          else if (isa<FloatType>(inElt) && isa<IntegerType>(outElt))
            v = b.create<arith::FPToUIOp>(bodyLoc, outElt, args[0]);
          else
            v = args[0];
          b.create<linalg::YieldOp>(bodyLoc, v);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};



//===----------------------------------------------------------------------===//
// Embedding survives for now (true gather requires index-based lookup which is
// slow for large vocab. The linalg.generic pattern was evaluated and discarded
// because linalg doesn't support non-affine indexing (gather). A proper
// implementation would use tensor.extract or scf.for loops, but for now
// the Rust runtime handles embedding on the model level.)
//===----------------------------------------------------------------------===//

template <typename SfOpTy>
struct SfActivationOpLowering : public OpConversionPattern<SfOpTy> {
  using OpConversionPattern<SfOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(SfOpTy op, typename SfOpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    if (!isa<ShapedType>(resultType)) return failure();
    auto loc = op.getLoc();
    Value empty = makeEmpty(rewriter, loc, resultType, {adaptor.getInput()});
    if (!empty) return failure();
    auto rank = cast<ShapedType>(resultType).getRank();
    auto eltType = getElementTypeOrSelf(resultType);

    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultType, adaptor.getInput(), empty,
        identityMaps(rank, 2, rewriter.getContext()), iterTypes);
    populateBody(generic, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value val;
      StringRef opName = SfOpTy::getOperationName();
      if (opName == "sf.gelu") {
        Value half = b.create<arith::ConstantOp>(loc, eltType, b.getFloatAttr(eltType, 0.5));
        Value one = b.create<arith::ConstantOp>(loc, eltType, b.getFloatAttr(eltType, 1.0));
        Value c1 = b.create<arith::ConstantOp>(loc, eltType, b.getFloatAttr(eltType, 0.7978845608));
        Value c2 = b.create<arith::ConstantOp>(loc, eltType, b.getFloatAttr(eltType, 0.044715));
        Value x = args[0]; Value x3 = b.create<arith::MulFOp>(loc, x, x);
        x3 = b.create<arith::MulFOp>(loc, x3, x);
        Value i1 = b.create<arith::MulFOp>(loc, c2, x3);
        Value i2 = b.create<arith::AddFOp>(loc, x, i1);
        Value sc = b.create<arith::MulFOp>(loc, c1, i2);
        Value th = b.create<math::TanhOp>(loc, sc);
        Value p1 = b.create<arith::AddFOp>(loc, one, th);
        Value hx = b.create<arith::MulFOp>(loc, half, x);
        val = b.create<arith::MulFOp>(loc, hx, p1);
      } else if (opName == "sf.silu") {
        Value x = args[0]; Value neg = b.create<arith::NegFOp>(loc, x);
        Value exp = b.create<math::ExpOp>(loc, neg);
        Value one = b.create<arith::ConstantOp>(loc, eltType, b.getFloatAttr(eltType, 1.0));
        Value denom = b.create<arith::AddFOp>(loc, one, exp);
        Value sig = b.create<arith::DivFOp>(loc, one, denom);
        val = b.create<arith::MulFOp>(loc, x, sig);
      } else if (opName == "sf.sigmoid") {
        Value neg = b.create<arith::NegFOp>(loc, args[0]);
        Value exp = b.create<math::ExpOp>(loc, neg);
        Value one = b.create<arith::ConstantOp>(loc, eltType, b.getFloatAttr(eltType, 1.0));
        Value denom = b.create<arith::AddFOp>(loc, one, exp);
        val = b.create<arith::DivFOp>(loc, one, denom);
      } else if (opName == "sf.exp") {
        val = b.create<math::ExpOp>(loc, args[0]);
      } else if (opName == "sf.neg") {
        val = b.create<arith::NegFOp>(loc, args[0]);
      } else if (opName == "sf.tanh") {
        val = b.create<math::TanhOp>(loc, args[0]);
      } else { return; }
      b.create<linalg::YieldOp>(loc, val);
    });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

// Matmul/Linear lowering
struct SfMatmulOpLowering : public OpConversionPattern<sf::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = adaptor.getLhs(), rhs = adaptor.getRhs();
    Type resultType = op.getResult().getType();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    int64_t lhsRank = lhsType.getRank(), rhsRank = rhsType.getRank();

    // Standard 2D matmul: use linalg.matmul
    if (lhsRank == 2 && rhsRank == 2) {
      Value empty = makeEmpty(rewriter, loc, resultType, {lhs});
      if (!empty) return failure();
      auto mo = rewriter.create<linalg::MatmulOp>(loc, resultType,
          ValueRange{lhs, rhs}, empty);
      mo->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({2, 1}));
      rewriter.replaceOp(op, mo.getResult(0));
      return success();
    }

    // Non-2D matmul: use linalg.generic with proper maps.
    // Contract the innermost dim of lhs with the first dim of rhs.
    //   lhs: [d0..d{m-2}, M, K]  rhs: [d0..d{r-2}, K, N]
    //   out: [d0..d{max(m,r)-2}, M, N]
    // We use a loop with (maxRank-1) parallel + 1 reduction iterator.
    auto eltType = lhsType.getElementType();
    int64_t contractDimL = lhsRank - 1;  // K in lhs
    int64_t contractDimR = 0;            // K in rhs
    int64_t outerRank = std::max(lhsRank - 1, rhsRank - 1) + 1; // M + N + batch
    SmallVector<int64_t> outShape;
    SmallVector<Value> dynSizes;
    auto resultRT = cast<RankedTensorType>(resultType);
    for (int64_t i = 0; i < resultRT.getRank(); ++i) {
      outShape.push_back(resultRT.getDimSize(i));
      if (resultRT.isDynamicDim(i))
        dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, resultRT.getRank() > lhsRank ? rhs : lhs, i));
    }
    while ((int64_t)outShape.size() < outerRank - 1)
      outShape.insert(outShape.begin(), 1);
    auto outType = RankedTensorType::get(outShape, eltType);
    Value empty = makeEmpty(rewriter, loc, resultType, {lhs});
    if (!empty) return failure();
    // Build maps: the loop has (outerRank) iterators: [batch..., M, N, K]
    int64_t loopRank = outerRank;  // [d0..d{LO}, K] where LO = outermost non-M/N/K dims
    // Actually: iterators = [batch_dims..., M_pos, N_pos, K_reduction]
    int64_t mPos = outerRank - 3 < 0 ? 0 : outerRank - 2;
    int64_t nPos = outerRank - 1;
    int64_t kPos = outerRank;
    loopRank = outerRank + 1;  // extra dim for K reduction
    
    SmallVector<AffineExpr> lhsExprs, rhsExprs, outExprs;
    auto ctx = rewriter.getContext();
    for (int64_t i = 0; i < lhsRank; ++i) {
      if (i == contractDimL) lhsExprs.push_back(getAffineDimExpr(kPos, ctx)); // K
      else if (i == lhsRank - 2) lhsExprs.push_back(getAffineDimExpr(mPos, ctx)); // M
      else lhsExprs.push_back(getAffineDimExpr(i, ctx)); // batch
    }
    for (int64_t i = 0; i < rhsRank; ++i) {
      if (i == contractDimR) rhsExprs.push_back(getAffineDimExpr(kPos, ctx)); // K
      else if (i == rhsRank - 1) rhsExprs.push_back(getAffineDimExpr(nPos, ctx)); // N  
      else rhsExprs.push_back(getAffineDimExpr(i, ctx)); // batch
    }
    for (int64_t i = 0; i < resultRT.getRank(); ++i) {
      if (i == resultRT.getRank() - 1) outExprs.push_back(getAffineDimExpr(nPos, ctx)); // N
      else if (i == resultRT.getRank() - 2) outExprs.push_back(getAffineDimExpr(mPos, ctx)); // M
      else outExprs.push_back(getAffineDimExpr(i, ctx)); // batch
    }
    SmallVector<utils::IteratorType> matIter(loopRank, utils::IteratorType::parallel);
    matIter[kPos] = utils::IteratorType::reduction;

    auto generic = linalg::GenericOp::create(rewriter, loc, resultType,
        ValueRange{lhs, rhs}, ValueRange{empty},
        {AffineMap::get(loopRank, 0, lhsExprs, ctx),
         AffineMap::get(loopRank, 0, rhsExprs, ctx),
         AffineMap::get(loopRank, 0, outExprs, ctx)}, matIter,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
          Value mul = b.create<arith::MulFOp>(bodyLoc, args[0], args[1]);
          Value add = b.create<arith::AddFOp>(bodyLoc, args[2], mul);
          b.create<linalg::YieldOp>(bodyLoc, add);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

struct SfLinearOpLowering : public OpConversionPattern<sf::LinearOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::LinearOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
  Value input = adaptor.getInput();
  Value weight = adaptor.getWeight();
  Type resultType = op.getResult().getType();
  auto inputType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
  auto wType = ::mlir::dyn_cast<::mlir::RankedTensorType>(weight.getType());
  if (!inputType || !wType) return failure();

  auto eltType = wType.getElementType();

  if (inputType.getRank() == 0 || wType.getRank() == 0) return failure(); // scalar not supported
  // Handle rank-1 input: promote to 2D [1, K], matmul to [1, N], reshape to result.
  if (inputType.getRank() == 1 && wType.getRank() >= 2) {
    int64_t kDim = inputType.getDimSize(0), nDim = wType.getDimSize(1);
    auto t1 = RankedTensorType::get({1, kDim < 0 ? ShapedType::kDynamic : kDim}, eltType);
    auto tOut = RankedTensorType::get({1, nDim < 0 ? ShapedType::kDynamic : nDim}, eltType);
    SmallVector<Value> t1Dyn; if (kDim < 0) t1Dyn.push_back(rewriter.create<tensor::DimOp>(loc, input, 0));
    SmallVector<Value> tOutDyn; if (nDim < 0) tOutDyn.push_back(rewriter.create<tensor::DimOp>(loc, weight, 1));
    Value pInput = rewriter.create<tensor::ExpandShapeOp>(loc, t1, input, ArrayRef<ReassociationIndices>{{0, 1}});
    Value pEmpty = rewriter.create<tensor::EmptyOp>(loc, tOut, tOutDyn);
    auto mo = rewriter.create<linalg::MatmulOp>(loc, tOut, ValueRange{pInput, weight}, pEmpty);
    mo->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({2, 1}));
    Value mmr = mo.getResult(0);
    // Reshape from [1, N] to result type via tensor.reshape
    auto rtt = cast<RankedTensorType>(resultType);
    SmallVector<Value> sv;
    for (int64_t i = 0; i < rtt.getRank(); ++i) {
      if (rtt.isDynamicDim(i)) sv.push_back(rewriter.create<tensor::DimOp>(loc, mmr, i == 0 ? 0 : 1));
      else sv.push_back(rewriter.create<arith::ConstantIndexOp>(loc, rtt.getDimSize(i)));
    }
    auto st = RankedTensorType::get({(int64_t)sv.size()}, rewriter.getIndexType());
    Value sh = sv.empty() ? (Value)rewriter.create<tensor::EmptyOp>(loc, st, ValueRange{})
                          : (Value)rewriter.create<tensor::FromElementsOp>(loc, st, sv);
    rewriter.replaceOp(op, rewriter.create<tensor::ReshapeOp>(loc, resultType, mmr, sh).getResult());
    return success();
  }

  auto inputRank = inputType.getRank();
  Value resultWeight;
  if (inputRank > 2 && wType.getRank() == 2) {
    // 3D input + 2D weight → batch_matmul. Weight needs to be [K, N] = [in, out].
    // The model stores weight as [out, in], so transpose to [in, out] first.
    SmallVector<int64_t> transShape = {wType.getDimSize(1), wType.getDimSize(0)};
    auto transType = RankedTensorType::get(transShape, eltType);
    auto emptyT = rewriter.create<tensor::EmptyOp>(loc, transType, ValueRange{});
    SmallVector<unsigned> perm = {1u, 0u};
    SmallVector<utils::IteratorType> titer(2, utils::IteratorType::parallel);
    Value emptyTVal = emptyT;
    auto transposeOp = linalg::GenericOp::create(rewriter, loc, transType,
        ValueRange{weight}, ValueRange{emptyTVal},
        {AffineMap::getPermutationMap(perm, rewriter.getContext()),
         AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())}, titer);
    populateBody(transposeOp, [&](OpBuilder &b, Location loc2, ValueRange args) {
      b.create<linalg::YieldOp>(loc2, args[0]);
    });
    Value transW = transposeOp.getResult(0);
    // Broadcast transposed weight from 2D to 3D: [in, out] → [batch, in, out].
    Value batchDim = rewriter.create<tensor::DimOp>(loc, input, 0);
    SmallVector<int64_t> w3dShape = {ShapedType::kDynamic,
                                       transType.getDimSize(0),
                                       transType.getDimSize(1)};
    auto w3dType = RankedTensorType::get(w3dShape, eltType);
    Value w3dEmpty = rewriter.create<tensor::EmptyOp>(loc, w3dType, ValueRange{batchDim});
    SmallVector<utils::IteratorType> biter(3, utils::IteratorType::parallel);
    Value w3dEmptyVal = w3dEmpty;
    auto w3dOp = linalg::GenericOp::create(rewriter, loc, w3dType,
        ValueRange{transW}, ValueRange{w3dEmptyVal},
        {broadcastMap(3, 2, rewriter.getContext()),
         AffineMap::getMultiDimIdentityMap(3, rewriter.getContext())}, biter);
    populateBody(w3dOp, [&](OpBuilder &b, Location loc2, ValueRange args) {
      b.create<linalg::YieldOp>(loc2, args[0]);
    });
    resultWeight = w3dOp.getResult(0);
  } else {
    // 2D input + 2D weight → standard matmul. Transpose weight to [out, in].
    resultWeight = weight;
    if (wType.getRank() == 2) {
      SmallVector<int64_t> transShape = {wType.getDimSize(1), wType.getDimSize(0)};
      auto transType = RankedTensorType::get(transShape, eltType);
      auto emptyT = rewriter.create<tensor::EmptyOp>(loc, transType, ValueRange{});
      SmallVector<unsigned> perm = {1u, 0u};
      SmallVector<utils::IteratorType> titer(2, utils::IteratorType::parallel);
      Value emptyTVal = emptyT;
      auto transposeOp = linalg::GenericOp::create(rewriter, loc, transType,
          ValueRange{weight}, ValueRange{emptyTVal},
          {AffineMap::getPermutationMap(perm, rewriter.getContext()),
           AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())}, titer);
      populateBody(transposeOp, [&](OpBuilder &b, Location loc2, ValueRange args) {
        b.create<linalg::YieldOp>(loc2, args[0]);
      });
      resultWeight = transposeOp.getResult(0);
    }
  }

  Value empty = makeEmpty(rewriter, loc, resultType, {input});
  if (!empty) { llvm::errs() << "  [SfLinear] makeEmpty failed\n"; return failure(); }
  Value result;
  auto finalWType = cast<RankedTensorType>(resultWeight.getType());
  auto resultTypeRT = cast<RankedTensorType>(resultType);
  llvm::errs() << "  [SfLinear] resultWeight rank=" << finalWType.getRank() << " resultType rank=" << resultTypeRT.getRank() << "\n";
  if (finalWType.getRank() > 2) {
    // Batch_matmul: all 3D. Weight is [batch, K, N] (broadcast from [in, out]).
    // Result may be 2D or 3D. If 2D, create 3D init, run batch_matmul, squeeze.
    auto bmResultType = resultType;
    if (resultTypeRT.getRank() != 3) {
      // Promote 2D result to 3D for batch_matmul
      auto inForShape = cast<RankedTensorType>(input.getType());
      int64_t d0 = inForShape.getRank() >= 2 ? inForShape.getDimSize(inForShape.getRank() - 2) : ShapedType::kDynamic;
      int64_t d1 = resultTypeRT.getDimSize(1);
      SmallVector<int64_t> bmShape = {ShapedType::kDynamic, d0, d1};
      auto bmType = RankedTensorType::get(bmShape, eltType);
      bmResultType = bmType;
    }
    Value bmEmpty = makeEmpty(rewriter, loc, bmResultType, {input});
    if (!bmEmpty) { llvm::errs() << "  [SfLinear] bmEmpty failed\n"; return failure(); }
    llvm::errs() << "  [SfLinear] creating batch_matmul target=" << bmResultType << "\n";
    auto mo = rewriter.create<linalg::BatchMatmulOp>(loc, bmResultType,
        ValueRange{input, resultWeight}, bmEmpty);
    mo->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({2, 1}));
    Value bmR = mo.getResult(0);
    // Squeeze 3D → 2D if needed
    if (resultTypeRT.getRank() != 3) {
      SmallVector<Value> sv;
      for (int64_t i = 0; i < resultTypeRT.getRank(); ++i) {
        if (resultTypeRT.isDynamicDim(i))
          sv.push_back(rewriter.create<tensor::DimOp>(loc, bmR, i + 1));
        else
          sv.push_back(rewriter.create<arith::ConstantIndexOp>(loc, resultTypeRT.getDimSize(i)));
      }
      auto st = RankedTensorType::get({(int64_t)sv.size()}, rewriter.getIndexType());
      Value sh = sv.empty() ? (Value)rewriter.create<tensor::EmptyOp>(loc, st, ValueRange{})
                            : (Value)rewriter.create<tensor::FromElementsOp>(loc, st, sv);
      result = rewriter.create<tensor::ReshapeOp>(loc, resultType, bmR, sh).getResult();
    } else {
      result = bmR;
    }
  } else {
    // 2D matmul
    auto mo = rewriter.create<linalg::MatmulOp>(loc, resultType,
        ValueRange{input, resultWeight}, empty);
    mo->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({2, 1}));
    result = mo.getResult(0);
  }
  rewriter.replaceOp(op, result);
  return success();
  }
};

// View → tensor reshape or expand/collapse
struct SfViewOpLowering : public OpConversionPattern<sf::ViewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::ViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Value input = adaptor.getInput();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inType || !outType) return failure();
    if (inType.getRank() == outType.getRank()) {
      rewriter.replaceOp(op, input);
      return success();
    }
    // Rank-changing view: use tensor.reshape with shape extracted from outType
    SmallVector<Value> shapeVals;
    for (int64_t i = 0; i < outType.getRank(); ++i) {
      if (outType.isDynamicDim(i))
        shapeVals.push_back(rewriter.create<tensor::DimOp>(loc, input, 0));
      else
        shapeVals.push_back(rewriter.create<arith::ConstantIndexOp>(loc, outType.getDimSize(i)));
    }
    auto shapeTensorType = RankedTensorType::get({(int64_t)shapeVals.size()},
                                                  rewriter.getIndexType());
    auto shapeTensor = rewriter.create<tensor::FromElementsOp>(loc, shapeTensorType, shapeVals);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, outType, input, shapeTensor);
    return success();
  }
};
struct SfExpandOpLowering : public OpConversionPattern<sf::ExpandOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::ExpandOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};
struct SfUnsqueezeOpLowering : public OpConversionPattern<sf::UnsqueezeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::UnsqueezeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Value input = adaptor.getInput();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inType || !outType) { rewriter.replaceOp(op, input); return success(); }
    if (inType.getRank() == outType.getRank()) {
      rewriter.replaceOp(op, input); return success();
    }
    // Rank-changing: use tensor.reshape with output shape values
    // (avoids linalg.copy rank mismatch verifier error)
    SmallVector<Value> shapeVals;
    for (int64_t i = 0; i < outType.getRank(); ++i) {
      if (outType.isDynamicDim(i))
        shapeVals.push_back(rewriter.create<tensor::DimOp>(loc, input, 0));
      else
        shapeVals.push_back(rewriter.create<arith::ConstantIndexOp>(loc, outType.getDimSize(i)));
    }
    auto shapeTensorType = RankedTensorType::get({(int64_t)shapeVals.size()},
                                                  rewriter.getIndexType());
    auto shapeTensor = rewriter.create<tensor::FromElementsOp>(loc, shapeTensorType, shapeVals);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, outType, input, shapeTensor);
    return success();
  }
};
struct SfSumOpLowering : public OpConversionPattern<sf::SumOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::SumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); auto rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();
    Value empty = makeEmpty(rewriter, loc, rt, {adaptor.getInput()});
    if (!empty) return failure();
    auto rank = cast<ShapedType>(rt).getRank();
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::reduction);
    auto g = linalg::GenericOp::create(rewriter, loc, rt, adaptor.getInput(), empty,
        identityMaps(rank, 2, rewriter.getContext()), iterTypes);
    populateBody(g, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value add = b.create<arith::AddFOp>(loc, args[0], args[1]);
      b.create<linalg::YieldOp>(loc, add);
    });
    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Broadcast-aware helpers for linalg.generic creation
//===----------------------------------------------------------------------===//

static Value makeBinaryOp(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                           Type outType, ConversionPatternRewriter &rewriter,
                           function_ref<Value(OpBuilder &, Location, Value, Value)> fn) {
  Value empty = makeEmpty(builder, loc, outType, {lhs});
  if (!empty) return Value();
  auto outRank = cast<ShapedType>(outType).getRank();
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  int64_t lhsRank = lhsType.getRank(), rhsRank = rhsType.getRank();

  SmallVector<AffineExpr> lhsExprs, rhsExprs;
  for (int64_t i = 0; i < outRank; ++i) {
    int64_t outDim = cast<ShapedType>(outType).getDimSize(i);
    int64_t lhsI = i - (outRank - lhsRank);
    int64_t rhsI = i - (outRank - rhsRank);
    if (lhsI >= 0) {
      int64_t lhsDim = lhsType.getDimSize(lhsI);
      lhsExprs.push_back((lhsDim == 1 && outDim > 1)
          ? getAffineConstantExpr(0, builder.getContext())
          : getAffineDimExpr(i, builder.getContext()));
    }
    if (rhsI >= 0) {
      int64_t rhsDim = rhsType.getDimSize(rhsI);
      rhsExprs.push_back((rhsDim == 1 && outDim > 1)
          ? getAffineConstantExpr(0, builder.getContext())
          : getAffineDimExpr(i, builder.getContext()));
    }
  }
  auto lhsMap = AffineMap::get(outRank, 0, lhsExprs, builder.getContext());
  auto rhsMap = AffineMap::get(outRank, 0, rhsExprs, builder.getContext());
  auto outMap = AffineMap::getMultiDimIdentityMap(outRank, builder.getContext());

  SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
  auto g = linalg::GenericOp::create(rewriter, loc, outType,
      ValueRange{lhs, rhs}, empty, {lhsMap, rhsMap, outMap}, iters);
  populateBody(g, [&](OpBuilder &b, Location loc, ValueRange args) {
    Value v = fn(b, loc, args[0], args[1]);
    b.create<linalg::YieldOp>(loc, v);
  });
  return g.getResult(0);
}

static Value makeUnaryOp(OpBuilder &builder, Location loc, Value in, Type outType,
                          ConversionPatternRewriter &rewriter,
                          function_ref<Value(OpBuilder &, Location, Value)> fn) {
  Value empty = makeEmpty(builder, loc, outType, {in});
  if (!empty) return Value();
  auto outRank = cast<ShapedType>(outType).getRank();
  auto inType = cast<RankedTensorType>(in.getType());
  int64_t inRank = inType.getRank();

  SmallVector<AffineExpr> inExprs;
  for (int64_t i = 0; i < outRank; ++i) {
    int64_t inI = i - (outRank - inRank);
    if (inI >= 0) {
      int64_t outDim = cast<ShapedType>(outType).getDimSize(i);
      int64_t inDim = inType.getDimSize(inI);
      inExprs.push_back((inDim == 1 && outDim > 1)
          ? getAffineConstantExpr(0, builder.getContext())
          : getAffineDimExpr(i, builder.getContext()));
    }
  }
  auto inMap = AffineMap::get(outRank, 0, inExprs, builder.getContext());
  auto outMap = AffineMap::getMultiDimIdentityMap(outRank, builder.getContext());

  SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
  auto g = linalg::GenericOp::create(rewriter, loc, outType,
      in, empty, {inMap, outMap}, iters);
  populateBody(g, [&](OpBuilder &b, Location loc, ValueRange args) {
    Value v = fn(b, loc, args[0]);
    b.create<linalg::YieldOp>(loc, v);
  });
  return g.getResult(0);
}

//===----------------------------------------------------------------------===//
// Scaled Dot-Product Attention lowering:
//   softmax(Q * K^T / sqrt(d_k)) * V
//   Step 1: K^T = transpose(K)
//   Step 2: scores = matmul(Q, K^T)
//   Step 3: scores_scaled = scores * (1/sqrt(d_k))
//   Step 4: attn = softmax(scores_scaled) along last dim
//   Step 5: output = matmul(attn, V)
//===----------------------------------------------------------------------===//

struct SfScaledDotProductAttentionOpLowering
    : public OpConversionPattern<sf::ScaledDotProductAttentionOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::ScaledDotProductAttentionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value Q = adaptor.getQuery();
    Value K = adaptor.getKey();
    Value V = adaptor.getValue();

    auto qType = ::mlir::dyn_cast<::mlir::RankedTensorType>(Q.getType());
    if (!qType || qType.getRank() < 3) return failure();

    auto eltType = qType.getElementType();
    int64_t rank = qType.getRank();
    int64_t dk = qType.getDimSize(rank - 1);
    if (dk < 0) return failure();
    float scaleVal = 1.0f / std::sqrt(static_cast<float>(dk));
    auto ctx = rewriter.getContext();

    // Step 1: transpose K (last two dims)
    SmallVector<int64_t> ktShape(qType.getShape());
    std::swap(ktShape[rank - 1], ktShape[rank - 2]);
    auto ktType = RankedTensorType::get(ktShape, eltType);
    // Helper: dynamic sizes for ktType (last dim = seq from K dim rank-2)
    auto ktDyn = [&]() -> SmallVector<Value> {
      SmallVector<Value> dyns;
      for (int64_t i = 0; i < rank; ++i) {
        if (!ktType.isDynamicDim(i)) continue;
        if (i == rank - 1)
          dyns.push_back(rewriter.create<tensor::DimOp>(loc, K, rank - 2));
        else
          dyns.push_back(rewriter.create<tensor::DimOp>(loc, K, i));
      }
      return dyns;
    };
    Value ktEmpty = rewriter.create<tensor::EmptyOp>(loc, ktType, ktDyn());
    SmallVector<unsigned> ktPerm(rank);
    for (int64_t i = 0; i < rank; ++i) ktPerm[i] = i;
    std::swap(ktPerm[rank - 1], ktPerm[rank - 2]);
    SmallVector<utils::IteratorType> ktIterTypes(rank, utils::IteratorType::parallel);
    auto ktOp = linalg::GenericOp::create(rewriter, loc, ktType, K, ktEmpty,
        {AffineMap::getPermutationMap(ktPerm, rewriter.getContext()),
         AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())}, ktIterTypes);
    populateBody(ktOp, [&](OpBuilder &b, Location loc, ValueRange args) {
      b.create<linalg::YieldOp>(loc, args[0]);
    });
    Value Kt = ktOp.getResult(0);

    // Helper: scores-type shapes (last dim = k_seq from K dim rank-2)
    auto scoresDyn = [&](RankedTensorType type) -> SmallVector<Value> {
      SmallVector<Value> dyns;
      for (int64_t i = 0; i < rank; ++i) {
        if (!type.isDynamicDim(i)) continue;
        if (i == rank - 1)
          dyns.push_back(rewriter.create<tensor::DimOp>(loc, K, rank - 2));
        else
          dyns.push_back(rewriter.create<tensor::DimOp>(loc, Q, i));
      }
      return dyns;
    };

    // Step 2: scores = matmul(Q, K^T) via linalg.generic (supports any rank)
    SmallVector<int64_t> scoresShape(qType.getShape());
    scoresShape[rank - 1] = qType.getDimSize(rank - 2);
    auto scoresType = RankedTensorType::get(scoresShape, eltType);
    Value scoresEmpty = rewriter.create<tensor::EmptyOp>(loc, scoresType, scoresDyn(scoresType));
    Value scores;
    {
      // Generic batch-matmul: loop [b..h, m, n, k]: parallel(b..h), parallel(m), parallel(n), reduction(k)
      // Maps: Q->(b..h,m,k) Kt->(b..h,k,n) Out->(b..h,m,n)
      int64_t mDim = rank - 2, kRed = rank; // loop has rank+1 dims
      SmallVector<AffineExpr> lhsMaps, rhsMaps, outMaps;
      for (int64_t i = 0; i < rank - 2; ++i)
        { lhsMaps.push_back(getAffineDimExpr(i, ctx)); rhsMaps.push_back(getAffineDimExpr(i, ctx)); outMaps.push_back(getAffineDimExpr(i, ctx)); }
      lhsMaps.push_back(getAffineDimExpr(mDim, ctx)); // M dim
      lhsMaps.push_back(getAffineDimExpr(kRed, ctx)); // K dim (contraction)
      rhsMaps.push_back(getAffineDimExpr(kRed, ctx)); // K dim
      rhsMaps.push_back(getAffineDimExpr(mDim + 1, ctx)); // N dim (= rank-1 in loop)
      outMaps.push_back(getAffineDimExpr(mDim, ctx)); // M dim
      outMaps.push_back(getAffineDimExpr(mDim + 1, ctx)); // N dim
      int64_t loopRank = rank + 1;
      SmallVector<utils::IteratorType> matIter(loopRank, utils::IteratorType::parallel);
      matIter[kRed] = utils::IteratorType::reduction;
      auto scoreGeneric = linalg::GenericOp::create(rewriter, loc, scoresType,
          ValueRange{Q, Kt}, ValueRange{scoresEmpty},
          {AffineMap::get(loopRank, 0, lhsMaps, ctx),
           AffineMap::get(loopRank, 0, rhsMaps, ctx),
           AffineMap::get(loopRank, 0, outMaps, ctx)}, matIter,
          [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
            Value _mul = b.create<arith::MulFOp>(bodyLoc, args[0], args[1]);
            Value _add = b.create<arith::AddFOp>(bodyLoc, args[2], _mul);
            b.create<linalg::YieldOp>(bodyLoc, _add);
          });
      scores = scoreGeneric.getResult(0);
    }

    // Step 3: scores_scaled = scores * (1/sqrt(d_k))
    Value scaleConst = rewriter.create<arith::ConstantOp>(loc, eltType,
        rewriter.getFloatAttr(eltType, scaleVal));
    Value scaledEmpty = rewriter.create<tensor::EmptyOp>(loc, scoresType, scoresDyn(scoresType));
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto scaleOp = linalg::GenericOp::create(rewriter, loc, scoresType,
        scores, scaledEmpty,
        identityMaps(rank, 2, rewriter.getContext()), iterTypes);
    populateBody(scaleOp, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value _scaled = b.create<arith::MulFOp>(loc, args[0], scaleConst); b.create<linalg::YieldOp>(loc, _scaled);
    });
    Value scoresScaled = scaleOp.getResult(0);

    // Step 4: softmax along last dim
    // softmax(x) = exp(x - max) / sum(exp(x - max))
    int64_t lastDim = rank - 1;

    // Helper: max-type shapes (last dim = 1, static; leading dims from Q)
    auto maxDyn = [&](RankedTensorType type) -> SmallVector<Value> {
      SmallVector<Value> dyns;
      for (int64_t i = 0; i < rank; ++i) {
        if (!type.isDynamicDim(i)) continue;
        if (i == rank - 2)
          dyns.push_back(rewriter.create<tensor::DimOp>(loc, Q, rank - 2));
        else
          dyns.push_back(rewriter.create<tensor::DimOp>(loc, Q, i));
      }
      return dyns;
    };

    // 4a: max reduction along last dim
    SmallVector<int64_t> maxShape(scoresShape);
    maxShape[lastDim] = 1;
    auto maxType = RankedTensorType::get(maxShape, eltType);
    Value maxEmpty = rewriter.create<tensor::EmptyOp>(loc, maxType, maxDyn(maxType));
    SmallVector<utils::IteratorType> reduIters(rank);
    for (int64_t i = 0; i < rank; ++i)
      reduIters[i] = (i == lastDim) ? utils::IteratorType::reduction : utils::IteratorType::parallel;
    auto maxOp = linalg::GenericOp::create(rewriter, loc, maxType, scoresScaled, maxEmpty,
        {AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()),
         AffineMap::get(rank, 0,
             llvm::map_to_vector(llvm::seq<int64_t>(0, rank), [&](int64_t i) -> AffineExpr {
               return (i == lastDim) ? getAffineConstantExpr(0, rewriter.getContext())
                                     : getAffineDimExpr(i, rewriter.getContext());
             }), rewriter.getContext())}, reduIters);
    populateBody(maxOp, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value _mx = b.create<arith::MaxNumFOp>(loc, args[0], args[1]); b.create<linalg::YieldOp>(loc, _mx);
    });
    Value maxVal = maxOp.getResult(0);

    // 4b: sub = x - max (broadcast)
    Value sub = makeBinaryOp(rewriter, loc, scoresScaled, maxVal, scoresType, rewriter,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return b.create<arith::SubFOp>(loc, a, bVal);
        });

    // 4c: exp(x - max)
    Value expVal = makeUnaryOp(rewriter, loc, sub, scoresType, rewriter,
        [&](OpBuilder &b, Location loc, Value v) {
          return b.create<math::ExpOp>(loc, v);
        });

    // 4d: sum reduction
    Value sumEmpty = rewriter.create<tensor::EmptyOp>(loc, maxType, maxDyn(maxType));
    auto sumOp = linalg::GenericOp::create(rewriter, loc, maxType, expVal, sumEmpty,
        {AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()),
         AffineMap::get(rank, 0,
             llvm::map_to_vector(llvm::seq<int64_t>(0, rank), [&](int64_t i) -> AffineExpr {
               return (i == lastDim) ? getAffineConstantExpr(0, rewriter.getContext())
                                     : getAffineDimExpr(i, rewriter.getContext());
             }), rewriter.getContext())}, reduIters);
    populateBody(sumOp, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value _ad = b.create<arith::AddFOp>(loc, args[0], args[1]); b.create<linalg::YieldOp>(loc, _ad);
    });
    Value sumVal = sumOp.getResult(0);

    // 4e: softmax = exp / sum (broadcast)
    Value attn = makeBinaryOp(rewriter, loc, expVal, sumVal, scoresType, rewriter,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return b.create<arith::DivFOp>(loc, a, bVal);
        });

    // Step 5: output = matmul(attn, V) via linalg.generic (supports any rank)
    SmallVector<int64_t> outShape(qType.getShape());
    outShape[rank - 1] = ::mlir::cast<::mlir::RankedTensorType>(V.getType()).getDimSize(rank - 1);
    auto outType = RankedTensorType::get(outShape, eltType);
    auto outEmptyType = RankedTensorType::get(outShape, eltType);
    SmallVector<Value> outDyn;
    for (int64_t i = 0; i < rank; ++i) {
      if (!outEmptyType.isDynamicDim(i)) continue;
      if (i == rank - 1)
        outDyn.push_back(rewriter.create<tensor::DimOp>(loc, V, rank - 1));
      else if (i == rank - 2)
        outDyn.push_back(rewriter.create<tensor::DimOp>(loc, Q, rank - 2));
      else
        outDyn.push_back(rewriter.create<tensor::DimOp>(loc, Q, i));
    }
    Value outEmpty = rewriter.create<tensor::EmptyOp>(loc, outEmptyType, outDyn);
    Value attnVResult;
    {
      // Generic batch-matmul: loop [b..h, m, s, d]: parallel(b..h), parallel(m), reduction(s), parallel(d)
      // Maps: attn->(b..h,m,s) V->(b..h,s,d) Out->(b..h,m,d)
      int64_t redDim = rank - 1; // s dim in attn, rank-1 in V = contraction
      int64_t loopRank = rank + 1;
      SmallVector<AffineExpr> attnMaps, vMaps, outMaps;
      for (int64_t i = 0; i < rank - 2; ++i)
        { attnMaps.push_back(getAffineDimExpr(i, ctx)); vMaps.push_back(getAffineDimExpr(i, ctx)); outMaps.push_back(getAffineDimExpr(i, ctx)); }
      attnMaps.push_back(getAffineDimExpr(rank - 2, ctx)); // M dim
      attnMaps.push_back(getAffineDimExpr(redDim, ctx));   // S dim (contraction)
      vMaps.push_back(getAffineDimExpr(rank - 2, ctx));    // S dim (contraction, at rank-1 in V but dim rank-2 in attn)
      vMaps.push_back(getAffineDimExpr(loopRank - 1, ctx)); // D dim (last in loop)
      outMaps.push_back(getAffineDimExpr(rank - 2, ctx));  // M dim
      outMaps.push_back(getAffineDimExpr(loopRank - 1, ctx)); // D dim
      SmallVector<utils::IteratorType> outMatIter(loopRank, utils::IteratorType::parallel);
      outMatIter[redDim] = utils::IteratorType::reduction;
      auto outGeneric = linalg::GenericOp::create(rewriter, loc, outType,
          ValueRange{attn, V}, ValueRange{outEmpty},
          {AffineMap::get(loopRank, 0, attnMaps, ctx),
           AffineMap::get(loopRank, 0, vMaps, ctx),
           AffineMap::get(loopRank, 0, outMaps, ctx)}, outMatIter,
          [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
            Value _mul = b.create<arith::MulFOp>(bodyLoc, args[0], args[1]);
            Value _add = b.create<arith::AddFOp>(bodyLoc, args[2], _mul);
            b.create<linalg::YieldOp>(bodyLoc, _add);
          });
      attnVResult = outGeneric.getResult(0);
    }

    rewriter.replaceOp(op, attnVResult);
    return success();
  }
};

struct SfLayerNormOpLowering : public OpConversionPattern<sf::LayerNormOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::LayerNormOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    Value bias = adaptor.getBias();
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
      Value dimVal = rewriter.create<tensor::DimOp>(loc, input, lastDim);
      Value dimI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), dimVal);
      Value dimF32 = rewriter.create<arith::UIToFPOp>(loc, eltType, dimI64);
      Value one = rewriter.create<arith::ConstantOp>(loc, eltType,
          rewriter.getFloatAttr(eltType, 1.0));
      invDim = rewriter.create<arith::DivFOp>(loc, one, dimF32);
    } else {
      invDim = rewriter.create<arith::ConstantOp>(loc, eltType,
          rewriter.getFloatAttr(eltType, 1.0f / dimSize));
    }

    // Output type for reductions (same as input but last dim = 1, then broadcast)
    SmallVector<int64_t> reducedShape(inputType.getShape());
    reducedShape[lastDim] = 1;
    auto reducedType = RankedTensorType::get(reducedShape, eltType);

    // Helper: create element-wise binary generic with broadcast maps
    auto makeBinary = [&](Value lhs, Value rhs, Type outType,
                           function_ref<Value(OpBuilder &, Location, Value, Value)> fn) -> Value {
      Value empty = makeEmpty(rewriter, loc, outType, {lhs});
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
          lhsExprs.push_back((lhsDim == 1 && outDim > 1)
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
      populateBody(g, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _v = fn(b, loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, _v);
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
          inExprs.push_back((inDim == 1 && outDim > 1)
              ? getAffineConstantExpr(0, rewriter.getContext())
              : getAffineDimExpr(i, rewriter.getContext()));
        }
      }
      auto inMap = AffineMap::get(outRank, 0, inExprs, rewriter.getContext());
      auto outMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());

      SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
      auto g = linalg::GenericOp::create(rewriter, loc, outType,
          in, empty, {inMap, outMap}, iters);
      populateBody(g, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _v = fn(b, loc, args[0]);
        b.create<linalg::YieldOp>(loc, _v);
      });
      return g.getResult(0);
    };

    // Helper: reduce along last dim
    auto makeReduce = [&](Value in, Type reduType) -> Value {
      Value empty = makeEmpty(rewriter, loc, reduType, {in});
      if (!empty) return Value();
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
      auto g = linalg::GenericOp::create(rewriter, loc, reduType, in, empty,
          {inMap, outMap}, iters);
      populateBody(g, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _ad = b.create<arith::AddFOp>(loc, args[0], args[1]); b.create<linalg::YieldOp>(loc, _ad);
      });
      return g.getResult(0);
    };

    // Step 1: mean = reduce_sum(x) / dimSize
    Value sumVal = makeReduce(input, reducedType);
    Value meanVal = makeUnary(sumVal, reducedType, [&](OpBuilder &b, Location loc, Value v) {
      return b.create<arith::MulFOp>(loc, v, invDim);
    });

    // Step 2: diff = x - mean (broadcast)
    Value diff = makeBinary(input, meanVal, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return b.create<arith::SubFOp>(loc, a, bVal);
        });

    // Step 3: diff_sq = diff * diff
    Value diffSq = makeBinary(diff, diff, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return b.create<arith::MulFOp>(loc, a, bVal);
        });

    // Step 4: var = reduce_sum(diff_sq) / dimSize
    Value varSum = makeReduce(diffSq, reducedType);
    Value varVal = makeUnary(varSum, reducedType, [&](OpBuilder &b, Location loc, Value v) {
      return b.create<arith::MulFOp>(loc, v, invDim);
    });

    // Step 5: std = sqrt(var + eps)
    Value stdVal = makeUnary(varVal, reducedType, [&](OpBuilder &b, Location loc, Value v) {
      Value epsVal = b.create<arith::ConstantOp>(loc, eltType,
          b.getFloatAttr(eltType, eps));
      Value add = b.create<arith::AddFOp>(loc, v, epsVal);
      return b.create<math::SqrtOp>(loc, add);
    });

    // Step 6: norm = diff / std (broadcast)
    Value norm = makeBinary(diff, stdVal, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return b.create<arith::DivFOp>(loc, a, bVal);
        });

    // Step 7: result = norm * weight + bias
    Value weighted = makeBinary(norm, weight, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return b.create<arith::MulFOp>(loc, a, bVal);
        });
    Value result = makeBinary(weighted, bias, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return b.create<arith::AddFOp>(loc, a, bVal);
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
struct SfRmsNormOpLowering : public OpConversionPattern<sf::RmsNormOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::RmsNormOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
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
      auto g = linalg::GenericOp::create(rewriter, loc, reduType, in, empty,
          {inMap, outMap}, iters);
      populateBody(g, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _ad = b.create<arith::AddFOp>(loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, _ad);
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
          inExprs.push_back((inDim == 1 && outDim > 1)
              ? getAffineConstantExpr(0, rewriter.getContext())
              : getAffineDimExpr(i, rewriter.getContext()));
        }
      }
      auto inMap = AffineMap::get(outRank, 0, inExprs, rewriter.getContext());
      auto outMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());
      SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
      auto g = linalg::GenericOp::create(rewriter, loc, outType, in, empty,
          {inMap, outMap}, iters);
      populateBody(g, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _v = fn(b, loc, args[0]);
        b.create<linalg::YieldOp>(loc, _v);
      });
      return g.getResult(0);
    };

    // Helper: binary generic with broadcast
    auto makeBinary = [&](Value lhs, Value rhs, Type outType,
                           function_ref<Value(OpBuilder &, Location, Value, Value)> fn) -> Value {
      Value empty = makeEmpty(rewriter, loc, outType, {lhs});
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
          lhsExprs.push_back((lhsDim == 1 && outDim > 1)
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
      populateBody(g, [&](OpBuilder &b, Location loc, ValueRange args) {
        Value _v = fn(b, loc, args[0], args[1]);
        b.create<linalg::YieldOp>(loc, _v);
      });
      return g.getResult(0);
    };

    // Step 1: x² = x * x
    Value sq = makeBinary(input, input, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return b.create<arith::MulFOp>(loc, a, bVal);
        });

    // Step 2: mean_x2 = reduce_sum(x²) / dimSize
    Value sumSq = makeReduce(sq, reducedType);
    Value meanSq = makeUnary(sumSq, reducedType, [&](OpBuilder &b, Location loc, Value v) {
      Value scale = b.create<arith::ConstantOp>(loc, eltType,
          b.getFloatAttr(eltType, 1.0f / dimSize));
      return b.create<arith::MulFOp>(loc, v, scale);
    });

    // Step 3: rms = sqrt(mean_x2 + eps)
    Value rmsVal = makeUnary(meanSq, reducedType, [&](OpBuilder &b, Location loc, Value v) {
      Value epsVal = b.create<arith::ConstantOp>(loc, eltType,
          b.getFloatAttr(eltType, eps));
      Value add = b.create<arith::AddFOp>(loc, v, epsVal);
      return b.create<math::SqrtOp>(loc, add);
    });

    // Step 4: normed = x / rms (broadcast)
    Value normed = makeBinary(input, rmsVal, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return b.create<arith::DivFOp>(loc, a, bVal);
        });

    // Step 5: out = normed * weight
    Value result = makeBinary(normed, weight, inputType,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return b.create<arith::MulFOp>(loc, a, bVal);
        });

    rewriter.replaceOp(op, result);
    return success();
  }
};
struct SfTransposeOpLowering : public OpConversionPattern<sf::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Value input = adaptor.getInput();
    Type resultType = op.getResult().getType();
    if (!isa<ShapedType>(resultType)) return failure();
    auto rt = cast<RankedTensorType>(resultType);
    auto rank = rt.getRank();
    if (rank < 2) { rewriter.replaceOp(op, input); return success(); }
    int64_t d0 = 0, d1 = 1;
    if (auto d0Attr = op.getOperation()->getAttrOfType<IntegerAttr>("dim0"))
      d0 = d0Attr.getInt();
    if (auto d1Attr = op.getOperation()->getAttrOfType<IntegerAttr>("dim1"))
      d1 = d1Attr.getInt();
    SmallVector<int64_t> perm(rank);
    for (int64_t i = 0; i < rank; ++i) perm[i] = i;
    std::swap(perm[d0], perm[d1]);
    Value empty = makeEmpty(rewriter, loc, resultType, {input});
    if (!empty) return failure();
    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, input, empty, rewriter.getDenseI64ArrayAttr(perm));
    rewriter.replaceOp(op, transposeOp->getResult(0));
    return success();
  }
};

// Slice → tensor.extract_slice
struct SfSliceOpLowering : public OpConversionPattern<sf::SliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::SliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Value input = adaptor.getInput();
    auto inType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    if (!inType) return failure();
    int64_t dim = 0, start = 0, sEnd = 0;
    if (auto attr = op.getOperation()->getAttrOfType<IntegerAttr>("dim")) dim = attr.getInt();
    if (auto attr = op.getOperation()->getAttrOfType<IntegerAttr>("start")) start = attr.getInt();
    if (auto attr = op.getOperation()->getAttrOfType<IntegerAttr>("end")) sEnd = attr.getInt();
    static constexpr int64_t kDynSentinel = 9223372036854775807LL;
    int64_t rank = inType.getRank();
    SmallVector<OpFoldResult> offs, szs, strs;
    for (int64_t i = 0; i < rank; ++i) {
      offs.push_back(rewriter.getIndexAttr((i == dim) ? start : 0));
      if (i == dim && sEnd == kDynSentinel) {
        // INT64_MAX sentinel: size = dim(input, i) - start (runtime)
        Value dimVal = rewriter.create<tensor::DimOp>(loc, input, i);
        Value startVal = rewriter.create<arith::ConstantIndexOp>(loc, start);
        szs.push_back(Value(rewriter.create<arith::SubIOp>(loc, dimVal, startVal).getResult()));
      } else if (inType.isDynamicDim(i)) {
        szs.push_back(Value(rewriter.create<tensor::DimOp>(loc, input, i).getResult()));
      } else {
        szs.push_back(rewriter.getIndexAttr((i == dim) ? (sEnd - start) : inType.getDimSize(i)));
      }
      strs.push_back(rewriter.getIndexAttr(1));
    }
    auto slice = tensor::ExtractSliceOp::create(rewriter, loc, input, offs, szs, strs);
    rewriter.replaceOp(op, slice->getResult(0));
    return success();
  }
};

// logical_and survives (f32 operands, needs i1 conversion)


// Le comparison → arith.cmpf in generic
// Uses body builder (not populateBody) so body args have input element type (f32),
// not output element type (i1).
struct SfLeOpLowering : public OpConversionPattern<sf::LeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::LeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();
    // Always use f32 output type to avoid i1→f32 unrealized_conversion_cast
    auto outType = cast<ShapedType>(rt);
    auto f32OutType = outType.cloneWith(outType.getShape(), rewriter.getF32Type());
    Value empty = makeEmpty(rewriter, loc, f32OutType, {adaptor.getLhs()});
    if (!empty) return failure();
    auto rank = outType.getRank();
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto lhsType2 = cast<RankedTensorType>(adaptor.getLhs().getType());
    auto rhsType2 = cast<RankedTensorType>(adaptor.getRhs().getType());
    auto lhsMap = broadcastMap(rank, lhsType2.getRank(), rewriter.getContext(), lhsType2.getShape());
    auto rhsMap = broadcastMap(rank, rhsType2.getRank(), rewriter.getContext(), rhsType2.getShape());
    auto outMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    auto g = linalg::GenericOp::create(rewriter, loc, f32OutType,
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, empty,
        {lhsMap, rhsMap, outMap}, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      Value lhsB, rhsB;
      if (isa<IntegerType>(args[0].getType())) {
        lhsB = args[0];
      } else {
        Value zero = b.create<arith::ConstantOp>(loc, rewriter.getF32Type(),
            b.getFloatAttr(rewriter.getF32Type(), 0.0));
        lhsB = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, args[0], zero);
      }
      if (isa<IntegerType>(args[1].getType())) {
        rhsB = args[1];
      } else {
        Value zero = b.create<arith::ConstantOp>(loc, rewriter.getF32Type(),
            b.getFloatAttr(rewriter.getF32Type(), 0.0));
        rhsB = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, args[1], zero);
      }
      Value andB = b.create<arith::AndIOp>(loc, lhsB, rhsB);
      Value result = b.create<arith::UIToFPOp>(loc, rewriter.getF32Type(), andB);
      b.create<linalg::YieldOp>(loc, result);
    });
    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

// LogicalAnd → linalg.generic with f32 operands
//   bool_a = cmp UGT(a, 0.0), bool_b = cmp UGT(b, 0.0)
//   and = andi(bool_a, bool_b)
//   result = uitofp(and) → f32
struct SfLogicalAndOpLowering : public OpConversionPattern<sf::LogicalAndOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::LogicalAndOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();
    // Always use f32 output type to avoid i1→f32 unrealized_conversion_cast
    auto outType = cast<ShapedType>(rt);
    auto f32OutType = outType.cloneWith(outType.getShape(), rewriter.getF32Type());
    Value empty = makeEmpty(rewriter, loc, f32OutType, {adaptor.getLhs()});
    if (!empty) return failure();
    auto rank = outType.getRank();
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto lhsType2 = cast<RankedTensorType>(adaptor.getLhs().getType());
    auto rhsType2 = cast<RankedTensorType>(adaptor.getRhs().getType());
    auto lhsMap = broadcastMap(rank, lhsType2.getRank(), rewriter.getContext(), lhsType2.getShape());
    auto rhsMap = broadcastMap(rank, rhsType2.getRank(), rewriter.getContext(), rhsType2.getShape());
    auto outMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    auto g = linalg::GenericOp::create(rewriter, loc, f32OutType,
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, empty,
        {lhsMap, rhsMap, outMap}, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      Value cmp = b.create<arith::CmpFOp>(bodyLoc, arith::CmpFPredicate::OLE, args[0], args[1]);
      Value result = b.create<arith::UIToFPOp>(bodyLoc, rewriter.getF32Type(), cmp);
      b.create<linalg::YieldOp>(bodyLoc, result);
    });
    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};

// OnesLike → linalg.fill(1.0)
struct SfOnesLikeOpLowering : public OpConversionPattern<sf::OnesLikeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::OnesLikeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();
    Value empty = makeEmpty(rewriter, loc, rt, {adaptor.getInput()});
    if (!empty) return failure();
    auto elt = getElementTypeOrSelf(rt);
    Value oneVal = rewriter.create<arith::ConstantOp>(loc, elt,
        rewriter.getFloatAttr(elt, 1.0));
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{oneVal}, ValueRange{empty});
    return success();
  }
};

// NewOnes → tensor.empty + linalg.fill(1.0)
struct SfNewOnesOpLowering : public OpConversionPattern<sf::NewOnesOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::NewOnesOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Type rt = op.getResult().getType();
    if (!isa<ShapedType>(rt)) return failure();
    Value empty = makeEmpty(rewriter, loc, rt, {adaptor.getInput()});
    if (!empty) return failure();
    auto elt = getElementTypeOrSelf(rt);
    Value oneVal = rewriter.create<arith::ConstantOp>(loc, elt,
        rewriter.getFloatAttr(elt, 1.0));
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{oneVal}, ValueRange{empty});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SymSize → tensor.dim + cast + tensor.insert
//===----------------------------------------------------------------------===//

struct SfSymSizeOpLowering : public OpConversionPattern<sf::SymSizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::SymSizeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Type rt = op.getResult().getType();
    auto inputType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    if (!inputType) return failure();

    int64_t dim = 0;
    if (auto dimAttr = op.getOperation()->getAttrOfType<IntegerAttr>("dim"))
      dim = dimAttr.getInt();
    if (dim < 0 || dim >= inputType.getRank()) return failure();

    Value dimVal = rewriter.create<tensor::DimOp>(loc, input, dim);
    Value dimI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), dimVal);
    auto f32Type = rewriter.getF32Type();
    Value dimF32 = rewriter.create<arith::UIToFPOp>(loc, f32Type, dimI64);
    RankedTensorType outTensorType = RankedTensorType::get({}, f32Type);
    Value empty = rewriter.create<tensor::EmptyOp>(loc, outTensorType, ValueRange{});
    Value result = rewriter.create<tensor::InsertOp>(loc, dimF32, empty, ValueRange{});
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Arange → tensor.empty + scf.for fill
//===----------------------------------------------------------------------===//

struct SfArangeOpLowering : public OpConversionPattern<sf::ArangeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::ArangeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Type rt = op.getResult().getType();
    auto outType = ::mlir::dyn_cast<::mlir::RankedTensorType>(rt);
    if (!outType) return failure();
    if (outType.getRank() == 0) {
      // Scalar arange: not meaningful; just return zero.
      auto eltType = getElementTypeOrSelf(rt);
      Value zero = rewriter.create<arith::ConstantOp>(loc, eltType,
          rewriter.getFloatAttr(eltType, 0.0));
      auto empty = rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{}, eltType, ValueRange{});
      rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, zero, empty, ValueRange{});
      return success();
    }
    if (outType.getRank() != 1) return failure();
    auto eltType = getElementTypeOrSelf(rt);

    // Extract first element from input (use zero indices for non-scalar input)
    auto inType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    SmallVector<Value> zeroIdx;
    if (inType) for (int64_t _i = 0; _i < inType.getRank(); ++_i)
      zeroIdx.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    Value nF32 = rewriter.create<tensor::ExtractOp>(loc, input, zeroIdx);
    Value nI64 = rewriter.create<arith::FPToUIOp>(loc, rewriter.getI64Type(), nF32);
    Value nIdx = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), nI64);

    // Create empty tensor with correct output type (static or dynamic)
    Value empty;
    if (outType.hasStaticShape()) {
      empty = rewriter.create<tensor::EmptyOp>(loc, outType, ValueRange{});
    } else {
      SmallVector<int64_t> dynShape = {ShapedType::kDynamic};
      empty = rewriter.create<tensor::EmptyOp>(loc, dynShape, eltType, ValueRange{nIdx});
    }

    // scf.for %i = 0 to N
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value zeroI64 = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    auto forOp = rewriter.create<scf::ForOp>(loc, c0, nIdx, c1, empty);
    Value iv = forOp.getInductionVar();
    Value initOut = forOp.getInitArgs()[0];

    rewriter.setInsertionPointToStart(forOp.getBody());
    Value ivI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), iv);
    Value ivF32 = rewriter.create<arith::UIToFPOp>(loc, eltType, ivI64);
    Value outVal = rewriter.create<tensor::InsertOp>(loc, eltType, ivF32, initOut, iv);
    rewriter.create<scf::YieldOp>(loc, outVal);

    rewriter.setInsertionPointAfter(forOp);
    rewriter.replaceOp(op, forOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Cumsum → scf.for loop accumulation along dim
//===----------------------------------------------------------------------===//

struct SfCumsumOpLowering : public OpConversionPattern<sf::CumsumOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::CumsumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getInput();
    Type rt = op.getResult().getType();
    auto inType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    auto outType = ::mlir::dyn_cast<::mlir::RankedTensorType>(rt);
    if (!inType || !outType) return failure();

    int64_t dim = 0;
    if (auto dimAttr = op.getOperation()->getAttrOfType<IntegerAttr>("dim"))
      dim = dimAttr.getInt();
    if (dim < 0 || dim >= inType.getRank()) {
      // Scalar cumsum: identity — return a copy of input
      Value empty = rewriter.create<tensor::EmptyOp>(loc, outType, ValueRange{});
      rewriter.replaceOpWithNewOp<linalg::CopyOp>(op, input, empty);
      return success();
    }
    auto eltType = inType.getElementType();

    int64_t dimSize = inType.getDimSize(dim);
    if (dimSize < 0) return failure(); // dynamic dim, skip

    // Copy input to output first
    Value empty = rewriter.create<tensor::EmptyOp>(loc, outType, ValueRange{});
    Value initOut = rewriter.create<linalg::CopyOp>(loc, input, empty).getResult(0);

    // For i = 1 to dimSize-1: output[..., i, ...] += output[..., i-1, ...]
    // We iterate over the dim axis and use tensor.extract/tensor.insert
    // to accumulate. For each i > 0, we iterate over all non-dim positions.
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value dimEnd = rewriter.create<arith::ConstantIndexOp>(loc, dimSize);

    Value forResult = initOut;
    for (int64_t i = 1; i < dimSize; ++i) {
      // For each i, we build a helper that updates position [..., i, ...]
      // by reading current at [..., i] and prev at [..., i-1]
      // Actually tensor.extract/insert already handles coordinates.
      // We need to iterate over all combinations of non-dim coords.
      // For simplicity in a first pass, handle the common case:
      // since dimSize is known, iterate over all other dims.
      auto forOuter = [&]() -> Value {
        // Build a recursive or flat iteration over non-dim dimensions
        SmallVector<int64_t> nonDims;
        for (int64_t j = 0; j < inType.getRank(); ++j)
          if (j != dim) nonDims.push_back(j);
        int64_t nonTotal = 1;
        for (int64_t j = 0; j < (int64_t)nonDims.size(); ++j)
          nonTotal *= inType.getDimSize(nonDims[j]);
        if (nonTotal <= 0) return forResult;

        scf::ForOp outerFor = nullptr;
        Value iterVal = forResult;

        // For each non-dim dimension, build a nested scf.for
        // Since we need to iterate over all combinations, process
        // in a single scf.for over total_non_dim elements
        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value total = rewriter.create<arith::ConstantIndexOp>(loc, nonTotal);

        outerFor = rewriter.create<scf::ForOp>(loc, c0, total, c1, iterVal);
        auto outerIv = outerFor.getInductionVar();
        rewriter.setInsertionPointToStart(outerFor.getBody());
        Value curOut = outerFor.getInitArgs()[0];

        // Convert linear index to multi-dimensional coords
        // For position [..., i, ...]: we need to compute coordinates
        // where the dim axis has value i, and other axes come from linear index
        SmallVector<Value> curCoords(inType.getRank());
        Value remaining = outerIv;
        for (int64_t j = 0; j < (int64_t)inType.getRank(); ++j) {
          if (j == dim) {
            curCoords[j] = rewriter.create<arith::ConstantIndexOp>(loc, i);
          } else {
            int64_t dSize = inType.getDimSize(j);
            if (dSize > 1) {
              Value dSz = rewriter.create<arith::ConstantIndexOp>(loc, dSize);
              Value idx = rewriter.create<arith::RemSIOp>(loc, remaining, dSz);
              curCoords[j] = idx;
              remaining = rewriter.create<arith::DivSIOp>(loc, remaining, dSz);
            } else {
              curCoords[j] = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            }
          }
        }

        // Read prev = output[..., i-1, ...] and cur = input[..., i, ...]
        SmallVector<Value> prevCoords = curCoords;
        prevCoords[dim] = rewriter.create<arith::ConstantIndexOp>(loc, i - 1);
        Value prev = rewriter.create<tensor::ExtractOp>(loc, curOut, prevCoords);
        Value cur = rewriter.create<tensor::ExtractOp>(loc, input, curCoords);
        Value sum = rewriter.create<arith::AddFOp>(loc, prev, cur);
        Value newOut = rewriter.create<tensor::InsertOp>(loc, outType, sum, curOut, curCoords);
        rewriter.create<scf::YieldOp>(loc, newOut);

        rewriter.setInsertionPointAfter(outerFor);
        return outerFor.getResult(0);
      };
      forResult = forOuter();
    }

    rewriter.replaceOp(op, forResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Embedding → scf.for gather
//===----------------------------------------------------------------------===//

struct SfEmbeddingOpLowering : public OpConversionPattern<sf::EmbeddingOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::EmbeddingOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value weight = adaptor.getWeight();
    Value indices = adaptor.getIndices();
    Type rt = op.getResult().getType();
    auto wType = ::mlir::dyn_cast<::mlir::RankedTensorType>(weight.getType());
    auto idxType = ::mlir::dyn_cast<::mlir::RankedTensorType>(indices.getType());
    if (!wType || !idxType) return failure();
    if (wType.getRank() != 2) return failure();

    auto eltType = wType.getElementType();
    int64_t idxRank = idxType.getRank();

    // Use the sf op's result type directly (Python fixup ensures correct shape).
    auto correctType = cast<RankedTensorType>(rt);
    int64_t correctRank = correctType.getRank();
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < idxRank; ++i)
      if (correctType.isDynamicDim(i))
        dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, indices, i));
    // If embed dim is dynamic in rt, add its size at runtime
    if (correctRank > idxRank && correctType.isDynamicDim(idxRank))
      dynSizes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, wType.getDimSize(1)));
    Value empty = rewriter.create<tensor::EmptyOp>(loc, correctType, dynSizes);

    // Affine maps: indices (batch, seq) → output (batch, seq, embed)
    int64_t embedRank = idxRank + 1;
    SmallVector<AffineExpr> idxExprs;
    for (int64_t i = 0; i < idxRank; ++i)
      idxExprs.push_back(rewriter.getAffineDimExpr(i));
    auto indicesMap = AffineMap::get(embedRank, 0, idxExprs, rewriter.getContext());
    auto outMap = AffineMap::getMultiDimIdentityMap(embedRank, rewriter.getContext());

    SmallVector<utils::IteratorType> iterTypes(embedRank, utils::IteratorType::parallel);

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, correctType, ValueRange{indices}, ValueRange{empty},
        {indicesMap, outMap}, iterTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange bodyArgs) {
          // bodyArgs[0] = indices element at the current output position
          Value rawIdx = bodyArgs[0];
          Value embedIdx;
          if (isa<IntegerType>(rawIdx.getType())) {
            embedIdx = b.create<arith::IndexCastOp>(bodyLoc, b.getIndexType(), rawIdx);
          } else if (isa<FloatType>(rawIdx.getType())) {
            Value i64Idx = b.create<arith::FPToUIOp>(bodyLoc, b.getI64Type(), rawIdx);
            embedIdx = b.create<arith::IndexCastOp>(bodyLoc, b.getIndexType(), i64Idx);
          } else {
            embedIdx = b.create<arith::ConstantIndexOp>(bodyLoc, 0);
          }
          // Extract embedding row: weight[embedIdx, embed_dim]
          Value embedDim = b.create<linalg::IndexOp>(bodyLoc, embedRank - 1);
          Value wVal = b.create<tensor::ExtractOp>(bodyLoc, weight,
                                                     ValueRange{embedIdx, embedDim});
          b.create<linalg::YieldOp>(bodyLoc, wVal);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Index → scf.for gather (multi-index)
//===----------------------------------------------------------------------===//

struct SfIndexOpLowering : public OpConversionPattern<sf::IndexOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(sf::IndexOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // First operand is data, rest are indices
    ValueRange operands = op->getOperands();
    if (operands.size() < 2) return failure();
    Value data = adaptor.getInput();
    // For index tensors, we need them in order after the data input
    SmallVector<Value> indexTensors;
    for (size_t i = 1; i < operands.size(); ++i)
      indexTensors.push_back(operands[i]);

    Type rt = op.getResult().getType();
    auto outType = ::mlir::dyn_cast<::mlir::RankedTensorType>(rt);
    auto dataType = ::mlir::dyn_cast<::mlir::RankedTensorType>(data.getType());
    if (!outType || !dataType) return failure();

    auto eltType = dataType.getElementType();

    // Output shape determines the iteration space
    int64_t outNumel = 1;
    bool hasDynamic = false;
    for (int64_t i = 0; i < outType.getRank(); ++i) {
      int64_t d = outType.getDimSize(i);
      if (ShapedType::isDynamic(d)) { hasDynamic = true; break; }
      outNumel *= d;
    }

    Value empty = rewriter.create<tensor::EmptyOp>(loc, outType, ValueRange{});
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value total;
    if (hasDynamic) {
      total = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      // Dynamic: use size from first operand as total
      auto dataOpnd = op->getOperand(0);
      for (int64_t i = 0; i < outType.getRank(); ++i) {
        Value dimV = rewriter.create<tensor::DimOp>(loc, dataOpnd, i);
        if (i == 0) total = dimV;
        else total = rewriter.create<arith::MulIOp>(loc, total, dimV);
      }
    } else {
      total = rewriter.create<arith::ConstantIndexOp>(loc, outNumel);
    }

    auto forOp = rewriter.create<scf::ForOp>(loc, c0, total, c1, ValueRange{empty});
    Value iv = forOp.getInductionVar();
    Value curOut = forOp.getInitArgs()[0];

    rewriter.setInsertionPointToStart(forOp.getBody());

    // Convert linear index to multi-dimensional output coordinates
    SmallVector<Value> outCoords(outType.getRank());
    Value remaining = iv;
    for (int64_t j = 0; j < outType.getRank(); ++j) {
      // Use tensor.dim for each output dimension (handles dynamic dims)
      Value dSz = rewriter.create<tensor::DimOp>(loc, empty, j);
      outCoords[j] = rewriter.create<arith::RemSIOp>(loc, remaining, dSz);
      remaining = rewriter.create<arith::DivSIOp>(loc, remaining, dSz);
    }

    // Read values from index tensors. idxCoords must match index tensor's rank (not outType's).
    // For index tensors with rank > outType rank, pad remaining coords with 0.
    SmallVector<Value> dataCoords(dataType.getRank());
    for (size_t i = 0; i < indexTensors.size() && i < (size_t)dataType.getRank(); ++i) {
      auto idxTensorType = ::mlir::cast<::mlir::RankedTensorType>(indexTensors[i].getType());
      SmallVector<Value> idxCoords;
      for (int64_t j = 0; j < idxTensorType.getRank(); ++j) {
        if (j < (int64_t)outCoords.size())
          idxCoords.push_back(outCoords[j]);
        else
          idxCoords.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }
      Value rawIdx = rewriter.create<tensor::ExtractOp>(loc, indexTensors[i], idxCoords);
      if (isa<IntegerType>(rawIdx.getType()))
        dataCoords[i] = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), rawIdx);
      else
        dataCoords[i] = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    }

    Value val;
    if (dataType.getRank() == 0) {
      val = rewriter.create<tensor::ExtractOp>(loc, data, ValueRange{});
    } else {
      val = rewriter.create<tensor::ExtractOp>(loc, data, dataCoords);
    }
    Value newOut = rewriter.create<tensor::InsertOp>(loc, outType, val, curOut, outCoords);
    rewriter.create<scf::YieldOp>(loc, newOut);

    rewriter.setInsertionPointAfter(forOp);
    rewriter.replaceOp(op, forOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass 1: SfPromoteWeightsPass — weight/constant → func arguments
//===----------------------------------------------------------------------===//
// Must run before sf-lower-to-linalg. Uses collect-then-modify pattern
// (not walk-and-erase) to avoid potential iterator invalidation from
// modifying parent FuncOp types during a walk of its child ops.
//===----------------------------------------------------------------------===//

namespace {
struct SfPromoteWeightsPass
    : public PassWrapper<SfPromoteWeightsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfPromoteWeightsPass)

  StringRef getArgument() const final { return "sf-promote-weights"; }
  StringRef getDescription() const final {
    return "Promote sf.weight and sf.constant ops to function arguments";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    llvm::errs() << "  [sf-promote-weights] collecting weight ops\n";

    // Phase 1a: collect all weight ops first
    SmallVector<sf::WeightOp> weightOps;
    getOperation()->walk([&](sf::WeightOp op) {
      if (op->template getParentOfType<func::FuncOp>())
        weightOps.push_back(op);
    });

    llvm::errs() << "  [sf-promote-weights] found " << weightOps.size()
                 << " weight ops\n";

    // Phase 1b: promote each weight op (outside the walk)
    unsigned promoted = 0;
    for (auto op : weightOps) {
      auto parentFunc = op->template getParentOfType<func::FuncOp>();
      if (!parentFunc) continue;

      auto resultType = op.getResult().getType();
      if (!isa<RankedTensorType>(resultType)) continue;

      auto loc = op.getLoc();
      Block &entry = parentFunc.getBody().front();
      Value newArg = entry.insertArgument(
          entry.getNumArguments(), resultType, loc);

      auto origType = parentFunc.getFunctionType();
      SmallVector<Type> newInputs(origType.getInputs());
      newInputs.push_back(resultType);
      parentFunc.setType(
          FunctionType::get(&getContext(), newInputs, origType.getResults()));

      op.replaceAllUsesWith(newArg);
      op.erase();
      ++promoted;
    }

    LLVM_DEBUG(llvm::dbgs() << "[sf-promote-weights] Promoted "
                            << promoted << " weights\n");

    // Phase 2a: collect all constant ops
    SmallVector<sf::ConstantOp> constOps;
    getOperation()->walk([&](sf::ConstantOp op) {
      if (op->template getParentOfType<func::FuncOp>())
        constOps.push_back(op);
    });

    LLVM_DEBUG(llvm::dbgs() << "[sf-promote-weights] Found "
                            << constOps.size() << " constant ops\n");

    // Phase 2b: promote each constant op
    for (auto op : constOps) {
      auto parentFunc = op->template getParentOfType<func::FuncOp>();
      if (!parentFunc) continue;

      auto resultType = op.getResult().getType();
      if (!isa<RankedTensorType>(resultType)) continue;

      auto loc = op.getLoc();
      Block &entry = parentFunc.getBody().front();
      Value newArg = entry.insertArgument(
          entry.getNumArguments(), resultType, loc);

      auto origType = parentFunc.getFunctionType();
      SmallVector<Type> newInputs(origType.getInputs());
      newInputs.push_back(resultType);
      parentFunc.setType(
          FunctionType::get(&getContext(), newInputs, origType.getResults()));

      op.replaceAllUsesWith(newArg);
      op.erase();
    }

    LLVM_DEBUG(llvm::dbgs() << "[sf-promote-weights] Promoted "
                            << promoted << " weight + "
                            << constOps.size() << " constant ops\n");

    llvm::errs() << "  [sf-promote-weights] done, promoted " << promoted << " + "
                 << constOps.size() << " constants\n";

    // Phase 3: verify no remaining weight/constant ops
    bool hasRemaining = false;
    getOperation()->walk([&](Operation *op) {
      if (isa<sf::WeightOp>(op) || isa<sf::ConstantOp>(op)) {
        op->emitError("weight/constant not promoted");
        hasRemaining = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasRemaining) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::sf::createSfPromoteWeights() {
  return std::make_unique<SfPromoteWeightsPass>();
}

//===----------------------------------------------------------------------===//
// Pass 2: SfLowerToLinalgPass — lower all remaining sf ops via conversion
//===----------------------------------------------------------------------===//
// Assumes sf-promote-weights has already run. All weight/constant ops are
// already converted to func arguments at this point.
//===----------------------------------------------------------------------===//

namespace {
struct SfLowerToLinalgPass
    : public PassWrapper<SfLowerToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfLowerToLinalgPass)

  StringRef getArgument() const final { return "sf-lower-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower remaining sf dialect ops to linalg/arith/math/tensor/scf";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    math::MathDialect, tensor::TensorDialect,
                    func::FuncDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    llvm::errs() << "  [sf-lower-to-linalg] starting\n";

    // Quick check: no weight/constant should remain
    bool hasWeight = false;
    getOperation()->walk([&](Operation *op) {
      if (isa<sf::WeightOp>(op) || isa<sf::ConstantOp>(op)) {
        op->emitError("weight/constant found — sf-promote-weights must run first");
        hasWeight = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasWeight) {
      llvm::errs() << "  [sf-lower-to-linalg] ERROR: weights remain\n";
      signalPassFailure();
      return;
    }

    // Count sf ops before conversion
    int64_t sfCount = 0;
    getOperation()->walk([&](Operation *op) {
      if (op->getDialect() &&
          isa<sf::SfDialect>(op->getDialect()))
        ++sfCount;
    });
    llvm::errs() << "  [sf-lower-to-linalg] found " << sfCount
                 << " sf ops to lower\n";

    if (sfCount == 0) {
      llvm::errs() << "  [sf-lower-to-linalg] nothing to do\n";
      return;
    }

    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                            math::MathDialect, tensor::TensorDialect,
                            func::FuncDialect, scf::SCFDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalDialect<sf::SfDialect>();

    // sf::LogicalAndOp with i1 output is handled by IdentityLowering's type cast

    // Ops that return failure() for scalar/dynamic normalized dim must
    // remain dynamically legal in those rare cases.
    auto isScalar = [](Operation *op) {
      for (auto r : op->getResults())
        if (auto t = dyn_cast<RankedTensorType>(r.getType()))
          if (t.getRank() == 0) return true;
      return false;
    };
    // Only RMS norm needs dynamic dim fallback (LayerNorm handles it at runtime)
    auto hasDynamicNormalizedDim = [](Operation *op) {
      if (auto rnOp = dyn_cast<sf::RmsNormOp>(op))
        if (auto t = dyn_cast<RankedTensorType>(rnOp.getResult().getType()))
          if (t.getRank() > 0 && t.isDynamicDim(t.getRank() - 1)) return true;
      return false;
    };
    target.addDynamicallyLegalOp<sf::RmsNormOp>(hasDynamicNormalizedDim);
    // All binary/activation ops are fully lowered — Python type inference
    // now correctly produces broadcasted output shapes, so no dynamic
    // legal ops are needed.
    RewritePatternSet patterns(&getContext());
    // Register all lowering patterns
    patterns.add<SfBinaryLowering<sf::AddOp, arith::AddFOp>,
                 SfBinaryLowering<sf::MulOp, arith::MulFOp>,
                 SfBinaryLowering<sf::SubOp, arith::SubFOp>,
                 SfBinaryLowering<sf::DivOp, arith::DivFOp>,
                 SfBinaryLowering<sf::MaxOp, arith::MaxNumFOp>,
                 ReluLowering,
                 IdentityLowering,
                 SfActivationOpLowering<sf::GeluOp>,
                 SfActivationOpLowering<sf::SiluOp>,
                 SfActivationOpLowering<sf::SigmoidOp>,
                 SfActivationOpLowering<sf::ExpOp>,
                 SfActivationOpLowering<sf::NegOp>,
                 SfActivationOpLowering<sf::TanhOp>,
                 SfMatmulOpLowering,
                 SfLinearOpLowering,
                 SfViewOpLowering,
                 SfExpandOpLowering,
                 SfUnsqueezeOpLowering,
                 SfSumOpLowering,
                 SfTransposeOpLowering,
                 SfSliceOpLowering,
                 SfLeOpLowering,
                 SfLogicalAndOpLowering,
                 SfOnesLikeOpLowering,
                 SfNewOnesOpLowering,
                 SfLayerNormOpLowering,
                 SfRmsNormOpLowering,
                 SfScaledDotProductAttentionOpLowering,
                 SfEmbeddingOpLowering,
                 SfSymSizeOpLowering,
                 SfArangeOpLowering,
                 SfCumsumOpLowering,
                 SfIndexOpLowering>(&getContext());


    llvm::errs() << "  [sf-lower-to-linalg] running applyPartialConversion\n";
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      llvm::errs() << "  [sf-lower-to-linalg] CONVERSION FAILED\n";
      signalPassFailure();
    } else {
      llvm::errs() << "  [sf-lower-to-linalg] conversion succeeded\n";
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::sf::createSfLowerToLinalg() {
  return std::make_unique<SfLowerToLinalgPass>();
}
