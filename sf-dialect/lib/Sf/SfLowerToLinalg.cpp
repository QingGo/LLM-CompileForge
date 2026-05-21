#define NDEBUG
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

static Value makeEmpty(OpBuilder &b, Location loc, Type t, ValueRange inputs) {
  auto shaped = dyn_cast<ShapedType>(t);
  if (!shaped) return Value();
  SmallVector<Value> dynSizes;
  SmallVector<bool> filled(shaped.getRank(), false);
  auto idxType = b.getIndexType();
  for (auto input : inputs) {
    if (auto inType = dyn_cast<RankedTensorType>(input.getType())) {
      int64_t rankDiff = (int64_t)shaped.getRank() - (int64_t)inType.getRank();
      for (int64_t i = 0; i < (int64_t)shaped.getRank(); ++i) {
        int64_t inIdx = i - rankDiff;
        if (inIdx >= 0 && inIdx < (int64_t)inType.getRank()) {
          if (shaped.isDynamicDim(i) && inType.isDynamicDim(inIdx) && !filled[i]) {
            dynSizes.push_back(tensor::DimOp::create(b, loc, input,
                arith::ConstantIndexOp::create(b, loc, inIdx)));
            filled[i] = true;
          }
        }
      }
    }
  }
  // Fill remaining dynamic dims with a distinctive sentinel value (-1).
  // This prevents silent wrong-code generation: any pass that encounters
  // a -1 dynamic dim knows the size could not be inferred at lowering time.
  for (int64_t i = 0; i < (int64_t)shaped.getRank(); ++i)
    if (shaped.isDynamicDim(i) && !filled[i])
      dynSizes.push_back(arith::ConstantIndexOp::create(b, loc, -1));
  return tensor::EmptyOp::create(b, loc, shaped, dynSizes);
}

// Create tensor.empty initialized to zero (for reduction init tensors).
static Value makeZeroedEmpty(OpBuilder &b, Location loc, Type t, ValueRange inputs) {
  Value empty = makeEmpty(b, loc, t, inputs);
  if (!empty) return Value();
  auto eltType = cast<ShapedType>(t).getElementType();
  Value zero;
  if (isa<FloatType>(eltType)) {
    zero = arith::ConstantOp::create(b, loc, eltType,
        b.getFloatAttr(eltType, 0.0f));
  } else if (eltType.isInteger(64)) {
    zero = arith::ConstantOp::create(b, loc, eltType,
        b.getIntegerAttr(eltType, 0));
  } else {
    llvm::errs() << "  [makeZeroedEmpty] unsupported element type: " << eltType << "\n";
    return Value();
  }
  auto fill = linalg::FillOp::create(b, loc, ValueRange{zero}, ValueRange{empty});
  return fill.getResult(0);
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
      dynSizes.push_back(arith::ConstantOp::create(b, loc, idxType,
          b.getIndexAttr(shape[i])));  // will be replaced by proper dim op
  // For now, all dynamic dims get 0 as placeholder (must be overridden)
  return tensor::EmptyOp::create(b, loc, tensorType, dynSizes);
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

// Refine the result type of a binary/broadcast op based on actual operand types
// (which may have more static shape info after earlier lowering steps).
// Each dimension is resolved by taking the max (broadcast) of operand dims,
// preferring static sizes over dynamic when available.
static RankedTensorType refineBroadcastType(RankedTensorType resultType,
                                            ValueRange inputs) {
  auto rank = resultType.getRank();
  SmallVector<int64_t> refinedShape(rank, ShapedType::kDynamic);
  for (int64_t i = 0; i < rank; ++i) {
    if (!resultType.isDynamicDim(i)) {
      refinedShape[i] = resultType.getDimSize(i);
      continue;
    }
    // Dynamic dim — try to resolve from inputs.
    // Only refine if ALL inputs either broadcast (dim not present) or agree
    // on the same static size.  If any input has a truly dynamic dim, the
    // result must stay dynamic (the runtime value is unknown at compile time).
    int64_t bestSize = ShapedType::kDynamic;
    bool anyDynamic = false;
    for (auto input : inputs) {
      auto inType = dyn_cast<RankedTensorType>(input.getType());
      if (!inType || i >= inType.getRank()) continue;
      int64_t inSize = inType.getDimSize(i);
      if (inSize == ShapedType::kDynamic) { anyDynamic = true; continue; }
      bestSize = std::max(bestSize, inSize);
    }
    refinedShape[i] = anyDynamic ? ShapedType::kDynamic : (bestSize == ShapedType::kDynamic ? 1 : bestSize);
  }
  return RankedTensorType::get(refinedShape, resultType.getElementType());
}

// Safe constant creation — checks that the type is handled.
static Value createSafeConst(OpBuilder &b, Location loc, Type eltType, double floatVal, int64_t intVal = 0) {
  if (isa<FloatType>(eltType))
    return arith::ConstantOp::create(b, loc, eltType, b.getFloatAttr(eltType, floatVal));
  if (eltType.isInteger(64))
    return arith::ConstantOp::create(b, loc, eltType, b.getIntegerAttr(eltType, intVal));
  return Value();
}

static void populateBody(linalg::GenericOp op, PatternRewriter &rewriter,
                          function_ref<void(OpBuilder &, Location, ValueRange)> f) {
  auto guard = OpBuilder::InsertionGuard(rewriter);
  Block *body = rewriter.createBlock(&op.getRegion(), {});
  auto shaped = cast<ShapedType>(op->getResult(0).getType());
  auto eltTy = shaped.getElementType();
  unsigned numInputs = op.getNumDpsInputs();
  unsigned numOutputs = op.getNumDpsInits();
  for (unsigned i = 0; i < numInputs + numOutputs; ++i)
    body->addArgument(eltTy, op.getLoc());
  rewriter.setInsertionPointToEnd(body);
  f(rewriter, op.getLoc(), body->getArguments());
}

// Binary lowering with broadcast support via affine maps
template <typename SfOpTy, typename ArithOpTy>
struct SfBinaryLowering : public OpRewritePattern<SfOpTy> {
  using OpRewritePattern<SfOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(SfOpTy op, PatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    if (!isa<ShapedType>(resultType)) return failure();
    auto loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    auto rank = cast<ShapedType>(resultType).getRank();
    auto lhsRank = lhsType.getRank();
    auto rhsRank = rhsType.getRank();
    // If result rank is less than operand rank(s), the op's result type was
    // incorrectly inferred (e.g. tensor<f32> scalar while inputs are higher
    // rank). Use the higher-rank operand type as the result type.
    if (rank < lhsRank || rank < rhsRank) {
      auto maxRankType = (lhsRank >= rhsRank) ? lhsType : rhsType;
      resultType = RankedTensorType::get(maxRankType.getShape(),
                                          cast<ShapedType>(resultType).getElementType());
      rank = cast<ShapedType>(resultType).getRank();
    }
    // Promote non-float operands: when result is float but operand is int,
    // insert sitofp via linalg.generic (bufferizable DPS pattern).
    auto outEltTy = cast<ShapedType>(resultType).getElementType();
    auto promoteIfNeeded = [&](Value val, Type valEltTy, StringRef side) -> Value {
      if (isa<FloatType>(valEltTy)) return val;
      if (!isa<FloatType>(outEltTy)) {
        llvm::errs() << "  [SfBinary] SKIP (output not float, can't promote: "
                     << valEltTy << " -> " << outEltTy << ")\n";
        return Value();
      }
      auto valType = cast<RankedTensorType>(val.getType());
      auto promType = RankedTensorType::get(valType.getShape(), outEltTy);
      Value promInit = makeEmpty(rewriter, loc, promType, {val});
      if (!promInit) return Value();
      auto promRank = valType.getRank();
      SmallVector<utils::IteratorType> promIter(promRank, utils::IteratorType::parallel);
      auto promOp = linalg::GenericOp::create(rewriter, loc, promType,
          ValueRange{val}, promInit,
          identityMaps(promRank, 2, rewriter.getContext()), promIter,
          [&](OpBuilder &b, Location ploc, ValueRange args) {
            Value f = arith::SIToFPOp::create(b, ploc, outEltTy, args[0]);
            linalg::YieldOp::create(b, ploc, f);
          });
      llvm::errs() << "  [SfBinary] type=" << SfOpTy::getOperationName()
                   << " promoted " << side << " " << valEltTy << " -> f32\n";
      return promOp.getResult(0);
    };
    lhs = promoteIfNeeded(lhs, lhsType.getElementType(), "lhs");
    rhs = promoteIfNeeded(rhs, rhsType.getElementType(), "rhs");
    if (!lhs || !rhs) return failure();
    lhsType = cast<RankedTensorType>(lhs.getType());
    rhsType = cast<RankedTensorType>(rhs.getType());
    llvm::errs() << "  [SfBinary] type=" << SfOpTy::getOperationName() << " lhs=" << lhsType << " rhs=" << rhsType << " out=" << resultType << "\n";

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
          shapeVals.push_back(tensor::DimOp::create(rewriter, loc, val, squeeze + i));
        else
          shapeVals.push_back(arith::ConstantIndexOp::create(rewriter, loc, outShape[i]));
      }
      auto shapeType = RankedTensorType::get({(int64_t)shapeVals.size()}, rewriter.getIndexType());
      Value shape;
      if (shapeVals.empty())
        shape = tensor::EmptyOp::create(rewriter, loc, shapeType, ValueRange{});
      else
        shape = tensor::FromElementsOp::create(rewriter, loc, shapeType, shapeVals);
      auto reshaped = tensor::ReshapeOp::create(rewriter, loc, squeezedType, val, shape);
      llvm::errs() << "  [squeeze] " << valRank << "->" << rank << " OK\n";
      return reshaped.getResult();
    };
    Value newLhs = squeezeToRank(lhs, lhsRank);
    Value newRhs = squeezeToRank(rhs, rhsRank);
    if (!newLhs || !newRhs) { llvm::errs() << "  [squeeze] FAILED for sf binary\n"; return failure(); }
    lhs = newLhs; rhs = newRhs;
    lhsRank = cast<RankedTensorType>(lhs.getType()).getRank();
    rhsRank = cast<RankedTensorType>(rhs.getType()).getRank();

    // Refine output type from actual operand types (which may have more static
    // shape info after earlier lowering steps, e.g. sym_size→1xf32).
    auto refinedType = refineBroadcastType(
        cast<RankedTensorType>(resultType),
        ValueRange{lhs, rhs});
    Value empty = makeEmpty(rewriter, loc, refinedType, {lhs, rhs});
    if (!empty) { llvm::errs() << "  [SfBinary] makeEmpty failed\n"; return failure(); }

    // Build broadcast-aware affine maps for each operand
    auto lhsShaped = cast<RankedTensorType>(lhs.getType());
    auto rhsShaped = cast<RankedTensorType>(rhs.getType());
    auto lhsMap = broadcastMap(rank, lhsRank, rewriter.getContext(), lhsShaped.getShape());
    auto rhsMap = broadcastMap(rank, rhsRank, rewriter.getContext(), rhsShaped.getShape());
    auto outMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());

    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, refinedType, ValueRange{lhs, rhs}, empty,
        {lhsMap, rhsMap, outMap}, iterTypes);
    populateBody(generic, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value v = ArithOpTy::create(b, loc, args[0], args[1]);
      linalg::YieldOp::create(b, loc, v);
    });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

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



//===----------------------------------------------------------------------===//
// Embedding survives for now (true gather requires index-based lookup which is
// slow for large vocab. The linalg.generic pattern was evaluated and discarded
// because linalg doesn't support non-affine indexing (gather). A proper
// implementation would use tensor.extract or scf.for loops, but for now
// the Rust runtime handles embedding on the model level.)
//===----------------------------------------------------------------------===//

template <typename SfOpTy>
struct SfActivationOpLowering : public OpRewritePattern<SfOpTy> {
  using OpRewritePattern<SfOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(SfOpTy op, PatternRewriter &rewriter) const override {
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
      Value val;
      StringRef opName = SfOpTy::getOperationName();
      if (opName == "sf.gelu") {
        Value half = arith::ConstantOp::create(b, loc, eltType, b.getFloatAttr(eltType, 0.5));
        Value one = arith::ConstantOp::create(b, loc, eltType, b.getFloatAttr(eltType, 1.0));
        Value c1 = arith::ConstantOp::create(b, loc, eltType, b.getFloatAttr(eltType, 0.7978845608));
        Value c2 = arith::ConstantOp::create(b, loc, eltType, b.getFloatAttr(eltType, 0.044715));
        Value x = args[0]; Value x3 = arith::MulFOp::create(b, loc, x, x);
        x3 = arith::MulFOp::create(b, loc, x3, x);
        Value i1 = arith::MulFOp::create(b, loc, c2, x3);
        Value i2 = arith::AddFOp::create(b, loc, x, i1);
        Value sc = arith::MulFOp::create(b, loc, c1, i2);
        Value th = math::TanhOp::create(b, loc, sc);
        Value p1 = arith::AddFOp::create(b, loc, one, th);
        Value hx = arith::MulFOp::create(b, loc, half, x);
        val = arith::MulFOp::create(b, loc, hx, p1);
      } else if (opName == "sf.silu") {
        Value x = args[0]; Value neg = arith::NegFOp::create(b, loc, x);
        Value exp = math::ExpOp::create(b, loc, neg);
        Value one = arith::ConstantOp::create(b, loc, eltType, b.getFloatAttr(eltType, 1.0));
        Value denom = arith::AddFOp::create(b, loc, one, exp);
        Value sig = arith::DivFOp::create(b, loc, one, denom);
        val = arith::MulFOp::create(b, loc, x, sig);
      } else if (opName == "sf.sigmoid") {
        Value neg = arith::NegFOp::create(b, loc, args[0]);
        Value exp = math::ExpOp::create(b, loc, neg);
        Value one = arith::ConstantOp::create(b, loc, eltType, b.getFloatAttr(eltType, 1.0));
        Value denom = arith::AddFOp::create(b, loc, one, exp);
        val = arith::DivFOp::create(b, loc, one, denom);
      } else if (opName == "sf.exp") {
        val = math::ExpOp::create(b, loc, args[0]);
      } else if (opName == "sf.neg") {
        val = arith::NegFOp::create(b, loc, args[0]);
      } else if (opName == "sf.tanh") {
        val = math::TanhOp::create(b, loc, args[0]);
      } else { return; }
      linalg::YieldOp::create(b, loc, val);
    });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

// Matmul/Linear lowering
struct SfMatmulOpLowering : public OpRewritePattern<sf::MatmulOp> {
  using OpRewritePattern<sf::MatmulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::MatmulOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value lhs = op.getLhs(), rhs = op.getRhs();
    Type resultType = op.getResult().getType();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    int64_t lhsRank = lhsType.getRank(), rhsRank = rhsType.getRank();

    auto eltType = lhsType.getElementType();

    // Standard 2D matmul: use linalg.matmul
    if (lhsRank == 2 && rhsRank == 2) {
    Value empty = makeZeroedEmpty(rewriter, loc, resultType, {lhs});
      if (!empty) return failure();
      auto mo = linalg::MatmulOp::create(rewriter, loc, resultType,
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
    int64_t contractDimL = lhsRank - 1;  // K in lhs
    int64_t contractDimR = rhsRank - 2;  // K in rhs (second-to-last dim)
    int64_t outerRank = std::max(lhsRank - 1, rhsRank - 1) + 1; // M + N + batch
    SmallVector<int64_t> outShape;
    SmallVector<Value> dynSizes;
    auto resultRT = cast<RankedTensorType>(resultType);
    for (int64_t i = 0; i < resultRT.getRank(); ++i) {
      outShape.push_back(resultRT.getDimSize(i));
      if (resultRT.isDynamicDim(i))
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, resultRT.getRank() > lhsRank ? rhs : lhs, i));
    }
    while ((int64_t)outShape.size() < outerRank - 1)
      outShape.insert(outShape.begin(), 1);
    Value empty = makeZeroedEmpty(rewriter, loc, resultType, {lhs});
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
          Value mul = arith::MulFOp::create(b, bodyLoc, args[0], args[1]);
          Value add = arith::AddFOp::create(b, bodyLoc, args[2], mul);
          linalg::YieldOp::create(b, bodyLoc, add);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

struct SfLinearOpLowering : public OpRewritePattern<sf::LinearOp> {
  using OpRewritePattern<sf::LinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::LinearOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
  Value input = op.getInput();
  Value weight = op.getWeight();
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
    SmallVector<Value> t1Dyn; if (kDim < 0) t1Dyn.push_back(tensor::DimOp::create(rewriter, loc, input, 0));
    SmallVector<Value> tOutDyn; if (nDim < 0) tOutDyn.push_back(tensor::DimOp::create(rewriter, loc, weight, 1));
    Value pInput = tensor::ExpandShapeOp::create(rewriter, loc, t1, input, ArrayRef<ReassociationIndices>{{0, 1}});
    Value pEmpty = makeZeroedEmpty(rewriter, loc, tOut, {input});
    auto mo = linalg::MatmulOp::create(rewriter, loc, tOut, ValueRange{pInput, weight}, pEmpty);
    mo->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({2, 1}));
    Value mmr = mo.getResult(0);
    // Reshape from [1, N] to result type via tensor.reshape
    auto rtt = cast<RankedTensorType>(resultType);
    SmallVector<Value> sv;
    for (int64_t i = 0; i < rtt.getRank(); ++i) {
      if (rtt.isDynamicDim(i)) sv.push_back(tensor::DimOp::create(rewriter, loc, mmr, i == 0 ? 0 : 1));
      else sv.push_back(arith::ConstantIndexOp::create(rewriter, loc, rtt.getDimSize(i)));
    }
    auto st = RankedTensorType::get({(int64_t)sv.size()}, rewriter.getIndexType());
    Value sh = sv.empty() ? (Value)tensor::EmptyOp::create(rewriter, loc, st, ValueRange{})
                          : (Value)tensor::FromElementsOp::create(rewriter, loc, st, sv);
    rewriter.replaceOp(op, tensor::ReshapeOp::create(rewriter, loc, resultType, mmr, sh).getResult());
    return success();
  }

  auto inputRank = inputType.getRank();
  Value resultWeight;
  if (inputRank > 2 && wType.getRank() == 2) {
    // 3D input + 2D weight → batch_matmul. Weight needs to be [K, N] = [in, out].
    // The model stores weight as [out, in], so transpose to [in, out] first.
    SmallVector<int64_t> transShape = {wType.getDimSize(1), wType.getDimSize(0)};
    auto transType = RankedTensorType::get(transShape, eltType);
    auto emptyT = tensor::EmptyOp::create(rewriter, loc, transType, ValueRange{});
    SmallVector<unsigned> perm = {1u, 0u};
    SmallVector<utils::IteratorType> titer(2, utils::IteratorType::parallel);
    Value emptyTVal = emptyT;
    auto transposeOp = linalg::GenericOp::create(rewriter, loc, transType,
        ValueRange{weight}, ValueRange{emptyTVal},
        {AffineMap::getPermutationMap(perm, rewriter.getContext()),
         AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())}, titer);
    populateBody(transposeOp, rewriter, [&](OpBuilder &b, Location loc2, ValueRange args) {
      linalg::YieldOp::create(b, loc2, args[0]);
    });
    Value transW = transposeOp.getResult(0);
    // Broadcast transposed weight from 2D to 3D: [in, out] → [batch, in, out].
    Value batchDim = tensor::DimOp::create(rewriter, loc, input, 0);
    SmallVector<int64_t> w3dShape = {ShapedType::kDynamic,
                                       transType.getDimSize(0),
                                       transType.getDimSize(1)};
    auto w3dType = RankedTensorType::get(w3dShape, eltType);
    Value w3dEmpty = tensor::EmptyOp::create(rewriter, loc, w3dType, ValueRange{batchDim});
    SmallVector<utils::IteratorType> biter(3, utils::IteratorType::parallel);
    Value w3dEmptyVal = w3dEmpty;
    auto w3dOp = linalg::GenericOp::create(rewriter, loc, w3dType,
        ValueRange{transW}, ValueRange{w3dEmptyVal},
        {broadcastMap(3, 2, rewriter.getContext()),
         AffineMap::getMultiDimIdentityMap(3, rewriter.getContext())}, biter);
    populateBody(w3dOp, rewriter, [&](OpBuilder &b, Location loc2, ValueRange args) {
      linalg::YieldOp::create(b, loc2, args[0]);
    });
    resultWeight = w3dOp.getResult(0);
  } else {
    // 2D input + 2D weight → standard matmul. Transpose weight to [out, in].
    resultWeight = weight;
    if (wType.getRank() == 2) {
      SmallVector<int64_t> transShape = {wType.getDimSize(1), wType.getDimSize(0)};
      auto transType = RankedTensorType::get(transShape, eltType);
      auto emptyT = tensor::EmptyOp::create(rewriter, loc, transType, ValueRange{});
      SmallVector<unsigned> perm = {1u, 0u};
      SmallVector<utils::IteratorType> titer(2, utils::IteratorType::parallel);
      Value emptyTVal = emptyT;
      auto transposeOp = linalg::GenericOp::create(rewriter, loc, transType,
          ValueRange{weight}, ValueRange{emptyTVal},
          {AffineMap::getPermutationMap(perm, rewriter.getContext()),
           AffineMap::getMultiDimIdentityMap(2, rewriter.getContext())}, titer);
      populateBody(transposeOp, rewriter, [&](OpBuilder &b, Location loc2, ValueRange args) {
        linalg::YieldOp::create(b, loc2, args[0]);
      });
      resultWeight = transposeOp.getResult(0);
    }
  }

  Value empty = makeZeroedEmpty(rewriter, loc, resultType, {input});
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
    Value bmEmpty = makeZeroedEmpty(rewriter, loc, bmResultType, {input});
    if (!bmEmpty) { llvm::errs() << "  [SfLinear] bmEmpty failed\n"; return failure(); }
    llvm::errs() << "  [SfLinear] creating batch_matmul target=" << bmResultType << "\n";
    auto mo = linalg::BatchMatmulOp::create(rewriter, loc, bmResultType,
        ValueRange{input, resultWeight}, bmEmpty);
    mo->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({2, 1}));
    Value bmR = mo.getResult(0);
    // Squeeze 3D → 2D if needed
    if (resultTypeRT.getRank() != 3) {
      SmallVector<Value> sv;
      for (int64_t i = 0; i < resultTypeRT.getRank(); ++i) {
        if (resultTypeRT.isDynamicDim(i))
          sv.push_back(tensor::DimOp::create(rewriter, loc, bmR, i + 1));
        else
          sv.push_back(arith::ConstantIndexOp::create(rewriter, loc, resultTypeRT.getDimSize(i)));
      }
      auto st = RankedTensorType::get({(int64_t)sv.size()}, rewriter.getIndexType());
      Value sh = sv.empty() ? (Value)tensor::EmptyOp::create(rewriter, loc, st, ValueRange{})
                            : (Value)tensor::FromElementsOp::create(rewriter, loc, st, sv);
      result = tensor::ReshapeOp::create(rewriter, loc, resultType, bmR, sh).getResult();
    } else {
      result = bmR;
    }
  } else {
    // 2D matmul
    auto mo = linalg::MatmulOp::create(rewriter, loc, resultType,
        ValueRange{input, resultWeight}, empty);
    mo->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({2, 1}));
    result = mo.getResult(0);
  }
  rewriter.replaceOp(op, result);
  return success();
  }
};

// View → tensor reshape or expand/collapse
struct SfViewOpLowering : public OpRewritePattern<sf::ViewOp> {
  using OpRewritePattern<sf::ViewOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::ViewOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Value input = op.getInput();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inType || !outType) return failure();
    if (inType.getRank() == outType.getRank()) {
      rewriter.replaceOp(op, input);
      return success();
    }
    // Rank-changing view: use tensor.reshape with the correct shape.
    // The shape attribute tells which output dims come from dyn_shape operands,
    // which are static, and which are -1 (inferred from element count).
    auto shapeAttr = op->getAttrOfType<ArrayAttr>("shape");
    auto dynShapeOperands = op.getDynShape();

    // Pass 1: collect shape values and track the -1 (inferred) dimension.
    SmallVector<Value> shapeVals;
    int64_t inferredIdx = -1;
    int64_t dynIdx = 0;  // counter into dynShapeOperands

    for (int64_t i = 0; i < outType.getRank(); ++i) {
      if (!outType.isDynamicDim(i)) {
        // Static dim → constant index
        shapeVals.push_back(arith::ConstantIndexOp::create(rewriter, loc, outType.getDimSize(i)));
        continue;
      }
      // Dynamic dim — consult the shape attribute
      if (shapeAttr && i < (int64_t)shapeAttr.size()) {
        Attribute elem = shapeAttr[i];
        if (auto intAttr = dyn_cast<IntegerAttr>(elem)) {
          int64_t val = intAttr.getInt();
          if (val == -1) {
            inferredIdx = i;
            shapeVals.push_back(nullptr);  // placeholder for pass 2
          } else {
            shapeVals.push_back(arith::ConstantIndexOp::create(rewriter, loc, val));
          }
        } else if (dyn_cast<StringAttr>(elem)) {
          // SSA reference → use corresponding dyn_shape operand.
          // dyn_shape values are 0D f32 tensors (from sf.sym_size).
          // Extract the scalar and cast to index.
          if (dynIdx < (int64_t)dynShapeOperands.size()) {
            Value dynVal = dynShapeOperands[dynIdx++];
            auto dynTy = dyn_cast<RankedTensorType>(dynVal.getType());
            if (dynTy && dynTy.getRank() == 0) {
              Value extracted = tensor::ExtractOp::create(rewriter, loc, dynVal, ValueRange{});
              Value asInt = arith::FPToUIOp::create(rewriter, loc, rewriter.getIntegerType(64), extracted);
              dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), asInt);
            } else if (dynTy && dynTy.getRank() == 1 && dynTy.getDimSize(0) == 1) {
              Value extracted = tensor::ExtractOp::create(rewriter, loc, dynVal,
                  ValueRange{arith::ConstantIndexOp::create(rewriter, loc, 0)});
              if (dynTy.getElementType().isF32() || dynTy.getElementType().isF64()) {
                Value asInt = arith::FPToUIOp::create(rewriter, loc, rewriter.getIntegerType(64), extracted);
                dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), asInt);
              } else {
                dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), extracted);
              }
            } else if (!dynVal.getType().isIndex()) {
              dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), dynVal);
            }
            shapeVals.push_back(dynVal);
          } else {
            return failure();
          }
        } else {
          return failure();
        }
      } else {
        return failure();
      }
    }

    // Pass 2: compute the -1 inferred dimension if present.
    if (inferredIdx >= 0) {
      // Compute total input elements = product of input dims
      Value total = arith::ConstantIndexOp::create(rewriter, loc, 1);
      for (int64_t i = 0; i < inType.getRank(); ++i)
        total = arith::MulIOp::create(rewriter, loc, total,
                    tensor::DimOp::create(rewriter, loc, input, i));

      // Compute product of known output dims
      Value known = arith::ConstantIndexOp::create(rewriter, loc, 1);
      for (int64_t i = 0; i < outType.getRank(); ++i) {
        if (i != inferredIdx && shapeVals[i])
          known = arith::MulIOp::create(rewriter, loc, known, shapeVals[i]);
      }

      // Inferred dim value: total / known (must be exact)
      shapeVals[inferredIdx] = arith::DivUIOp::create(rewriter, loc, total, known);
    }

    auto shapeTensorType = RankedTensorType::get({(int64_t)shapeVals.size()},
                                                  rewriter.getIndexType());
    auto shapeTensor = tensor::FromElementsOp::create(rewriter, loc, shapeTensorType, shapeVals);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, outType, input, shapeTensor);
    return success();
  }
};
struct SfExpandOpLowering : public OpRewritePattern<sf::ExpandOp> {
  using OpRewritePattern<sf::ExpandOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::ExpandOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Value input = op.getInput();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inType || !outType) return failure();
    int64_t outRank = outType.getRank(), inRank = inType.getRank();

    // Build tensor.empty with dynamic dims from shape attribute / dyn_shape operands
    auto shapeAttr = op->getAttrOfType<ArrayAttr>("shape");
    auto dynShapeOperands = op.getDynShape();
    SmallVector<Value> dynSizes;
    int64_t dynIdx = 0;
    for (int64_t i = 0; i < outRank; ++i) {
      if (!outType.isDynamicDim(i)) continue;
      if (shapeAttr && i < (int64_t)shapeAttr.size()) {
        Attribute elem = shapeAttr[i];
        if (dyn_cast<StringAttr>(elem)) {
          // SSA reference → dyn_shape operand
          if (dynIdx >= (int64_t)dynShapeOperands.size()) return failure();
          Value dynVal = dynShapeOperands[dynIdx++];
          auto dynTy = dyn_cast<RankedTensorType>(dynVal.getType());
          if (dynTy && dynTy.getRank() == 0) {
            Value extracted = tensor::ExtractOp::create(rewriter, loc, dynVal, ValueRange{});
            Value asInt = arith::FPToUIOp::create(rewriter, loc, rewriter.getIntegerType(64), extracted);
            dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), asInt);
          } else if (dynTy && dynTy.getRank() == 1 && dynTy.getDimSize(0) == 1) {
            Value extracted = tensor::ExtractOp::create(rewriter, loc, dynVal,
                ValueRange{arith::ConstantIndexOp::create(rewriter, loc, 0)});
            if (dynTy.getElementType().isF32() || dynTy.getElementType().isF64()) {
              Value asInt = arith::FPToUIOp::create(rewriter, loc, rewriter.getIntegerType(64), extracted);
              dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), asInt);
            } else {
              dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), extracted);
            }
          } else if (!dynVal.getType().isIndex()) {
            dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), dynVal);
          }
          dynSizes.push_back(dynVal);
        } else if (auto intAttr = dyn_cast<IntegerAttr>(elem)) {
          int64_t val = intAttr.getInt();
          if (val == -1) {
            // -1 means "keep input dim at this position" → get from input
            int64_t inIdx = i - (outRank - inRank);
            if (inIdx >= 0 && inIdx < inRank && inType.isDynamicDim(inIdx))
              dynSizes.push_back(tensor::DimOp::create(rewriter, loc, input, inIdx));
            else
              dynSizes.push_back(arith::ConstantIndexOp::create(rewriter, loc, 1));
          } else {
            dynSizes.push_back(arith::ConstantIndexOp::create(rewriter, loc, val));
          }
        }
      }
    }
    Value empty = tensor::EmptyOp::create(rewriter, loc, outType, dynSizes);

    // linalg.generic with broadcast: input maps to trailing output dims.
    // Size-1 input dims must use affine constant 0, not the loop dim,
    // to be compatible with linalg-to-loops conversion.
    SmallVector<AffineExpr> inExprs;
    for (int64_t i = 0; i < outRank; ++i) {
      int64_t inIdx = i - (outRank - inRank);
      if (inIdx < 0) continue;
      int64_t outDimSize = outType.getDimSize(i);
      int64_t inDimSize = inType.getDimSize(inIdx);
      if (inDimSize == 1 && (outDimSize == ShapedType::kDynamic || outDimSize > 1))
        inExprs.push_back(getAffineConstantExpr(0, rewriter.getContext()));
      else
        inExprs.push_back(getAffineDimExpr(i, rewriter.getContext()));
    }
    auto inMap = AffineMap::get(outRank, 0, inExprs, rewriter.getContext());
    auto outMap = AffineMap::getMultiDimIdentityMap(outRank, rewriter.getContext());
    SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
    auto g = linalg::GenericOp::create(rewriter, loc, outType,
        ValueRange{input}, ValueRange{empty}, {inMap, outMap}, iters);
    populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      linalg::YieldOp::create(b, loc, args[0]);
    });
    rewriter.replaceOp(op, g.getResult(0));
    return success();
  }
};
struct SfUnsqueezeOpLowering : public OpRewritePattern<sf::UnsqueezeOp> {
  using OpRewritePattern<sf::UnsqueezeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::UnsqueezeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Value input = op.getInput();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inType || !outType) { rewriter.replaceOp(op, input); return success(); }
    if (inType.getRank() == outType.getRank()) {
      rewriter.replaceOp(op, input); return success();
    }
    // Rank-changing: use tensor.reshape with output shape values.
    // For unsqueeze, dims before `dim` map 1:1, a new 1 is inserted at `dim`,
    // and dims after `dim` are shifted by 1.
    int64_t unsqueezeDim = 0;
    if (auto dimAttr = op->getAttrOfType<IntegerAttr>("dim"))
      unsqueezeDim = dimAttr.getInt();
    if (unsqueezeDim < 0) {
      unsqueezeDim += inType.getRank() + 1;  // +1 because unsqueeze adds a dimension
    }
    SmallVector<Value> shapeVals;
    for (int64_t i = 0; i < outType.getRank(); ++i) {
      if (outType.isDynamicDim(i)) {
        if (i < unsqueezeDim)
          shapeVals.push_back(tensor::DimOp::create(rewriter, loc, input, i));
        else if (i == unsqueezeDim)
          shapeVals.push_back(arith::ConstantIndexOp::create(rewriter, loc, 1));
        else
          shapeVals.push_back(tensor::DimOp::create(rewriter, loc, input, i - 1));
      } else {
        shapeVals.push_back(arith::ConstantIndexOp::create(rewriter, loc, outType.getDimSize(i)));
      }
    }
    auto shapeTensorType = RankedTensorType::get({(int64_t)shapeVals.size()},
                                                  rewriter.getIndexType());
    auto shapeTensor = tensor::FromElementsOp::create(rewriter, loc, shapeTensorType, shapeVals);
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, outType, input, shapeTensor);
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

//===----------------------------------------------------------------------===//
// Broadcast-aware helpers for linalg.generic creation
//===----------------------------------------------------------------------===//

static Value makeBinaryOp(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                           Type outType, PatternRewriter &rewriter,
                           function_ref<Value(OpBuilder &, Location, Value, Value)> fn) {
  Value empty = makeEmpty(builder, loc, outType, {lhs, rhs});
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
      lhsExprs.push_back((lhsDim == 1 && (outDim == ShapedType::kDynamic || outDim > 1))
          ? getAffineConstantExpr(0, builder.getContext())
          : getAffineDimExpr(i, builder.getContext()));
    }
    if (rhsI >= 0) {
      int64_t rhsDim = rhsType.getDimSize(rhsI);
      rhsExprs.push_back((rhsDim == 1 && (outDim == ShapedType::kDynamic || outDim > 1))
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
  populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
    Value v = fn(b, loc, args[0], args[1]);
    linalg::YieldOp::create(b, loc, v);
  });
  return g.getResult(0);
}

static Value makeUnaryOp(OpBuilder &builder, Location loc, Value in, Type outType,
                          PatternRewriter &rewriter,
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
      inExprs.push_back((inDim == 1 && (outDim == ShapedType::kDynamic || outDim > 1))
          ? getAffineConstantExpr(0, builder.getContext())
          : getAffineDimExpr(i, builder.getContext()));
    }
  }
  auto inMap = AffineMap::get(outRank, 0, inExprs, builder.getContext());
  auto outMap = AffineMap::getMultiDimIdentityMap(outRank, builder.getContext());

  SmallVector<utils::IteratorType> iters(outRank, utils::IteratorType::parallel);
  auto g = linalg::GenericOp::create(rewriter, loc, outType,
      in, empty, {inMap, outMap}, iters);
  populateBody(g, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
    Value v = fn(b, loc, args[0]);
    linalg::YieldOp::create(b, loc, v);
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
    : public OpRewritePattern<sf::ScaledDotProductAttentionOp> {
  using OpRewritePattern<sf::ScaledDotProductAttentionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::ScaledDotProductAttentionOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value Q = op.getQuery();
    Value K = op.getKey();
    Value V = op.getValue();

    auto qType = ::mlir::dyn_cast<::mlir::RankedTensorType>(Q.getType());
    if (!qType || qType.getRank() < 3) return failure();

    auto eltType = qType.getElementType();
    int64_t rank = qType.getRank();
  int64_t dk = qType.getDimSize(rank - 1);
  if (dk < 0) return failure();
  // Use explicit scale attribute if present; otherwise default to 1/sqrt(d_k)
  float scaleVal = 1.0f;
  if (auto scaleAttr = op->getAttrOfType<mlir::FloatAttr>("scale"))
    scaleVal = scaleAttr.getValueAsDouble();
  else
    scaleVal = 1.0f / std::sqrt(static_cast<float>(dk));
  auto ctx = rewriter.getContext();

  // Phase 1: tiled online softmax decision
  // For seq_kv > 64 or dynamic, use tiled attention to avoid O(seq²) memory.
  // (Implementation placeholder — currently falls through to standard path.)
  auto kType = ::mlir::dyn_cast<::mlir::RankedTensorType>(K.getType());
  int64_t seqKVDim = kType ? kType.getDimSize(rank - 2) : ShapedType::kDynamic;
  if (seqKVDim == ShapedType::kDynamic || seqKVDim > 64) {
    // Tiled attention needed — emit scf.for with online softmax.
    // For Phase 1 this falls back to standard attention (which creates
    // O(seq²) intermediate buffers but produces correct results).
    // The full tiled implementation will be added in a follow-up.
  }

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
          dyns.push_back(tensor::DimOp::create(rewriter, loc, K, rank - 2));
        else
          dyns.push_back(tensor::DimOp::create(rewriter, loc, K, i));
      }
      return dyns;
    };
    Value ktEmpty = tensor::EmptyOp::create(rewriter, loc, ktType, ktDyn());
    SmallVector<unsigned> ktPerm(rank);
    for (int64_t i = 0; i < rank; ++i) ktPerm[i] = i;
    std::swap(ktPerm[rank - 1], ktPerm[rank - 2]);
    SmallVector<utils::IteratorType> ktIterTypes(rank, utils::IteratorType::parallel);
    auto ktOp = linalg::GenericOp::create(rewriter, loc, ktType, K, ktEmpty,
        {AffineMap::getPermutationMap(ktPerm, rewriter.getContext()),
         AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())}, ktIterTypes);
    populateBody(ktOp, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      linalg::YieldOp::create(b, loc, args[0]);
    });
    Value Kt = ktOp.getResult(0);

    // Helper: scores-type shapes (last dim = k_seq from K dim rank-2)
    auto scoresDyn = [&](RankedTensorType type) -> SmallVector<Value> {
      SmallVector<Value> dyns;
      for (int64_t i = 0; i < rank; ++i) {
        if (!type.isDynamicDim(i)) continue;
        if (i == rank - 1)
          dyns.push_back(tensor::DimOp::create(rewriter, loc, K, rank - 2));
        else
          dyns.push_back(tensor::DimOp::create(rewriter, loc, Q, i));
      }
      return dyns;
    };

    // Step 2: scores = matmul(Q, K^T) via linalg.generic (supports any rank)
    SmallVector<int64_t> scoresShape(qType.getShape());
    scoresShape[rank - 1] = qType.getDimSize(rank - 2);
    auto scoresType = RankedTensorType::get(scoresShape, eltType);
    Value scoresInit = tensor::EmptyOp::create(rewriter, loc, scoresType, scoresDyn(scoresType));
    Value scoresZero = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, 0.0f));
    Value scoresEmpty = linalg::FillOp::create(rewriter, loc,
        ValueRange{scoresZero}, ValueRange{scoresInit}).getResult(0);
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
            Value _mul = arith::MulFOp::create(b, bodyLoc, args[0], args[1]);
            Value _add = arith::AddFOp::create(b, bodyLoc, args[2], _mul);
            linalg::YieldOp::create(b, bodyLoc, _add);
          });
      scores = scoreGeneric.getResult(0);
    }

    // Step 3: scores_scaled = scores * (1/sqrt(d_k))
    Value scaleConst = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, scaleVal));
    Value scaledEmpty = tensor::EmptyOp::create(rewriter, loc, scoresType, scoresDyn(scoresType));
    SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
    auto scaleOp = linalg::GenericOp::create(rewriter, loc, scoresType,
        scores, scaledEmpty,
        identityMaps(rank, 2, rewriter.getContext()), iterTypes);
    populateBody(scaleOp, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value _scaled = arith::MulFOp::create(b, loc, args[0], scaleConst); linalg::YieldOp::create(b, loc, _scaled);
    });
    Value scoresScaled = scaleOp.getResult(0);

    // Step 3b: Apply attention mask if present
    // mask contains POSITION VALUES (0, 1, 2, 3...) — NOT booleans.
    // mask shape: [batch, 1, seq, seq], all values in row i = position[i].
    // Causal condition: position[row] >= position[col] → attend (0.0), else → -inf.
    // Approach: transpose mask (swap last two dims) so maskT[i,j] = mask[j,i] = position[j],
    // then compare: mask >= maskT  =>  position[i] >= position[j].
    if (Value mask = op.getAttnMask()) {
      Value zeroF32 = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, 0.0f));
      Value negLarge = arith::ConstantOp::create(rewriter, loc, eltType,
          rewriter.getFloatAttr(eltType, -1.0e20f));

      // Step 3b-i: Transpose mask (swap last two dims)
      auto maskType = cast<RankedTensorType>(mask.getType());
      int64_t maskRank = maskType.getRank();
      SmallVector<unsigned> perm;
      for (int64_t i = 0; i < maskRank; ++i) perm.push_back(i);
      std::swap(perm[maskRank - 1], perm[maskRank - 2]);

      SmallVector<int64_t> maskTShape;
      for (int64_t i = 0; i < maskRank; ++i)
        maskTShape.push_back(maskType.getDimSize(perm[i]));
      auto maskTType = RankedTensorType::get(maskTShape, maskType.getElementType());

      // Build dynamic sizes for the transposed tensor (correctly remapped via perm)
      SmallVector<Value> maskTDynSizes;
      for (int64_t i = 0; i < maskRank; ++i) {
        if (maskTType.isDynamicDim(i))
          maskTDynSizes.push_back(tensor::DimOp::create(rewriter, loc, mask, perm[i]));
      }
      Value maskTEmpty = tensor::EmptyOp::create(rewriter, loc, maskTType, maskTDynSizes);

      SmallVector<utils::IteratorType> tIter(maskRank, utils::IteratorType::parallel);
      auto transposeOp = linalg::GenericOp::create(rewriter, loc, maskTType,
          ValueRange{mask}, ValueRange{maskTEmpty},
          {AffineMap::getPermutationMap(perm, rewriter.getContext()),
           AffineMap::getMultiDimIdentityMap(maskRank, rewriter.getContext())}, tIter);
      populateBody(transposeOp, rewriter, [&](OpBuilder &b, Location loc2, ValueRange args) {
        linalg::YieldOp::create(b, loc2, args[0]);
      });
      Value maskT = transposeOp.getResult(0);

      // Step 3b-ii: additive[i,j] = (mask[i,j] >= maskT[i,j]) ? 0.0 : -inf
      // makeBinaryOp broadcasts maskT from [b,1,s,s] -> scoresScaled shape [b,h,s,s]
      Value additive = makeBinaryOp(rewriter, loc, mask, maskT, scoresScaled.getType(), rewriter,
          [&](OpBuilder &b, Location loc, Value m, Value mT) {
            Value cmp = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGE, m, mT);
            Value sel = arith::SelectOp::create(b, loc, cmp, zeroF32, negLarge);
            return sel;
          });
      // Step 3b-iii: scoresScaled = scoresScaled + additive
      scoresScaled = makeBinaryOp(rewriter, loc, scoresScaled, additive, scoresScaled.getType(), rewriter,
          [&](OpBuilder &b, Location loc, Value a, Value bVal) {
            return arith::AddFOp::create(b, loc, a, bVal);
          });
    }

    // Step 4: softmax along last dim
    // softmax(x) = exp(x - max) / sum(exp(x - max))
    int64_t lastDim = rank - 1;

    // Helper: max-type shapes (last dim = 1, static; leading dims from Q)
    auto maxDyn = [&](RankedTensorType type) -> SmallVector<Value> {
      SmallVector<Value> dyns;
      for (int64_t i = 0; i < rank; ++i) {
        if (!type.isDynamicDim(i)) continue;
        if (i == rank - 2)
          dyns.push_back(tensor::DimOp::create(rewriter, loc, Q, rank - 2));
        else
          dyns.push_back(tensor::DimOp::create(rewriter, loc, Q, i));
      }
      return dyns;
    };

    // 4a: max reduction along last dim
    SmallVector<int64_t> maxShape(scoresShape);
    maxShape[lastDim] = 1;
    auto maxType = RankedTensorType::get(maxShape, eltType);
    // Softmax max reduction: init to -inf
    Value negInf = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, -1.0e20f));
    Value maxEmpty = tensor::EmptyOp::create(rewriter, loc, maxType, maxDyn(maxType));
    auto fillMax = linalg::FillOp::create(rewriter, loc, ValueRange{negInf}, ValueRange{maxEmpty});
    maxEmpty = fillMax.getResult(0);
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
    populateBody(maxOp, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value _mx = arith::MaxNumFOp::create(b, loc, args[0], args[1]); linalg::YieldOp::create(b, loc, _mx);
    });
    Value maxVal = maxOp.getResult(0);

    // 4b: sub = x - max (broadcast)
    Value sub = makeBinaryOp(rewriter, loc, scoresScaled, maxVal, scoresType, rewriter,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::SubFOp::create(b, loc, a, bVal);
        });

    // 4c: exp(x - max)
    Value expVal = makeUnaryOp(rewriter, loc, sub, scoresType, rewriter,
        [&](OpBuilder &b, Location loc, Value v) {
          return math::ExpOp::create(b, loc, v);
        });

    // 4d: sum reduction
    Value sumInit = tensor::EmptyOp::create(rewriter, loc, maxType, maxDyn(maxType));
    Value sumZero = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, 0.0f));
    Value sumEmpty = linalg::FillOp::create(rewriter, loc,
        ValueRange{sumZero}, ValueRange{sumInit}).getResult(0);
    auto sumOp = linalg::GenericOp::create(rewriter, loc, maxType, expVal, sumEmpty,
        {AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()),
         AffineMap::get(rank, 0,
             llvm::map_to_vector(llvm::seq<int64_t>(0, rank), [&](int64_t i) -> AffineExpr {
               return (i == lastDim) ? getAffineConstantExpr(0, rewriter.getContext())
                                     : getAffineDimExpr(i, rewriter.getContext());
             }), rewriter.getContext())}, reduIters);
    populateBody(sumOp, rewriter, [&](OpBuilder &b, Location loc, ValueRange args) {
      Value _ad = arith::AddFOp::create(b, loc, args[0], args[1]); linalg::YieldOp::create(b, loc, _ad);
    });
    Value sumVal = sumOp.getResult(0);

    // 4e: softmax = exp / sum (broadcast)
    Value attn = makeBinaryOp(rewriter, loc, expVal, sumVal, scoresType, rewriter,
        [&](OpBuilder &b, Location loc, Value a, Value bVal) {
          return arith::DivFOp::create(b, loc, a, bVal);
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
        outDyn.push_back(tensor::DimOp::create(rewriter, loc, V, rank - 1));
      else if (i == rank - 2)
        outDyn.push_back(tensor::DimOp::create(rewriter, loc, Q, rank - 2));
      else
        outDyn.push_back(tensor::DimOp::create(rewriter, loc, Q, i));
    }
    Value outInit = tensor::EmptyOp::create(rewriter, loc, outEmptyType, outDyn);
    Value outZero = arith::ConstantOp::create(rewriter, loc, eltType,
        rewriter.getFloatAttr(eltType, 0.0f));
    Value outEmpty = linalg::FillOp::create(rewriter, loc,
        ValueRange{outZero}, ValueRange{outInit}).getResult(0);
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
            Value _mul = arith::MulFOp::create(b, bodyLoc, args[0], args[1]);
            Value _add = arith::AddFOp::create(b, bodyLoc, args[2], _mul);
            linalg::YieldOp::create(b, bodyLoc, _add);
          });
      attnVResult = outGeneric.getResult(0);
    }

    rewriter.replaceOp(op, attnVResult);
    return success();
  }
};

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
struct SfTransposeOpLowering : public OpRewritePattern<sf::TransposeOp> {
  using OpRewritePattern<sf::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::TransposeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Value input = op.getInput();
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

    // Build inverse permutation: for each output dim j, which input dim
    // provides its size.  makeEmpty({input}) at line 1437 used same-index
    // matching which is WRONG for transpose — dynamic dims move positions.
    SmallVector<int64_t> invPerm(rank);
    for (int64_t i = 0; i < rank; ++i)
      invPerm[perm[i]] = i;

    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (!rt.isDynamicDim(i)) continue;
      int64_t srcDim = invPerm[i];
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, input, srcDim));
    }
    Value empty = tensor::EmptyOp::create(rewriter, loc, rt, dynSizes);
    if (!empty) return failure();

    auto transposeOp = linalg::TransposeOp::create(rewriter, 
        loc, input, empty, rewriter.getDenseI64ArrayAttr(perm));
    rewriter.replaceOp(op, transposeOp->getResult(0));
    return success();
  }
};

// Slice → tensor.extract_slice
struct SfSliceOpLowering : public OpRewritePattern<sf::SliceOp> {
  using OpRewritePattern<sf::SliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::SliceOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc(); Value input = op.getInput();
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
        Value dimVal = tensor::DimOp::create(rewriter, loc, input, i);
        Value startVal = arith::ConstantIndexOp::create(rewriter, loc, start);
        szs.push_back(Value(arith::SubIOp::create(rewriter, loc, dimVal, startVal).getResult()));
      } else if (inType.isDynamicDim(i)) {
        szs.push_back(Value(tensor::DimOp::create(rewriter, loc, input, i).getResult()));
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
// SymSize → tensor.dim + cast + tensor.insert
//===----------------------------------------------------------------------===//

struct SfSymSizeOpLowering : public OpRewritePattern<sf::SymSizeOp> {
  using OpRewritePattern<sf::SymSizeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::SymSizeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Type rt = op.getResult().getType();
    auto inputType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    if (!inputType) return failure();

    int64_t dim = 0;
    if (auto dimAttr = op.getOperation()->getAttrOfType<IntegerAttr>("dim"))
      dim = dimAttr.getInt();
    if (dim < 0 || dim >= inputType.getRank()) return failure();

    Value dimVal = tensor::DimOp::create(rewriter, loc, input, dim);
    Value dimI64 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), dimVal);
    auto f32Type = rewriter.getF32Type();
    Value dimF32 = arith::UIToFPOp::create(rewriter, loc, f32Type, dimI64);
    RankedTensorType outTensorType = RankedTensorType::get({1}, f32Type);
    Value empty = tensor::EmptyOp::create(rewriter, loc, outTensorType, ValueRange{});
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value result = tensor::InsertOp::create(rewriter, loc, dimF32, empty, ValueRange{c0});
    rewriter.replaceOp(op, result);
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
    bool outputWasPromoted = false;
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
    auto fillOp = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroInit}, ValueRange{rawEmpty});
    Value empty = fillOp.getResult(0);

    // scf.for %i = 0 to N
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value zeroI64 = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
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

//===----------------------------------------------------------------------===//
// Cumsum → scf.for loop accumulation along dim
//===----------------------------------------------------------------------===//

struct SfCumsumOpLowering : public OpRewritePattern<sf::CumsumOp> {
  using OpRewritePattern<sf::CumsumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::CumsumOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    Type rt = op.getResult().getType();
    auto outType = ::mlir::dyn_cast<::mlir::RankedTensorType>(rt);
    auto inType = ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    if (!inType || !outType) return failure();

    int64_t dim = 0;
    if (auto dimAttr = op.getOperation()->getAttrOfType<IntegerAttr>("dim"))
      dim = dimAttr.getInt();
    if (dim < 0 || dim >= inType.getRank())
      return failure();

    auto eltType = inType.getElementType();
    int64_t rank = inType.getRank();

    // Copy input to output first.
    Value empty = makeEmpty(rewriter, loc, outType, {input});
    if (!empty) return failure();
    Value initOut = linalg::CopyOp::create(rewriter, loc, input, empty).getResult(0);

    // Get runtime dim size for the cumsum axis
    Value dimSize;
    bool dimIsStatic = (inType.getDimSize(dim) > 0);
    if (dimIsStatic) {
      dimSize = arith::ConstantIndexOp::create(rewriter, loc, inType.getDimSize(dim));
    } else {
      dimSize = tensor::DimOp::create(rewriter, loc, input, dim);
    }

    // Build non-dim runtime sizes for linearization
    SmallVector<Value> nonDimSizes;
    SmallVector<int64_t> nonDimIdxs;
    Value nonTotalVal;
    bool nonDimAllStatic = true;
    int64_t nonTotalStatic = 1;
    for (int64_t j = 0; j < rank; ++j) {
      if (j == dim) continue;
      nonDimIdxs.push_back(j);
      int64_t sz = inType.getDimSize(j);
      if (sz > 0) {
        nonDimSizes.push_back(arith::ConstantIndexOp::create(rewriter, loc, sz));
        nonTotalStatic *= sz;
      } else {
        nonDimAllStatic = false;
        Value dynSz = tensor::DimOp::create(rewriter, loc, input, j);
        nonDimSizes.push_back(dynSz);
      }
    }

    if (nonDimAllStatic && nonTotalStatic <= 0) {
      rewriter.replaceOp(op, initOut);
      return success();
    }

    // Compute total non-dim elements at runtime if any dim is dynamic
    if (nonDimAllStatic) {
      nonTotalVal = arith::ConstantIndexOp::create(rewriter, loc, nonTotalStatic);
    } else {
      nonTotalVal = nonDimSizes[0];
      for (size_t s = 1; s < nonDimSizes.size(); ++s) {
        nonTotalVal = arith::MulIOp::create(rewriter, loc, nonTotalVal, nonDimSizes[s]);
      }
    }

    // For i = 1 to dimSize-1: cumsum along dim
    // scf.for %i = 1 to dimSize
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto dimLoop = scf::ForOp::create(rewriter, loc, c1, dimSize, c1, initOut);
    Value iv = dimLoop.getInductionVar();
    rewriter.setInsertionPointToStart(dimLoop.getBody());
    Value dimIterOut = dimLoop.getBody()->getArgument(1);  // loop-carried tensor

    // Inner loop: iterate over all non-dim positions
    auto innerLoop = scf::ForOp::create(rewriter, loc, c0, nonTotalVal, c1, dimIterOut);
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    Value innerIv = innerLoop.getInductionVar();
    Value curOut = innerLoop.getBody()->getArgument(1);

    // Linear index -> multi-dimensional coords
    SmallVector<Value> coords(rank);
    Value remaining = innerIv;
    for (int64_t j = 0; j < rank; ++j) {
      if (j == dim) {
        coords[j] = iv;
      } else {
        // Find the index in nonDimIdxs
        int64_t localIdx = -1;
        for (size_t si = 0; si < nonDimIdxs.size(); ++si) {
          if (nonDimIdxs[si] == j) { localIdx = si; break; }
        }
        if (localIdx < 0) { coords[j] = c0; continue; }
        Value dSz = nonDimSizes[localIdx];
        Value idx = arith::RemSIOp::create(rewriter, loc, remaining, dSz);
        coords[j] = idx;
        remaining = arith::DivSIOp::create(rewriter, loc, remaining, dSz);
      }
    }

    // prev = curOut[..., i-1, ...], cur = input[..., i, ...]
    SmallVector<Value> prevCoords = coords;
    Value oneIdx = arith::ConstantIndexOp::create(rewriter, loc, 1);
    prevCoords[dim] = arith::SubIOp::create(rewriter, loc, coords[dim], oneIdx);
    Value prev = tensor::ExtractOp::create(rewriter, loc, curOut, prevCoords);
    Value cur = tensor::ExtractOp::create(rewriter, loc, input, coords);
    Value sum = arith::AddFOp::create(rewriter, loc, prev, cur);
    Value newOutVal = tensor::InsertOp::create(rewriter, loc, outType, sum, curOut, coords);
    scf::YieldOp::create(rewriter, loc, newOutVal);

    rewriter.setInsertionPointAfter(innerLoop);
    scf::YieldOp::create(rewriter, loc, innerLoop.getResult(0));

    rewriter.setInsertionPointAfter(dimLoop);
    rewriter.replaceOp(op, dimLoop.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Embedding → scf.for gather
//===----------------------------------------------------------------------===//

struct SfEmbeddingOpLowering : public OpRewritePattern<sf::EmbeddingOp> {
  using OpRewritePattern<sf::EmbeddingOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::EmbeddingOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value weight = op.getWeight();
    Value indices = op.getIndices();
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
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, indices, i));
    // If embed dim is dynamic in rt, add its size at runtime
    if (correctRank > idxRank && correctType.isDynamicDim(idxRank))
      dynSizes.push_back(arith::ConstantIndexOp::create(rewriter, loc, wType.getDimSize(1)));
    Value empty = tensor::EmptyOp::create(rewriter, loc, correctType, dynSizes);

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
            embedIdx = arith::IndexCastOp::create(b, bodyLoc, b.getIndexType(), rawIdx);
          } else if (isa<FloatType>(rawIdx.getType())) {
            Value i64Idx = arith::FPToUIOp::create(b, bodyLoc, b.getI64Type(), rawIdx);
            embedIdx = arith::IndexCastOp::create(b, bodyLoc, b.getIndexType(), i64Idx);
          } else {
            embedIdx = arith::ConstantIndexOp::create(b, bodyLoc, 0);
          }
          // Extract embedding row: weight[embedIdx, embed_dim]
          Value embedDim = linalg::IndexOp::create(b, bodyLoc, embedRank - 1);
          Value wVal = tensor::ExtractOp::create(b, bodyLoc, weight,
                                                     ValueRange{embedIdx, embedDim});
          linalg::YieldOp::create(b, bodyLoc, wVal);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Index → scf.for gather (multi-index)
//===----------------------------------------------------------------------===//

struct SfIndexOpLowering : public OpRewritePattern<sf::IndexOp> {
  using OpRewritePattern<sf::IndexOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::IndexOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // First operand is data, rest are indices
    ValueRange operands = op->getOperands();
    if (operands.size() < 2) return failure();
    Value data = op.getInput();
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

    // Pre-compute output dimension sizes outside the loop.
    // For dynamic dims, extract from the data tensor with broadcasting offset:
    //   output-dim-i = data-dim-(i - rank_offset) where rank_offset = outRank - dataRank.
    SmallVector<Value> outDims;
    int64_t dataRank = dataType.getRank();
    int64_t dynOffset = outType.getRank() - dataRank;
    for (int64_t i = 0; i < outType.getRank(); ++i) {
      if (outType.isDynamicDim(i)) {
        int64_t dataIdx = i - dynOffset;
        if (dataIdx >= 0 && dataIdx < dataRank && dataType.isDynamicDim(dataIdx))
          outDims.push_back(tensor::DimOp::create(rewriter, loc, data, dataIdx));
        else
          outDims.push_back(arith::ConstantIndexOp::create(rewriter, loc, 1));
      } else {
        outDims.push_back(arith::ConstantIndexOp::create(rewriter, loc, outType.getDimSize(i)));
      }
    }

    // Create empty tensor with correct dynamic sizes.
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < outType.getRank(); ++i)
      if (outType.isDynamicDim(i)) dynSizes.push_back(outDims[i]);
    Value empty = tensor::EmptyOp::create(rewriter, loc, outType, dynSizes);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value total;
    if (hasDynamic) {
      total = arith::ConstantIndexOp::create(rewriter, loc, 1);
      for (int64_t i = 0; i < outType.getRank(); ++i)
        total = (i == 0) ? outDims[i]
                         : arith::MulIOp::create(rewriter, loc, total, outDims[i]);
    } else {
      total = arith::ConstantIndexOp::create(rewriter, loc, outNumel);
    }

    auto forOp = scf::ForOp::create(rewriter, loc, c0, total, c1, ValueRange{empty});
    Value iv = forOp.getInductionVar();

    rewriter.setInsertionPointToStart(forOp.getBody());
    // Region iter arg (not init value) for bufferization correctness
    Value curOut = forOp.getBody()->getArgument(1);

    // Convert linear index to multi-dimensional output coordinates using
    // pre-computed dim sizes (no tensor.dim inside the loop body).
    SmallVector<Value> outCoords(outType.getRank());
    Value remaining = iv;
    for (int64_t j = 0; j < outType.getRank(); ++j) {
      outCoords[j] = arith::RemSIOp::create(rewriter, loc, remaining, outDims[j]);
      remaining = arith::DivSIOp::create(rewriter, loc, remaining, outDims[j]);
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
          idxCoords.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
      }
      Value rawIdx = tensor::ExtractOp::create(rewriter, loc, indexTensors[i], idxCoords);
      if (isa<IntegerType>(rawIdx.getType()))
        dataCoords[i] = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), rawIdx);
      else
        dataCoords[i] = arith::ConstantIndexOp::create(rewriter, loc, 0);
    }

    Value val;
    if (dataType.getRank() == 0) {
      val = tensor::ExtractOp::create(rewriter, loc, data, ValueRange{});
    } else {
      val = tensor::ExtractOp::create(rewriter, loc, data, dataCoords);
    }
    Value newOut = tensor::InsertOp::create(rewriter, loc, outType, val, curOut, outCoords);
    scf::YieldOp::create(rewriter, loc, newOut);

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

      // Collect weight name for downstream verification
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name")) {
        auto existing =
            parentFunc->getAttrOfType<ArrayAttr>("sf.weight_names");
        SmallVector<Attribute> names;
        if (existing)
          names.assign(existing.begin(), existing.end());
        names.push_back(nameAttr);
        parentFunc->setAttr("sf.weight_names",
                            ArrayAttr::get(&getContext(), names));
      }

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

    // Collect functions that contain sf ops
    SmallVector<func::FuncOp> targetFuncs;
    getOperation()->walk([&](func::FuncOp funcOp) {
      bool hasSf = false;
      funcOp.walk([&](Operation *op) {
        if (op->getDialect() && isa<sf::SfDialect>(op->getDialect())) {
          hasSf = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hasSf)
        targetFuncs.push_back(funcOp);
    });

    // Lower per-function using greedy pattern rewriter (avoids
    // dialect-conversion framework worklist divergence at scale).
    for (auto func : targetFuncs) {
      RewritePatternSet patterns(&getContext());
      patterns.add<SfBinaryLowering<sf::AddOp, arith::AddFOp>,
                   SfBinaryLowering<sf::MulOp, arith::MulFOp>,
                   SfBinaryLowering<sf::SubOp, arith::SubFOp>,
                   SfBinaryLowering<sf::DivOp, arith::DivFOp>,
                   SfBinaryLowering<sf::MaxOp, arith::MaxNumFOp>,
                   ReluLowering, IdentityLowering,
                   SfActivationOpLowering<sf::GeluOp>,
                   SfActivationOpLowering<sf::SiluOp>,
                   SfActivationOpLowering<sf::SigmoidOp>,
                   SfActivationOpLowering<sf::ExpOp>,
                   SfActivationOpLowering<sf::NegOp>,
                   SfActivationOpLowering<sf::TanhOp>,
                   SfMatmulOpLowering, SfLinearOpLowering,
                   SfViewOpLowering, SfExpandOpLowering,
                   SfUnsqueezeOpLowering, SfSumOpLowering,
                   SfTransposeOpLowering, SfSliceOpLowering,
                   SfLeOpLowering, SfLogicalAndOpLowering,
                   SfOnesLikeOpLowering, SfNewOnesOpLowering,
                   SfLayerNormOpLowering, SfRmsNormOpLowering,
                   SfScaledDotProductAttentionOpLowering,
                   SfEmbeddingOpLowering, SfSymSizeOpLowering,
                   SfArangeOpLowering, SfCumsumOpLowering,
                   SfIndexOpLowering>(&getContext());

      int64_t sfBefore = 0;
      func.walk([&](Operation *op) {
        if (op->getDialect() && isa<sf::SfDialect>(op->getDialect())) ++sfBefore;
      });
      int64_t bodyOps = 0;
      func.walk([&](Operation *) { ++bodyOps; });
      llvm::errs() << "  [sf-lower-to-linalg] lowering func '" << func.getName()
                   << "' (" << bodyOps << " body ops, " << sfBefore << " sf ops)\n";
      fprintf(stderr, "  [VERIFY] pre-walk about to start\n");
      // Pre-walk: fix result types for ops whose dyn_shape operands imply a
      // higher rank than the declared result type (e.g. sf.ones_like with
      // 2 dyn_shape operands but tensor<f32> result). Without this fix, the
      // greedy rewriter's LIFO processing would lower type-changing ops after
      // their users, creating type mismatches in the lowered IR.
      llvm::errs() << "  [sf-lower-to-linalg] pre-walk type fixing...\n";
      func.walk([&](Operation *op) {
        if (auto onesLike = dyn_cast<sf::OnesLikeOp>(op)) {
          // Total dimensions = input (scalar tensor containing dim 0's size)
          // + dyn_shape operands (additional dim sizes).
          size_t numDims = 1 + onesLike.getDynShape().size();
          auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          if (resultTy && resultTy.getRank() < (int64_t)numDims) {
            auto newTy = RankedTensorType::get(
                SmallVector<int64_t>(numDims, ShapedType::kDynamic),
                resultTy.getElementType());
            op->getResult(0).setType(newTy);
            llvm::errs() << "  [sf-lower-to-linalg] fix ones_like type: "
                         << resultTy << " -> " << newTy << "\n";
          }
        }
        if (auto arangeOp = dyn_cast<sf::ArangeOp>(op)) {
          size_t numDims = 1 + arangeOp.getDynShape().size();
          auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          if (resultTy && resultTy.getRank() < (int64_t)numDims) {
            auto newTy = RankedTensorType::get(
                SmallVector<int64_t>(numDims, ShapedType::kDynamic),
                resultTy.getElementType());
            op->getResult(0).setType(newTy);
            llvm::errs() << "  [sf-lower-to-linalg] fix arange type: "
                         << resultTy << " -> " << newTy << "\n";
          }
        }
      });
      LogicalResult result = applyPatternsGreedily(func, std::move(patterns));
      if (failed(result)) {
        llvm::errs() << "  [sf-lower-to-linalg] greedy rewriter did not converge for '"
                     << func.getName() << "'\n";
        signalPassFailure();
      }
      llvm::errs() << "  [sf-lower-to-linalg] after lowering func '" << func.getName() << "'\n";
    }

    // Post-conversion check: report remaining sf ops with their names
    int64_t remaining = 0;
    getOperation()->walk([&](Operation *op) {
      if (op->getDialect() && isa<sf::SfDialect>(op->getDialect())) {
        if (remaining == 0)
          llvm::errs() << "  [sf-lower-to-linalg] remaining sf ops:\n";
        llvm::errs() << "    " << op->getName().getStringRef() << "\n";
        ++remaining;
      }
    });
    if (remaining > 0) {
      llvm::errs() << "  [sf-lower-to-linalg] " << remaining
                   << " sf ops remain unconverted\n";
      signalPassFailure();
    } else {
      llvm::errs() << "  [sf-lower-to-linalg] all sf ops converted\n";
    }
    llvm::errs() << "  [sf-lower-to-linalg] done\n";
  }
};
} // namespace

std::unique_ptr<Pass> mlir::sf::createSfLowerToLinalg() {
  return std::make_unique<SfLowerToLinalgPass>();
}
