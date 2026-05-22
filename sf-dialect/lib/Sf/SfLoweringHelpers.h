#pragma once

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

inline Value makeEmpty(OpBuilder &b, Location loc, Type t, ValueRange inputs) {
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
inline Value makeZeroedEmpty(OpBuilder &b, Location loc, Type t, ValueRange inputs) {
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
inline Value makeEmptyFromShape(OpBuilder &b, Location loc,
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

inline SmallVector<AffineMap> identityMaps(unsigned rank, unsigned count, MLIRContext *ctx) {
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
inline AffineMap broadcastMap(unsigned loopRank, unsigned operandRank, MLIRContext *ctx,
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
inline RankedTensorType refineBroadcastType(RankedTensorType resultType,
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
      if (!inType) continue;
      // Align trailing dimensions: operand dim j maps to output dim (rank - inType.getRank() + j)
      int64_t inDim = i - (rank - inType.getRank());
      if (inDim < 0 || inDim >= inType.getRank()) continue;
      int64_t inSize = inType.getDimSize(inDim);
      if (inSize == ShapedType::kDynamic) { anyDynamic = true; continue; }
      bestSize = std::max(bestSize, inSize);
    }
    refinedShape[i] = anyDynamic ? ShapedType::kDynamic : (bestSize == ShapedType::kDynamic ? 1 : bestSize);
  }
  return RankedTensorType::get(refinedShape, resultType.getElementType());
}

// Safe constant creation — checks that the type is handled.
inline Value createSafeConst(OpBuilder &b, Location loc, Type eltType, double floatVal, int64_t intVal = 0) {
  if (isa<FloatType>(eltType))
    return arith::ConstantOp::create(b, loc, eltType, b.getFloatAttr(eltType, floatVal));
  if (eltType.isInteger(64))
    return arith::ConstantOp::create(b, loc, eltType, b.getIntegerAttr(eltType, intVal));
  return Value();
}

inline void populateBody(linalg::GenericOp op, PatternRewriter &rewriter,
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

inline Value makeBinaryOp(OpBuilder &builder, Location loc, Value lhs, Value rhs,
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

inline Value makeUnaryOp(OpBuilder &builder, Location loc, Value in, Type outType,
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
// Template pattern structs (shared across all lowering files)
//===----------------------------------------------------------------------===//

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
