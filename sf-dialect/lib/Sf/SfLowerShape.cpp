#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace mlir::sf {

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
      // Dynamic dim — consult the shape attribute.
      // IntegerAttr(-1) means "infer from product of other dims" (only for the
      // LAST unresolved dim).  If dyn_shape operands are still available, they
      // take priority over the -1 sentinel — the -1 was placed there by the
      // compiler to mark a dynamic dimension that comes from an operand, not
      // from inference.
      if (shapeAttr && i < (int64_t)shapeAttr.size()) {
        Attribute elem = shapeAttr[i];
        if (auto intAttr = dyn_cast<IntegerAttr>(elem)) {
          int64_t val = intAttr.getInt();
          if (val == -1 && dynIdx < (int64_t)dynShapeOperands.size()) {
            // -1 with remaining operands → dynamic dim from operand
            Value dynVal = dynShapeOperands[dynIdx++];
            auto dynTy = dyn_cast<RankedTensorType>(dynVal.getType());
            if (dynTy && dynTy.getRank() == 0) {
              Value extracted = tensor::ExtractOp::create(rewriter, loc,
                  dynTy.getElementType(), dynVal, ValueRange{});
              Value asInt = arith::FPToUIOp::create(rewriter, loc, rewriter.getIntegerType(64), extracted);
              dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), asInt);
            } else if (dynTy && dynTy.getRank() == 1 && dynTy.getDimSize(0) == 1) {
              Value extracted = tensor::ExtractOp::create(rewriter, loc,
                  dynTy.getElementType(), dynVal,
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
          } else if (val < -1 && (size_t)(-val - 2) < dynShapeOperands.size()) {
            // Negative sentinel (-2, -3, ...) → use dyn_shape operand at position
            // (-val-2).  The Python compiler emits -(dyn_pos+2) for each dynamic dim
            // to distinguish it from -1 (inferred) and static dims.
            Value dynVal = dynShapeOperands[-val - 2];
            auto dynTy = dyn_cast<RankedTensorType>(dynVal.getType());
            if (dynTy && dynTy.getRank() == 0) {
              Value extracted = tensor::ExtractOp::create(rewriter, loc,
                  dynTy.getElementType(), dynVal, ValueRange{});
              Value asInt = arith::FPToUIOp::create(rewriter, loc, rewriter.getIntegerType(64), extracted);
              dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), asInt);
            } else if (dynTy && dynTy.getRank() == 1 && dynTy.getDimSize(0) == 1) {
              Value extracted = tensor::ExtractOp::create(rewriter, loc,
                  dynTy.getElementType(), dynVal,
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
          } else if (val == -1) {
            // -1 with no remaining operands → truly inferred (pass 2)
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
              Value extracted = tensor::ExtractOp::create(rewriter, loc,
                  dynTy.getElementType(), dynVal, ValueRange{});
              Value asInt = arith::FPToUIOp::create(rewriter, loc, rewriter.getIntegerType(64), extracted);
              dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), asInt);
            } else if (dynTy && dynTy.getRank() == 1 && dynTy.getDimSize(0) == 1) {
              Value extracted = tensor::ExtractOp::create(rewriter, loc,
                  dynTy.getElementType(), dynVal,
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
            Value extracted = tensor::ExtractOp::create(rewriter, loc,
                dynTy.getElementType(), dynVal, ValueRange{});
            Value asInt = arith::FPToUIOp::create(rewriter, loc, rewriter.getIntegerType(64), extracted);
            dynVal = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), asInt);
          } else if (dynTy && dynTy.getRank() == 1 && dynTy.getDimSize(0) == 1) {
            Value extracted = tensor::ExtractOp::create(rewriter, loc,
                dynTy.getElementType(), dynVal,
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
    // When both input and output dims are kDynamic, we cannot tell at compile
    // time whether broadcast is needed (e.g. expand [1,?,?,?]→[N,?,?,?] where
    // N is passed via dyn_shape).  In that case, check if the output has a
    // dyn_shape operand that makes it larger — if so, broadcast.
    SmallVector<AffineExpr> inExprs;
    for (int64_t i = 0; i < outRank; ++i) {
      int64_t inIdx = i - (outRank - inRank);
      if (inIdx < 0) continue;
      int64_t outDimSize = outType.getDimSize(i);
      int64_t inDimSize = inType.getDimSize(inIdx);
      bool needsBroadcast = false;
      if ((inDimSize == 1 || (inDimSize == ShapedType::kDynamic &&
           outDimSize != ShapedType::kDynamic && outDimSize > 1)) &&
          (outDimSize == ShapedType::kDynamic || outDimSize > 1)) {
        needsBroadcast = true;
      }
      if (needsBroadcast)
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
    // Build loc prefixes for readable lowered IR
    auto dimLoc = [&](int64_t srcDim) -> Location {
      return NameLoc::get(StringAttr::get(rewriter.getContext(),
          ("dim" + std::to_string(srcDim)).c_str()), loc);
    };
    auto namedLoc = [&](const char *suffix) -> Location {
      return NameLoc::get(StringAttr::get(rewriter.getContext(), suffix), loc);
    };

    SmallVector<Value> shapeVals;
    for (int64_t i = 0; i < outType.getRank(); ++i) {
      if (outType.isDynamicDim(i)) {
        if (i < unsqueezeDim)
          shapeVals.push_back(tensor::DimOp::create(rewriter, dimLoc(i), input, i));
        else if (i == unsqueezeDim)
          shapeVals.push_back(arith::ConstantIndexOp::create(rewriter, namedLoc("const_1"), 1));
        else
          shapeVals.push_back(tensor::DimOp::create(rewriter, dimLoc(i - 1), input, i - 1));
      } else {
        shapeVals.push_back(arith::ConstantIndexOp::create(rewriter, namedLoc(("c" + std::to_string(outType.getDimSize(i))).c_str()), outType.getDimSize(i)));
      }
    }
    auto shapeTensorType = RankedTensorType::get({(int64_t)shapeVals.size()},
                                                  rewriter.getIndexType());
    auto shapeTensor = tensor::FromElementsOp::create(rewriter, namedLoc("shape"), shapeTensorType, shapeVals);
    auto reshaped = tensor::ReshapeOp::create(rewriter, namedLoc("reshape"), outType, input, shapeTensor);
    rewriter.replaceOp(op, reshaped.getResult());
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

} // namespace

namespace mlir::sf {
void registerShapePatterns(RewritePatternSet &patterns) {
  patterns.add<SfViewOpLowering, SfExpandOpLowering,
               SfUnsqueezeOpLowering, SfTransposeOpLowering,
               SfSliceOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
