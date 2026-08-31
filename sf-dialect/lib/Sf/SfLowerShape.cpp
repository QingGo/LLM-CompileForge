#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"
#include "Sf/ViewShapeSolver.h"

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
    auto shapeAttr = op->getAttrOfType<ArrayAttr>("shape");
    if (!shapeAttr) return failure();
    auto dynShapeOperands = op.getDynShape();

    // Convert ArrayAttr to int64_t vector for pure decision logic.
    // StringAttr entries become kSSARefSentinel.
    std::vector<int64_t> shapeVec;
    shapeVec.reserve(shapeAttr.size());
    for (Attribute elem : shapeAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(elem))
        shapeVec.push_back(intAttr.getInt());
      else if (dyn_cast<StringAttr>(elem))
        shapeVec.push_back(kSSARefSentinel);
      else
        return failure();
    }

    // Delegate all shape resolution logic to the pure function.
    auto plan = resolveViewShape(shapeVec,
                                  (int64_t)dynShapeOperands.size());
    if (plan.empty() && !shapeVec.empty()) return failure();

    // Pass 1: collect shape values and track the -1 (inferred) dimension.
    SmallVector<Value> shapeVals;
    int64_t inferredIdx = -1;

    for (int64_t i = 0; i < outType.getRank(); ++i) {
      if (!outType.isDynamicDim(i)) {
        shapeVals.push_back(arith::ConstantIndexOp::create(rewriter, loc,
            outType.getDimSize(i)));
        continue;
      }
      if (i >= (int64_t)plan.size()) return failure();
      auto& src = plan[i];
      switch (src.kind) {
        case DimSourceKind::Static:
          shapeVals.push_back(arith::ConstantIndexOp::create(rewriter, loc,
              *src.staticVal));
          break;
        case DimSourceKind::DynOperand:
          shapeVals.push_back(extractDynDimAsIndex(rewriter, loc,
              dynShapeOperands[*src.operandIdx]));
          break;
        case DimSourceKind::Inferred:
          inferredIdx = i;
          shapeVals.push_back(nullptr);
          break;
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
          dynSizes.push_back(extractDynDimAsIndex(rewriter, loc,
              dynShapeOperands[dynIdx++]));
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
    if (d0 < 0) d0 += rank;
    if (d1 < 0) d1 += rank;
    std::swap(perm[d0], perm[d1]);

    // Build inverse permutation: for each output dim j, which input dim
    // provides its value coordinate.
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

    // Use an explicit linalg.generic instead of linalg.transpose.  The
    // named op path has been observed to crash for rank>=3 dynamic tensors;
    // a generic with an explicit inverse-permutation input map is both
    // rank-generic and avoids the bad path.
    auto ctx = rewriter.getContext();
    SmallVector<AffineExpr> inputExprs;
    inputExprs.reserve(rank);
    for (int64_t i = 0; i < rank; ++i)
      inputExprs.push_back(getAffineDimExpr(invPerm[i], ctx));
    auto inputMap = AffineMap::get(rank, 0, inputExprs, ctx);
    auto outMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto generic = linalg::GenericOp::create(
        rewriter, loc, rt, ValueRange{input}, ValueRange{empty},
        {inputMap, outMap}, iters);
    populateBody(generic, rewriter, [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
      linalg::YieldOp::create(b, bodyLoc, args[0]);
    });
    rewriter.replaceOp(op, generic.getResult(0));
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
    int64_t dim = 0, start = 0, sEnd = 0, step = 1;
    if (auto attr = op.getOperation()->getAttrOfType<IntegerAttr>("dim")) dim = attr.getInt();
    if (auto attr = op.getOperation()->getAttrOfType<IntegerAttr>("start")) start = attr.getInt();
    if (auto attr = op.getOperation()->getAttrOfType<IntegerAttr>("end")) sEnd = attr.getInt();
    if (auto attr = op.getOperation()->getAttrOfType<IntegerAttr>("step")) step = attr.getInt();
    if (step <= 0) step = 1;
    int64_t rank = inType.getRank();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) return failure();
    llvm::errs() << "  [SfSlice] input=" << inType << " dim=" << dim
                 << " start=" << start << " end=" << sEnd
                 << " step=" << step << "\n";
    static constexpr int64_t kDynSentinel = 9223372036854775807LL;
    SmallVector<OpFoldResult> offs, szs, strs;
    for (int64_t i = 0; i < rank; ++i) {
      offs.push_back(rewriter.getIndexAttr((i == dim) ? start : 0));
      if (i == dim && sEnd == kDynSentinel) {
        // INT64_MAX sentinel: size = ceil((dim(input, i) - start) / step)
        Value dimVal = tensor::DimOp::create(rewriter, loc, input, i);
        Value startVal = arith::ConstantIndexOp::create(rewriter, loc, start);
        Value lenVal = arith::SubIOp::create(rewriter, loc, dimVal, startVal);
        if (step != 1) {
          Value stepVal = arith::ConstantIndexOp::create(rewriter, loc, step);
          Value stepMinus1 = arith::ConstantIndexOp::create(rewriter, loc, step - 1);
          lenVal = arith::AddIOp::create(rewriter, loc, lenVal, stepMinus1);
          lenVal = arith::DivSIOp::create(rewriter, loc, lenVal, stepVal);
        }
        szs.push_back(lenVal);
      } else if (i == dim) {
        // Explicit static end: use the exact sliced length even if the input
        // dimension is dynamic (previously this fell into the dynamic-dim
        // branch and accidentally sliced the full dimension).
        int64_t len = (sEnd >= start) ? (sEnd - start) : 0;
        if (step != 1)
          len = (len + step - 1) / step;
        llvm::errs() << "  [SfSlice] len=" << len << "\n";
        szs.push_back(rewriter.getIndexAttr(len));
      } else if (inType.isDynamicDim(i)) {
        szs.push_back(Value(tensor::DimOp::create(rewriter, loc, input, i).getResult()));
      } else {
        szs.push_back(rewriter.getIndexAttr(inType.getDimSize(i)));
      }
      strs.push_back(rewriter.getIndexAttr(step == 1 ? 1 : step));
    }
    auto slice = tensor::ExtractSliceOp::create(rewriter, loc, input, offs, szs, strs);
    llvm::errs() << "  [SfSlice] created type=" << slice->getResult(0).getType() << "\n";
    rewriter.replaceOp(op, slice->getResult(0));
    return success();
  }
};

// Pad → tensor.pad (constant mode) or identity when the pad is all zeros.
// PyTorch's pad list is (left, right, top, bottom, ...) from the last
// dimension backwards; MLIR tensor.pad uses per-dimension low/high in
// forward order.
struct SfPadOpLowering : public OpRewritePattern<sf::PadOp> {
  using OpRewritePattern<sf::PadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::PadOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!inType || !outType) return failure();
    if (outType.getRank() != inType.getRank()) return failure();

    auto padAttr = op->getAttrOfType<ArrayAttr>("pad");
    if (!padAttr) return failure();
    SmallVector<int64_t> pad;
    pad.reserve(padAttr.size());
    for (Attribute a : padAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(a))
        pad.push_back(intAttr.getInt());
      else
        return failure();
    }

    bool allZero = true;
    for (int64_t v : pad)
      if (v != 0) { allZero = false; break; }
    if (allZero) {
      rewriter.replaceOp(op, input);
      return success();
    }

    int64_t rank = inType.getRank();
    SmallVector<OpFoldResult> lows, highs;
    lows.reserve(rank);
    highs.reserve(rank);
    for (int64_t d = 0; d < rank; ++d) {
      int64_t lo = 0, hi = 0;
      // PyTorch pad pair index for this dim (from last dim backwards).
      int64_t pairIdx = rank - 1 - d;
      if (pairIdx >= 0 && pairIdx * 2 + 1 < (int64_t)pad.size()) {
        lo = pad[pairIdx * 2];
        hi = pad[pairIdx * 2 + 1];
      }
      lows.push_back(rewriter.getIndexAttr(lo));
      highs.push_back(rewriter.getIndexAttr(hi));
    }

    auto eltType = outType.getElementType();
    Value padValue;
    auto aux = op.getAux();
    if (!aux.empty()) {
      Value pv = aux[aux.size() - 1];
      auto pvType = dyn_cast<RankedTensorType>(pv.getType());
      if (!pvType) return failure();
      SmallVector<Value> idx(pvType.getRank(),
          arith::ConstantIndexOp::create(rewriter, loc, 0));
      Value extracted = tensor::ExtractOp::create(
          rewriter, loc, pvType.getElementType(), pv, idx);
      if (extracted.getType() != eltType) {
        if (isa<FloatType>(extracted.getType()) && isa<FloatType>(eltType)) {
          auto srcF = cast<FloatType>(extracted.getType());
          auto dstF = cast<FloatType>(eltType);
          if (srcF.getWidth() < dstF.getWidth())
            extracted = arith::ExtFOp::create(rewriter, loc, eltType, extracted);
          else if (srcF.getWidth() > dstF.getWidth())
            extracted = arith::TruncFOp::create(rewriter, loc, eltType, extracted);
          else
            return failure();
        } else {
          return failure();
        }
      }
      padValue = extracted;
    } else {
      padValue = arith::ConstantOp::create(
          rewriter, loc, eltType, rewriter.getFloatAttr(eltType, 0.0f));
    }

    Value padded = rewriter.create<tensor::PadOp>(
        loc, outType, input, lows, highs, padValue);
    rewriter.replaceOp(op, padded);
    return success();
  }
};

// Select → extract a slice of size 1 along `dim`, then collapse that unit
// dimension away.  PyTorch `select.int` removes the selected dimension.
struct SfSelectOpLowering : public OpRewritePattern<sf::SelectOp> {
  using OpRewritePattern<sf::SelectOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(sf::SelectOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = op.getInput();
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
    llvm::errs() << "  [SfSelect] lowering " << inType << " -> " << outType << "\n";
    if (!inType || !outType) return failure();
    int64_t rank = inType.getRank();
    if (rank < 1 || outType.getRank() != rank - 1) return failure();

    int64_t dim = 0;
    int64_t index = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("dim")) dim = attr.getInt();
    if (auto attr = op->getAttrOfType<IntegerAttr>("index")) index = attr.getInt();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) return failure();

    SmallVector<OpFoldResult> offsets(rank, OpFoldResult(rewriter.getIndexAttr(0)));
    SmallVector<OpFoldResult> sizes(rank, OpFoldResult(rewriter.getIndexAttr(1)));
    SmallVector<OpFoldResult> strides(rank, OpFoldResult(rewriter.getIndexAttr(1)));
    offsets[dim] = OpFoldResult(rewriter.getIndexAttr(index));
    for (int64_t d = 0; d < rank; ++d) {
      if (d == dim) continue;
      if (inType.isDynamicDim(d)) {
        sizes[d] = Value(tensor::DimOp::create(rewriter, loc, input, d).getResult());
      } else {
        sizes[d] = OpFoldResult(rewriter.getIndexAttr(inType.getDimSize(d)));
      }
    }

    // Slice keeps the selected unit dim, so build the intermediate type.
    SmallVector<int64_t> sliceShape(inType.getShape().begin(), inType.getShape().end());
    sliceShape[dim] = 1;
    auto sliceTy = RankedTensorType::get(sliceShape, inType.getElementType());
    auto slice = tensor::ExtractSliceOp::create(rewriter, loc, sliceTy, input, offsets, sizes, strides);

    // Collapse the unit dimension.  The reassociation groups input dims
    // into output dims; the selected dimension merges with an adjacent
    // dimension (with a previous neighbor if one exists, else the next).
    SmallVector<ReassociationIndices> reassoc;
    if (rank == 1) {
      reassoc.push_back({0});
    } else if (dim == 0) {
      reassoc.push_back({0, 1});
      for (int64_t d = 2; d < rank; ++d)
        reassoc.push_back({d});
    } else {
      for (int64_t d = 0; d < rank; ++d) {
        if (d == dim) {
          reassoc.back().push_back(d);
        } else {
          reassoc.push_back({d});
        }
      }
    }
    auto collapsed = tensor::CollapseShapeOp::create(rewriter, loc, outType, slice, reassoc);
    rewriter.replaceOp(op, collapsed.getResult());
    return success();
  }
};

} // namespace

namespace mlir::sf {
void registerShapePatterns(RewritePatternSet &patterns) {
  patterns.add<SfViewOpLowering, SfExpandOpLowering,
               SfUnsqueezeOpLowering, SfTransposeOpLowering,
               SfSliceOpLowering, SfPadOpLowering,
               SfSelectOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
