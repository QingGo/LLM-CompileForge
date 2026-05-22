#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace {

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

} // namespace

namespace mlir::sf {
void registerSeqOpsPatterns(RewritePatternSet &patterns) {
  patterns.add<SfCumsumOpLowering, SfEmbeddingOpLowering,
               SfIndexOpLowering, SfSymSizeOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
