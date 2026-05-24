#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

namespace mlir::sf {

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
  // Add bias after matmul
  if (Value bias = op.getBias()) {
    Value emptyOut = makeEmpty(rewriter, loc, resultType, {result, bias});
    if (emptyOut) {
      auto rt = cast<RankedTensorType>(resultType);
      int64_t rank = rt.getRank();
      auto biasRt = cast<RankedTensorType>(bias.getType());
      int64_t biasRank = biasRt.getRank();
      // Bias map: trailing dims align, leading dims are implicit broadcast.
      // For rank=3, biasRank=1: map = (d0,d1,d2) -> (d2)  [not (0,0,d2)]
      SmallVector<AffineExpr> rhsExprs;
      int64_t biasOffset = rank - biasRank;
      for (int64_t i = 0; i < biasRank; ++i) {
        rhsExprs.push_back(getAffineDimExpr(biasOffset + i, rewriter.getContext()));
      }
      auto idMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
      auto biasMap = AffineMap::get(rank, 0, rhsExprs, rewriter.getContext());
      SmallVector<utils::IteratorType> iter(rank, utils::IteratorType::parallel);
      auto gen = linalg::GenericOp::create(rewriter, loc, resultType,
          ValueRange{result, bias}, emptyOut,
          {idMap, biasMap, idMap}, iter);
      populateBody(gen, rewriter, [&](OpBuilder &b, Location loc2, ValueRange args) {
        Value _v = arith::AddFOp::create(b, loc2, args[0], args[1]);
        linalg::YieldOp::create(b, loc2, ValueRange{_v});
      });
      result = gen.getResult(0);
    }
  }
  rewriter.replaceOp(op, result);
  return success();
  }
};

} // namespace

namespace mlir::sf {
void registerMatmulPatterns(RewritePatternSet &patterns) {
  patterns.add<SfMatmulOpLowering, SfLinearOpLowering>(patterns.getContext());
}
} // namespace mlir::sf
