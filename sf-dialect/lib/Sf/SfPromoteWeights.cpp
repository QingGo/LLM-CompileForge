#include "Sf/SfPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"

#define GEN_PASS_DEF_SFPROMOTEWEIGHTS
#include "Sf/SfPasses.h.inc"

#define NDEBUG
#include "SfLoweringHelpers.h"
#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"

#define DEBUG_TYPE "sf-lower-to-linalg"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass 1: SfPromoteWeightsPass — weight/constant → func arguments
//===----------------------------------------------------------------------===//
// Must run before sf-lower-to-linalg. Uses collect-then-modify pattern
// (not walk-and-erase) to avoid potential iterator invalidation from
// modifying parent FuncOp types during a walk of its child ops.
//===----------------------------------------------------------------------===//

namespace {
struct SfPromoteWeightsPass
    : public ::impl::SfPromoteWeightsBase<SfPromoteWeightsPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfPromoteWeightsPass)

  void runOnOperation() override {
    llvm::errs() << "  [sf-promote-weights] collecting weight ops\n";

    // Phase 1a: collect all weight ops first
    SmallVector<sf::WeightOp> weightOps;
    this->getOperation()->walk([&](sf::WeightOp op) {
      if (op->template getParentOfType<func::FuncOp>())
        weightOps.push_back(op);
    });

    llvm::errs() << "  [sf-promote-weights] found " << weightOps.size()
                 << " weight ops\n";

    // Phase 1b: promote each weight op, preserving sf.weight_names order
    // when the attribute was already set by Python.
    // Group ops by parent function.
    llvm::MapVector<func::FuncOp, SmallVector<sf::WeightOp>> funcToOps;
    for (auto op : weightOps) {
      auto parentFunc = op->template getParentOfType<func::FuncOp>();
      if (parentFunc)
        funcToOps[parentFunc].push_back(op);
    }
    unsigned promoted = 0;
    for (auto &[parentFunc, ops] : funcToOps) {
      auto wnames = parentFunc->getAttrOfType<ArrayAttr>("sf.weight_names");
      if (wnames) {
        // Reorder ops to match weight_names (consumed first, exported in
        // return order).  Preserves Python-set ordering for callers.
        llvm::StringMap<sf::WeightOp> nameToOp;
        for (auto op : ops)
          if (auto na = op->getAttrOfType<StringAttr>("name"))
            nameToOp[na.getValue()] = op;
        SmallVector<sf::WeightOp> orderedOps;
        for (auto attr : wnames) {
          auto name = mlir::cast<StringAttr>(attr).getValue();
          auto it = nameToOp.find(name);
          if (it != nameToOp.end()) {
            orderedOps.push_back(it->second);
            nameToOp.erase(it);
          }
        }
        for (auto &[_, op] : nameToOp)
          orderedOps.push_back(op);
        ops = std::move(orderedOps);
      }
      for (auto op : ops) {
        auto resultType = op.getResult().getType();
        if (!isa<RankedTensorType>(resultType)) continue;

        auto loc = op.getLoc();
        Block &entry = parentFunc.getBody().front();
        Value newArg = entry.insertArgument(
            entry.getNumArguments(), resultType, loc);

        if (auto nameAttr = op->getAttrOfType<StringAttr>("name")) {
          auto existing =
              parentFunc->getAttrOfType<ArrayAttr>("sf.weight_names");
          SmallVector<Attribute> names;
          if (existing)
            names.assign(existing.begin(), existing.end());
          names.push_back(nameAttr);
          parentFunc->setAttr("sf.weight_names",
                              ArrayAttr::get(&this->getContext(), names));
        }

        auto origType = parentFunc.getFunctionType();
        SmallVector<Type> newInputs(origType.getInputs());
        newInputs.push_back(resultType);
        parentFunc.setType(
            FunctionType::get(&this->getContext(), newInputs, origType.getResults()));

        op.replaceAllUsesWith(newArg);
        op.erase();
        ++promoted;
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "[sf-promote-weights] Promoted "
                            << promoted << " weights\n");

    // Phase 2a: collect all constant ops
    SmallVector<sf::ConstantOp> constOps;
    this->getOperation()->walk([&](sf::ConstantOp op) {
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
          FunctionType::get(&this->getContext(), newInputs, origType.getResults()));

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
    this->getOperation()->walk([&](Operation *op) {
      if (isa<sf::WeightOp>(op) || isa<sf::ConstantOp>(op)) {
        op->emitError("weight/constant not promoted");
        hasRemaining = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasRemaining) {
      this->signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::sf::createSfPromoteWeights() {
  return std::make_unique<SfPromoteWeightsPass>();
}
