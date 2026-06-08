#include "Sf/SfPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringMap.h"

#define GEN_PASS_DEF_SFCHAINWRAPPER
#include "Sf/SfPasses.h.inc"

using namespace mlir;

namespace {

static SmallVector<StringRef> dedupWeightNames(ArrayAttr wnames) {
  SmallVector<StringRef> result;
  for (auto attr : wnames) {
    StringRef name = mlir::cast<StringAttr>(attr).getValue();
    if (llvm::find(result, name) == result.end())
      result.push_back(name);
  }
  return result;
}

struct SfChainWrapperPass
    : public ::impl::SfChainWrapperBase<SfChainWrapperPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfChainWrapperPass)

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();

    auto orderAttr = module->getAttrOfType<ArrayAttr>("chain_order");
    if (!orderAttr)
      return;

    SmallVector<StringRef> chainOrder;
    for (auto attr : orderAttr)
      chainOrder.push_back(mlir::cast<StringAttr>(attr).getValue());

    // Build name → FuncOp map
    llvm::StringMap<func::FuncOp> nameToFunc;
    func::FuncOp main0;
    for (auto op : module.getOps<func::FuncOp>()) {
      nameToFunc[op.getSymName()] = op;
      if (op.getSymName() == "main_0")
        main0 = op;
    }

    if (chainOrder.size() < 2) {
      // Single-function: generate pass-through main()
      if (chainOrder.size() == 1) {
        auto it = nameToFunc.find(chainOrder[0]);
        if (it == nameToFunc.end())
          return;
        auto func = it->second;
        auto loc = module.getLoc();
        auto mainType = func.getFunctionType();
        OpBuilder builder(ctx);
        builder.setInsertionPointToEnd(module.getBody());
        auto mainFunc = builder.create<func::FuncOp>(loc, "main", mainType);
        // Do NOT set Private — public visibility is required for
        // llvm.emit_c_interface and _mlir_ciface_main export.
        auto *entryBlock = mainFunc.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);
        SmallVector<Value> callArgs;
        for (auto arg : entryBlock->getArguments())
          callArgs.push_back(arg);
        auto callOp = builder.create<func::CallOp>(loc, func, callArgs);
        builder.create<func::ReturnOp>(loc, callOp.getResults());
      }
      return;
    }

    SmallVector<func::FuncOp> funcs;
    for (auto name : chainOrder) {
      auto it = nameToFunc.find(name);
      if (it != nameToFunc.end())
        funcs.push_back(it->second);
    }
    if (funcs.size() < 2)
      return;

    auto wnamesRaw = main0->getAttrOfType<ArrayAttr>("sf.weight_names");
    if (!wnamesRaw)
      wnamesRaw = main0->getAttrOfType<ArrayAttr>("debug_weight_names");
    auto wnames = wnamesRaw ? dedupWeightNames(wnamesRaw) : SmallVector<StringRef>();

    auto loc = module.getLoc();
    auto mainType = FunctionType::get(
        ctx, main0.getFunctionType().getInputs(),
        funcs.back().getFunctionType().getResults());

    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());
    auto mainFunc = builder.create<func::FuncOp>(loc, "main", mainType);
    mainFunc.setVisibility(SymbolTable::Visibility::Private);
    auto *entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    SmallVector<Value> main0CallArgs;
    for (auto arg : entryBlock->getArguments())
      main0CallArgs.push_back(arg);
    auto main0Call = builder.create<func::CallOp>(loc, main0, main0CallArgs);

    unsigned numReturnWeights = main0.getFunctionType().getResults().size() - 1;
    unsigned numConsumed = 0;
    if (wnames.size() >= numReturnWeights)
      numConsumed = wnames.size() - numReturnWeights;

    llvm::StringMap<unsigned> nameToReturnIdx;
    for (unsigned i = 0; i < numReturnWeights; ++i)
      nameToReturnIdx[wnames[numConsumed + i]] = i + 1;

    Value prevHidden = main0Call.getResult(0);

    for (size_t i = 1; i < funcs.size(); ++i) {
      auto func = funcs[i];
      auto funcType = func.getFunctionType();
      unsigned numArgs = funcType.getNumInputs();
      SmallVector<Value> callArgs;
      callArgs.push_back(prevHidden);

      auto argNamesRaw = func->getAttrOfType<ArrayAttr>("sf.weight_names");
      if (!argNamesRaw)
        argNamesRaw = func->getAttrOfType<ArrayAttr>("debug_weight_names");
      auto argNames = argNamesRaw ? dedupWeightNames(argNamesRaw) : SmallVector<StringRef>();

      for (unsigned a = 1; a < numArgs; ++a) {
        StringRef argName = (a - 1) < argNames.size() ? argNames[a - 1] : "";
        auto it = nameToReturnIdx.find(argName);
        if (it != nameToReturnIdx.end()) {
          callArgs.push_back(main0Call.getResult(it->second));
        } else if (!argName.empty()) {
          func.emitError("wrapper: cannot map arg ") << a
              << " ('" << argName << "') — not in main_0 weight_names";
          signalPassFailure();
          return;
        } else {
          // Non-weight SSA arg (sym_size, mask, etc.): match by type
          auto expectedType = funcType.getInput(a);
          bool found = false;
          for (unsigned r = 1; r < main0Call.getNumResults(); ++r) {
            if (main0Call.getResult(r).getType() == expectedType) {
              callArgs.push_back(main0Call.getResult(r));
              found = true;
              break;
            }
          }
          if (!found) {
            func.emitError("wrapper: cannot map arg ") << a
                << " (non-weight SSA, no type match)";
            signalPassFailure();
            return;
          }
        }
      }

      auto callOp = builder.create<func::CallOp>(loc, func, callArgs);
      prevHidden = callOp.getResult(0);
    }

    builder.create<func::ReturnOp>(loc, prevHidden);
  }
};

} // namespace

namespace mlir {
namespace sf {
std::unique_ptr<Pass> createSfChainWrapper() {
  return ::createSfChainWrapper();
}
} // namespace sf
} // namespace mlir
