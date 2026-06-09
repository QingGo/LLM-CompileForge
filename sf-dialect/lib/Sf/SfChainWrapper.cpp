#include "Sf/SfPasses.h"
#include "Sf/SfOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringMap.h"

#define GEN_PASS_DEF_SFCHAINWRAPPER
#include "Sf/SfPasses.h.inc"

using namespace mlir;

namespace {

struct ParsedEdge {
    unsigned source;      // 0=GLOBAL_INPUT, 1=STEP_OUTPUT
    unsigned sourceIdx;   // global_inputs index or producer step index
    unsigned outputIdx;   // output index of the producer (0 for GLOBAL_INPUT)
};

struct ParsedStep {
    StringRef funcName;
    SmallVector<ParsedEdge> edges;
};

static bool parseExecPlan(ArrayAttr planAttr, StringRef chainOrder[],
                           unsigned numChain, SmallVectorImpl<ParsedStep> &steps) {
    SmallVector<int64_t> data;
    for (auto attr : planAttr) {
        auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
        if (!intAttr)
            return false;
        data.push_back(intAttr.getInt());
    }
    if (data.size() < 2)
        return false;

    unsigned idx = 0;
    unsigned numSteps = data[idx++];
    unsigned numGlobal = data[idx++];  // unused here, validated by producer
    (void)numGlobal;

    if (numSteps != numChain)
        return false;

    for (unsigned s = 0; s < numSteps; ++s) {
        if (idx >= data.size())
            return false;
        ParsedStep step;
        step.funcName = chainOrder[s];
        unsigned numInputs = data[idx++];
        for (unsigned i = 0; i < numInputs; ++i) {
            if (idx + 2 >= data.size())
                return false;
            ParsedEdge edge;
            edge.source = data[idx++];
            edge.sourceIdx = data[idx++];
            edge.outputIdx = data[idx++];
            if (edge.source > 1)
                return false;
            step.edges.push_back(edge);
        }
        steps.push_back(step);
    }
    return idx == data.size();
}

struct SfChainWrapperPass
    : public ::impl::SfChainWrapperBase<SfChainWrapperPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SfChainWrapperPass)

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();

    auto orderAttr = module->getAttr("sf.chain_order");
    if (!orderAttr)
      return;

    ArrayAttr orderArr = mlir::dyn_cast<ArrayAttr>(orderAttr);
    if (!orderArr) {
      auto ca = mlir::dyn_cast<sf::ChainOrderAttr>(orderAttr);
      if (ca) orderArr = ca.getValue();
      else return;
    }

    SmallVector<StringRef> chainOrder;
    for (auto attr : orderArr)
      chainOrder.push_back(mlir::cast<StringAttr>(attr).getValue());
    if (chainOrder.empty())
      return;

    llvm::StringMap<func::FuncOp> nameToFunc;
    func::FuncOp main0;
    for (auto op : module.getOps<func::FuncOp>()) {
      nameToFunc[op.getSymName()] = op;
      if (op.getSymName() == chainOrder[0])
        main0 = op;
    }
    if (!main0)
      return;

    if (chainOrder.size() == 1) {
      auto loc = module.getLoc();
      auto mainType = main0.getFunctionType();
      OpBuilder builder(ctx);
      builder.setInsertionPointToEnd(module.getBody());
      auto mainFunc = builder.create<func::FuncOp>(loc, "main", mainType);
      auto *entryBlock = mainFunc.addEntryBlock();
      builder.setInsertionPointToStart(entryBlock);
      SmallVector<Value> callArgs;
      for (auto arg : entryBlock->getArguments())
        callArgs.push_back(arg);
      auto callOp = builder.create<func::CallOp>(loc, main0, callArgs);
      builder.create<func::ReturnOp>(loc, callOp.getResults());
      return;
    }

    auto planAttr = module->getAttr("sf.exec_plan_data");
    if (!planAttr)
      return;
    ArrayAttr planArr = mlir::dyn_cast<ArrayAttr>(planAttr);
    if (!planArr) {
      auto ep = mlir::dyn_cast<sf::ExecPlanDataAttr>(planAttr);
      if (ep) planArr = ep.getValue();
      else return;
    }

    SmallVector<ParsedStep> steps;
    if (!parseExecPlan(planArr, chainOrder.data(), chainOrder.size(), steps))
      return;

    SmallVector<func::FuncOp> funcs;
    for (auto &s : steps) {
      auto it = nameToFunc.find(s.funcName);
      if (it == nameToFunc.end())
        return;
      funcs.push_back(it->second);
    }

    auto loc = module.getLoc();
    auto mainType = FunctionType::get(
        ctx, main0.getFunctionType().getInputs(),
        funcs.back().getFunctionType().getResults());

    OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(module.getBody());
    auto mainFunc = builder.create<func::FuncOp>(loc, "main", mainType);
    auto *entryBlock = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    SmallVector<Value> mainArgs(entryBlock->getArguments().begin(),
                                 entryBlock->getArguments().end());

    SmallVector<Value> main0CallArgs;
    for (auto arg : entryBlock->getArguments())
      main0CallArgs.push_back(arg);
    auto main0Call = builder.create<func::CallOp>(loc, main0, main0CallArgs);

    SmallVector<SmallVector<Value>> stepResults;
    stepResults.push_back({});
    for (unsigned r = 0; r < main0Call.getNumResults(); ++r)
      stepResults.back().push_back(main0Call.getResult(r));

    for (unsigned s = 1; s < steps.size(); ++s) {
      auto func = funcs[s];
      SmallVector<Value> callArgs;
      for (auto &edge : steps[s].edges) {
        if (edge.source == 0) {
          if (edge.sourceIdx >= mainArgs.size()) {
            func.emitError("global input index ") << edge.sourceIdx
                << " out of range (main has " << mainArgs.size() << " args)";
            signalPassFailure();
            return;
          }
          callArgs.push_back(mainArgs[edge.sourceIdx]);
        } else {
          if (edge.sourceIdx >= stepResults.size() ||
              edge.outputIdx >= stepResults[edge.sourceIdx].size()) {
            func.emitError("STEP_OUTPUT reference out of range: step ")
                << edge.sourceIdx << " out " << edge.outputIdx;
            signalPassFailure();
            return;
          }
          callArgs.push_back(
              stepResults[edge.sourceIdx][edge.outputIdx]);
        }
      }
      auto callOp = builder.create<func::CallOp>(loc, func, callArgs);
      stepResults.push_back({});
      for (unsigned r = 0; r < callOp.getNumResults(); ++r)
        stepResults.back().push_back(callOp.getResult(r));
    }

    auto &lastResults = stepResults.back();
    if (!lastResults.empty())
      builder.create<func::ReturnOp>(loc, lastResults.front());
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
