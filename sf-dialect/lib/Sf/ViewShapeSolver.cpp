#include "Sf/ViewShapeSolver.h"

namespace mlir::sf {

std::vector<DimSource> resolveViewShape(
    const std::vector<int64_t>& shapeAttr, int64_t numDynOperands) {
  std::vector<DimSource> plan(shapeAttr.size());
  int64_t dynIdx = 0;

  for (size_t i = 0; i < shapeAttr.size(); ++i) {
    int64_t val = shapeAttr[i];

    if (val == kSSARefSentinel) {
      // StringAttr / SSA reference — consume next dyn_shape operand
      if (dynIdx < numDynOperands) {
        plan[i] = DimSource{DimSourceKind::DynOperand, std::nullopt,
                             dynIdx++};
      } else {
        return {}; // error: not enough dyn operands
      }
    } else if (val >= 0) {
      plan[i] = DimSource{DimSourceKind::Static, val, std::nullopt};
    } else if (val == -1) {
      // -1 with remaining operands → DynOperand (backward compat)
      // -1 with all operands consumed → truly inferred
      if (dynIdx < numDynOperands) {
        plan[i] = DimSource{DimSourceKind::DynOperand, std::nullopt,
                             dynIdx++};
      } else {
        plan[i] = DimSource{DimSourceKind::Inferred, std::nullopt,
                             std::nullopt};
      }
    } else {
      // val < -1: negative sentinel (-2, -3, ...) → operand at (-val-2)
      plan[i] = DimSource{DimSourceKind::DynOperand, std::nullopt,
                           -val - 2};
      ++dynIdx;
    }
  }

  return plan;
}

} // namespace mlir::sf
