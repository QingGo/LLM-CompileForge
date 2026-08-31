#pragma once
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace mlir::sf {

/// Sentinel value in shapeAttr that represents a StringAttr/SSA reference
/// entry (as opposed to an IntegerAttr).  SSA refs consume dyn_shape operands
/// in order, identical to the IntegerAttr sentinel path but without a
/// specific operand index baked into an integer.
static constexpr int64_t kSSARefSentinel = std::numeric_limits<int64_t>::min();

enum class DimSourceKind { Static, DynOperand, Inferred };

struct DimSource {
  DimSourceKind kind;
  std::optional<int64_t> staticVal;   // only for Static
  std::optional<int64_t> operandIdx;  // only for DynOperand
  // Inferred has no extra data
};

/// Pure function — zero MLIR dependencies, directly unit-testable.
/// Resolves which output dimension comes from which source given
/// the shape attribute (e.g. [-2, -3, -1]) and number of dyn_shape operands.
///
/// Returns one DimSource per output dimension. The -1 "infer from total
/// elements" sentinel is only used when all dyn_shape operands have been
/// consumed by preceding sentinels (-2, -3, ...).
///
/// StringAttr entries must be encoded as kSSARefSentinel in the shapeAttr
/// vector.
std::vector<DimSource> resolveViewShape(
    const std::vector<int64_t>& shapeAttr, int64_t numDynOperands);

} // namespace mlir::sf
