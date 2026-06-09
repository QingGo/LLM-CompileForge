#ifndef SF_OPS_H
#define SF_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "Sf/SfOps.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Sf/SfOpsAttributes.h.inc"

namespace mlir {
class OpBuilder;
}

namespace mlir::sf {
/// Compute the broadcast shape of two ranked tensor types following NumPy
/// broadcasting rules.  Returns failure if shapes are not broadcast-compatible.
LogicalResult computeBroadcastShape(RankedTensorType lhsType,
                                    RankedTensorType rhsType,
                                    SmallVectorImpl<int64_t> &outShape);
} // namespace mlir::sf

#endif // SF_OPS_H
