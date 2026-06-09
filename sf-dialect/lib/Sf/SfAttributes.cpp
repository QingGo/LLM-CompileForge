#include "Sf/SfDialect.h"
#include "Sf/SfOps.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sf;

//===----------------------------------------------------------------------===//
// ChainOrderAttr
//===----------------------------------------------------------------------===//

Attribute ChainOrderAttr::parse(AsmParser &parser, Type type) {
  ArrayAttr value;
  if (parser.parseAttribute(value))
    return {};
  return ChainOrderAttr::get(parser.getContext(), value);
}

void ChainOrderAttr::print(AsmPrinter &printer) const {
  printer.printStrippedAttrOrType(getValue());
}

//===----------------------------------------------------------------------===//
// WeightNamesAttr
//===----------------------------------------------------------------------===//

Attribute WeightNamesAttr::parse(AsmParser &parser, Type type) {
  ArrayAttr value;
  if (parser.parseAttribute(value))
    return {};
  return WeightNamesAttr::get(parser.getContext(), value);
}

void WeightNamesAttr::print(AsmPrinter &printer) const {
  printer.printStrippedAttrOrType(getValue());
}

//===----------------------------------------------------------------------===//
// ConsumedInternallyAttr
//===----------------------------------------------------------------------===//

Attribute ConsumedInternallyAttr::parse(AsmParser &parser, Type type) {
  ArrayAttr value;
  if (parser.parseAttribute(value))
    return {};
  return ConsumedInternallyAttr::get(parser.getContext(), value);
}

void ConsumedInternallyAttr::print(AsmPrinter &printer) const {
  printer.printStrippedAttrOrType(getValue());
}

//===----------------------------------------------------------------------===//
// ExecPlanDataAttr
//===----------------------------------------------------------------------===//

Attribute ExecPlanDataAttr::parse(AsmParser &parser, Type type) {
  ArrayAttr value;
  if (parser.parseAttribute(value))
    return {};
  return ExecPlanDataAttr::get(parser.getContext(), value);
}

void ExecPlanDataAttr::print(AsmPrinter &printer) const {
  printer.printStrippedAttrOrType(getValue());
}
