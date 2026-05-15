#include "Sf-c/Dialects.h"
#include "Sf/SfPasses.h"
#include "mlir/Bindings/Python/Nanobind.h"

// Register passes at module load time
using namespace mlir::sf;
#define GEN_PASS_REGISTRATION
#include "Sf/SfPasses.h.inc"

namespace nb = nanobind;

NB_MODULE(_sfDialectsNanobind, m) {
  auto sfM = m.def_submodule("sf");

  sfM.def(
      "register_dialects",
      [](nb::capsule context_capsule, bool load) {
        void *ptr = static_cast<void *>(context_capsule.data());
        MlirContext ctx;
        ctx.ptr = ptr;
        MlirDialectHandle handle = mlirGetDialectHandle__sf__();
        mlirDialectHandleRegisterDialect(handle, ctx);
        if (load) {
          mlirDialectHandleLoadDialect(handle, ctx);
        }
      },
      nb::arg("context_capsule"), nb::arg("load") = true);

  // Register passes
  registerSfPasses();
}
