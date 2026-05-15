#include "Sf-c/Dialects.h"

#include "Sf/SfDialect.h"
#include "Sf/SfPasses.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Sf, sf, mlir::sf::SfDialect)

// Force all sf pass symbols to be linked into the shared library.
// Without this, the linker's dead-strip removes them since no CAPI code
// directly references the pass constructors.
// The `_sfDialectsNanobind` extension needs these symbols at runtime.
__attribute__((used))
static void *_sf_force_pass_linkage() {
  (void)mlir::sf::createSfLowerToLinalg();
  return nullptr;
}
