#ifndef SF_C_DIALECTS_H
#define SF_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Sf, sf);

#ifdef __cplusplus
}
#endif

#endif // SF_C_DIALECTS_H
