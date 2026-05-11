#ifndef KERO_C_REGISTRATION_H_
#define KERO_C_REGISTRATION_H_

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void KeroRegisterAllDialects(MlirContext context);
MLIR_CAPI_EXPORTED void KeroRegisterAllPasses(void);

#ifdef __cplusplus
}
#endif

#endif // KERO_C_REGISTRATION_H_
