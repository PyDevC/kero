#ifndef KERO_C_DIALECT_H_
#define KERO_C_DIALECT_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(DB, db);

#ifdef __cplusplus
}
#endif

#endif // KERO_C_DIALECT_H_
