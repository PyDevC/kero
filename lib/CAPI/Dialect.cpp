#include "mlir/CAPI/Registration.h"

#include "Dialect/DB/IR/DBDialect.h"
#include "kero-c/Dialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DBDialect, db, mlir::db::DBDialect)
