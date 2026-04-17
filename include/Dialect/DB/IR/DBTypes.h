#ifndef DIALECT_DB_IR_DBTYPES_H_
#define DIALECT_DB_IR_DBTYPES_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/DB/IR/DBDialectType.h.inc"

#endif // DIALECT_DB_IR_DBTYPES_H_
