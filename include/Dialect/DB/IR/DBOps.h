#ifndef DIALECT_DB_IR_DBOPS_H_
#define DIALECT_DB_IR_DBOPS_H_

#include "Dialect/DB/IR/DBDialect.h"
#include "Dialect/DB/IR/DBTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "Dialect/DB/IR/DBOps.h.inc"

#endif // DIALECT_DB_IR_DBOPS_H_
