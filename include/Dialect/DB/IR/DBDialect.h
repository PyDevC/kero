//===- DBDialect.h - Arith dialect --------------------------------*- C++-*-==//

#ifndef DIALECT_DB_IR_DBDIALECT_H_
#define DIALECT_DB_IR_DBDIALECT_H_

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// DB Dialect
//===----------------------------------------------------------------------===//

#include "Dialect/DB/IR/DBDialect.h.inc"

#include "Dialect/DB/IR/DBDialectEnum.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/DB/IR/DBDialectAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/DB/IR/DBDialectType.h.inc"

#define GET_OP_CLASSES
#include "Dialect/DB/IR/DBOps.h.inc"

#endif // DIALECT_DB_IR_DBDIALECT_H_
