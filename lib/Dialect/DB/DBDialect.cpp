#include "Dialect/DB/IR/DBDialect.h"
#include "Dialect/DB/IR/DBOps.h"
#include "Dialect/DB/IR/DBTypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include <string>

#include "Dialect/DB/IR/DBDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/DB/IR/DBDialectType.cpp.inc"
#define GET_OP_CLASSES
#include "Dialect/DB/IR/DBOps.cpp.inc"

namespace mlir {
namespace db {

void DBDialect::initialize() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/DB/IR/DBDialectType.cpp.inc"
        >();

    addOperations<
        #define GET_OP_LIST
    #include "Dialect/DB/IR/DBOps.cpp.inc"
        >();
}

} // namespace db
} // namespace mlir
