#include "Dialect/DB/IR/DBDialect.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/DB/IR/DBDialect.cpp.inc"

namespace mlir {
namespace db {

void DBDialect::initialize() {

#define GET_TYPEDEF_LIST
  addTypes<
#include "Dialect/DB/IR/DBDialect.cpp.inc"
      >;
};

} // namespace db
} // namespace mlir
