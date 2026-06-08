#include "Dialect/DB/IR/DBDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

#include "Dialect/DB/IR/DBDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/DB/IR/DBDialectAttrs.cpp.inc"

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

    addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/DB/IR/DBDialectAttrs.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "Dialect/DB/IR/DBOps.cpp.inc"
        >();
}

llvm::LogicalResult TableType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<::mlir::db::ColumnAttr> columns,
    int64_t ncols, int64_t nrows) {
    // Verify the size ncols == colAttrArrSize
    auto colAttrArrSize = columns.size();

    if ((size_t) ncols != colAttrArrSize) {
        return emitError() << "ncols parameter is not equal to number of Column"
                              "Attributes in Table: Number of Attributes = "
                           << colAttrArrSize;
    }

    // Verify ColumnAttr nrows == Table nrows
    for (auto column : columns) {
        auto colNrows = column.getNrows();
        if (colNrows != nrows) {
            return emitError() << "nrows in column attribute " 
            << column.getName() << " is not equal to nrows in whole table";
        }
    }

    return llvm::success();
}

} // namespace db
} // namespace mlir
