#include "Dialect/DB/IR/DBDialect.h"

#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

#include "Dialect/DB/IR/DBDialect.cpp.inc"
#include "mlir/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/DB/IR/DBDialectAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/DB/IR/DBDialectType.cpp.inc"

#define GET_OP_CLASSES
#include "Dialect/DB/IR/DBOps.cpp.inc"

#include "Dialect/DB/IR/DBDialectEnum.cpp.inc"

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

//===----------------------------------------------------------------------===//
// Type Verifiers
//===----------------------------------------------------------------------===//
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
                               << column.getName()
                               << " is not equal to nrows in whole table";
        }
    }

    return llvm::success();
}

//===----------------------------------------------------------------------===//
// Operation Verifiers
//===----------------------------------------------------------------------===//
llvm::LogicalResult OutputOp::verify() {
    auto tableType = mlir::dyn_cast_or_null<TableType>(getOperand().getType());
    if (!tableType) {
        return emitError() << "Expected a operand to be of table type";
    }

    auto selectAttrArray = getSelectAttr();
    auto columns = tableType.getColumns();

    // Verify if elements in selectAttrArray are present in tableType
    for (auto selectAttr : selectAttrArray) {
        auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(selectAttr);
        auto columnName = stringAttr.getValue();
        auto it = std::find_if(columns.begin(),
                               columns.end(),
                               [&](const mlir::db::ColumnAttr col) {
                                   return col.getName() == columnName;
                               });

        if (it == columns.end()) {
            return emitError() << "select column " << selectAttr
                               << " is not present in the table";
        }
    }

    // Verify if output contains only selected columns or not
    auto output = mlir::dyn_cast<mlir::db::TableType>(getResult().getType());
    auto outputColumns = output.getColumns();

    if (outputColumns.size() != selectAttrArray.size()) {
        return emitError() << "number of selected columns does not match number"
                              " of column attributes in output table";
    }

    for (auto outColumn : outputColumns) {
        auto outColumnName = outColumn.getName();
        auto it = std::find_if(selectAttrArray.begin(),
                               selectAttrArray.end(),
                               [&](mlir::Attribute selectAttr) {
                                   auto attr = mlir::dyn_cast<mlir::StringAttr>(selectAttr);
                                   return attr.getValue() == outColumnName;
                               });

        if (it == selectAttrArray.end()) {
            return emitError() << "output column " << outColumnName
                               << " was not specified in the selected attributes";
        }
    }

    return llvm::success();
}

llvm::LogicalResult FilterOp::verify() {
    // Verify argument types in Basic Blocks
    auto blockArgs = getRegion().getBlocks().front().getArguments();
    for (auto arg : blockArgs) {
        if (!isa<ColumnType>(arg.getType())) {
            return emitError() << "Block Argument should be of DBColumnType";
        }
    }

    // Verify Filtered output Type
    auto outputType = getFiltered().getType();
    // Only need to check the nrows of TableType since all nrows should be same
    if (outputType.getNrows() != -1) {
        return emitError() << "Nrows of filter region output should always be -1";
    }

    return llvm::success();
}

llvm::LogicalResult verifyOpInFilterRegion(Operation* op) {
    if (!op->getParentOfType<FilterOp>()) {
        return failure();
    }
    return llvm::success();
}

llvm::LogicalResult verifyBooleanColumnType(Type type) {
    auto dtype = cast<ColumnType>(type).getDtype();
    if (!dtype.isInteger(1)) {
        return failure();
    }
    return llvm::success();
}

llvm::LogicalResult verifyOpsFromFilterOp(Operation* op, Type lhs, Type rhs, Location loc) {
    if (failed(verifyOpInFilterRegion(op))) {
        return emitError(loc) << "Operation: " << op->getName()
                              << " Doesn't exists in the filter region";
    }

    if (failed(verifyBooleanColumnType(lhs))) {
        return emitError(loc) << "Operand LHS is not of boolean column type.";
    }

    if (failed(verifyBooleanColumnType(rhs))) {
        return emitError(loc) << "Operand RHS is not of boolean column type.";
    }

    return success();
}

llvm::LogicalResult verifyOpsFromFilterOp(Operation* op, Type type, Location loc) {
    if (failed(verifyOpInFilterRegion(op))) {
        return emitError(loc) << "Operation: " << op->getName()
                              << " Doesn't exists in the filter region";
    }

    if (failed(verifyBooleanColumnType(type))) {
        return emitError(loc) << "Operand RHS is not of boolean column type.";
    }

    return success();
}

llvm::LogicalResult CmpOp::verify() {
    auto cmpOp = getOperation();
    if (failed(verifyOpInFilterRegion(cmpOp))) {
        return emitError() << "Cmp Op can only be used inside filter region";
    }

    // Same dtype of column and number
    auto columnDtype = dyn_cast<ColumnType>(cmpOp->getOperand(0).getType()).getDtype();
    auto numberType = cmpOp->getOperand(1).getType();

    if (columnDtype != numberType) {
        return emitError() << "Cmp Op cannot be applied for Column of type "
                           << columnDtype << " and " << numberType
                           << " . Both must be of same type";
    }
    return success();
}

llvm::LogicalResult LogicalAndOp::verify() {
    auto andOp = getOperation();
    auto lhs = getOperand(0);
    auto rhs = getOperand(1);
    auto loc = getLoc();
    return verifyOpsFromFilterOp(andOp, lhs.getType(), rhs.getType(), loc);
}

llvm::LogicalResult LogicalOrOp::verify() {
    auto orOp = getOperation();
    auto lhs = getOperand(0);
    auto rhs = getOperand(1);
    auto loc = getLoc();
    return verifyOpsFromFilterOp(orOp, lhs.getType(), rhs.getType(), loc);
}

llvm::LogicalResult LogicalNotOp::verify() {
    auto notOp = getOperation();
    auto rhs = getOperand();
    auto loc = getLoc();
    return verifyOpsFromFilterOp(notOp, rhs.getType(), loc);
}

} // namespace db
} // namespace mlir
