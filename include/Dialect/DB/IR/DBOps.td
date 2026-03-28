#ifndef DIALECT_DB_IR_DBOPS_TD_
#define DIALECT_DB_IR_DBOPS_TD_

include "DB.td"
include "mlir/IR/OpBase.td"

/// Table Container which is like pointer to a table
class DB_Op<string mnemonic> : Op<DB_Dialect, mnemonic> {
}

class DB_TableOp<string mnemonic> : DB_Op<mnemonic> {
    let arguments = (ins DBTable:$table);
    let results = (outs DBResult:$result);
    let assemblyFormat = "$table attr-dict `:` type($table) `->` type($result)";
}

def DB_ScanOp : DB_TableOp<"scan"> {
    let summary = "Scans the table for columns";
}

#endif // DIALECT_DB_IR_DBOPS_TD_
