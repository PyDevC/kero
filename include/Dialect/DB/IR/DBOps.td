//===---- DbOps.td - Arith op definitions ------------------*- tablegen -*-===//

#ifndef DIALECT_DB_IR_DBOPS_TD_
#define DIALECT_DB_IR_DBOPS_TD_

include "Dialect/DB/IR/DB.td"
include "mlir/IR/OpBase.td"


// Base class for DB dialect ops
class DB_Op<string mnemonic> : Op<DB_Dialect, mnemonic> {
}

def DB_ScanOp : DB_Op<"scan"> {
    let summary = "Scans the table for columns";
    // TODO(PyDevC): Write the detailed description for this operation
    // let description = [{
    // }];
    let arguments = (ins
        DBTable:$table,
        OptionalAttr<StrAttr>:$table_name
    );

    let results = (outs DBTable:$output);

    let assemblyFormat = "attr-dict $table `:` type($table) `->` type($output)";
}

def DB_OutputOp : DB_Op<"output"> {
    let summary = "It selects some the columns from table type to output";

    let arguments = (ins
        DBTable:$table,
        StrArrayAttr:$select
    );

    let results = (outs DBTable:$output);

    let assemblyFormat = "attr-dict $table `:` type($table) `->` type($output)";

    let hasVerifier = 1;
}

#endif // DIALECT_DB_IR_DBOPS_TD_
