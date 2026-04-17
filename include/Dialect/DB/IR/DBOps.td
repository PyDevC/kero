#ifndef DIALECT_DB_IR_DBOPS_TD_
#define DIALECT_DB_IR_DBOPS_TD_

include "DB.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

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

def DB_ReturnOp : Op<DB_Dialect, "return", [Terminator]> {
    let arguments = (ins AnyType:$input);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def DB_FilterOp : DB_Op<"filter"> {
    let summary = "Filter based on predicate.";
    let arguments = (ins DBResult:$input);
    let results = (outs DBResult:$output);

    let regions = (region SizedRegion<1>:$region);
    let assemblyFormat = "$input $region attr-dict `:` `(` type($input) `)` `->` type($output)";
}

def DB_GetColumnOp : DB_Op<"getcol"> {
    let summary = "Get specific column from the table result";
    let arguments = (ins DBRow:$row, DBColumn:$column);
    let results = (outs AnyType:$val);

    let assemblyFormat = "$row `,` $column attr-dict `:` `(` type($row) `,` type($column) `)` `->` type($val)";
}

def DB_ProjectOp : DB_Op<"project"> {
    let summary = "Project the perdicate from the scanned tables.";
    let arguments = (ins DBResult:$filter);
    let results = (outs DBResult:$result);

    let assemblyFormat = "$filter attr-dict `:` type($filter) `->` type($result)";
}

#endif // DIALECT_DB_IR_DBOPS_TD_
