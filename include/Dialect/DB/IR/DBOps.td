//===---- DbOps.td - Arith op definitions ------------------*- tablegen -*-===//

#ifndef DIALECT_DB_IR_DBOPS_TD_
#define DIALECT_DB_IR_DBOPS_TD_

include "Dialect/DB/IR/DB.td"
include "mlir/IR/OpBase.td"


//===----------------------------------------------------------------------===//
// Base class for all DB ops
//===----------------------------------------------------------------------===//
class DB_Op<string mnemonic, list<Trait> traits = []> : Op<DB_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// General Ops for DB
//===----------------------------------------------------------------------===//
// TODO(PyDevC): we practically don't need this operation but I am keeping it for future lowering applications
def DB_ScanOp : DB_Op<"scan"> {
    let summary = "Scan Op just to indicate that this is a select query operation";
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

//===----------------------------------------------------------------------===//
// Ops for DB Filter Op region
//===----------------------------------------------------------------------===//
// TODO(PyDevC): write ops for logical comparison with their verifiers
// TODO(PyDevC): These functions should only be present in filter op region
class DB_LogicalOps<string mnemonic> : DB_Op<mnemonic> {
    let arguments = (ins DBColumn:$col1, DBColumn:$col2);

    let results = (outs DBColumn:$result);

    let assemblyFormat = "attr-dict $col1 `,` $col2 `:` `(` type($col1) `,` type($col2) `)` `->` type($result)";

    let hasVerifier = 1;
}

def DB_LogicalAndOp : DB_LogicalOps<"and"> {
    let summary = "Performs logical and on two boolean columns";
}

def DB_LogicalOrOp : DB_LogicalOps<"or"> {
    let summary = "Performs logical or on two boolean columns";
}

// Logical not only takes in a column so we overwrite args and asm
def DB_LogicalNotOp : DB_LogicalOps<"not"> {
    let summary = "Performs logical not on boolean column";

    let arguments = (ins DBColumn:$col);

    let assemblyFormat = "attr-dict $col `:` `(` type($col) `)` `->` type($result)";
}

def DB_CmpIOp : DB_Op<"cmpi"> {
    let summary = "Performs comparison between two column based on given predicate";

    let arguments = (ins DB_CmpIPredicateAttr:$predicate, DBColumn:$col1, DBColumn:$col2);

    let results = (outs DBColumn:$result);

    let assemblyFormat = "attr-dict $predicate `,` $col1 `,` $col2 `:` `(` type($col1) `,` type($col2) `)` `->` type($result)";

    let hasVerifier = 1;
}

#endif // DIALECT_DB_IR_DBOPS_TD_
