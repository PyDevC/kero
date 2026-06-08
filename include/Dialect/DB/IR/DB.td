//===-------- DB.td - Base defs for db dialect -------------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//
// This file contains the type dialect and type definitions for `db` dialect.
//===----------------------------------------------------------------------===//

#ifndef DB_DIALECT_TD_
#define DB_DIALECT_TD_


include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// DB Dialect definition
//===----------------------------------------------------------------------===//
def DB_Dialect : Dialect {
    let summary = "Database Dialect to model SQL types and operations.";
    let description = [{
        `db` dialect is designed to generate IR from DB AST which mimics the 
        operations from Apache Arrow and SQL, our main goal is to provide a 
        middle ground for ease of use and direct compute mapping to 'kero' 
        dialect.

        As of now we try to not have any other dialect dependency other than 
        `func` dialect.
    }];

    let name = "db";
    let cppNamespace = "::mlir::db";
    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
}

// Column Attribute for Table type
def ColumnAttr : AttrDef<DB_Dialect, "Column"> {
    let mnemonic = "column";
    let summary = "Attribute for single column in the table";
    let description = [{
        Represents one column in the Table
    }];

    let parameters = (ins 
        "::mlir::StringAttr":$name, 
        "::mlir::Type":$type, 
        "int64_t":$nrows
    );

    let assemblyFormat = "`<` $name `,` $type `,` $nrows `>`";
}

//===----------------------------------------------------------------------===//
// DB Dialect Base Types
//===----------------------------------------------------------------------===//
class DB_Dialect_Type<string name, string typeMnemonic> : 
    TypeDef<DB_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def DBTable : DB_Dialect_Type<"Table", "table"> {
    let summary = "Table type shows the collection of Columns.";
    let description = [{
        Table Represents the collection of column tied together.
    }];

    let parameters = (ins 
        ArrayRefParameter<"::mlir::db::ColumnAttr">:$columns, 
        "int64_t":$ncols, 
        "int64_t":$nrows
    );

    let assemblyFormat = "`<` $ncols `,` $nrows `:` `[` $columns `]` `>`";

    let genVerifyDecl = 1;
}

#endif // DB_DIALECT_TD_
