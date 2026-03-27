#ifndef DB_Dialect
#define DB_Dialect

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"

def DB_Dialect : Dialect {
    let summary = "Database Dialect for sql query representation.";
    let description = [{
        DB dialect represents various operations of sql query such as select, where, from, agg, etc.
    }];

    let name = "db";
    let cppNamespace = "::mlir::db";
    let useDefaultTypePrinterParser = 1;
}

/// Base Class for Types
class DB_Dialect_Type<string name, string typeMnemonic> : TypeDef<DB_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def DB_Table_Type : DB_Dialect_Type<"Table", "table"> {
    let summary = "DB Table type to represent table.";
    let description = [{
        DB Table type is used with from clause for pulling data from the table.
    }];

    let parameters = (ins "::std::string":$table_name);
    let assemblyFormat = "`<` $table_name `>`";
}

def DB_Column_Type : DB_Dialect_Type<"Column", "column"> {
    let summary = "DB Table type to represent table.";
    let description = [{
        DB Table type is used with from clause for pulling data from the table.
    }];

    let parameters = (ins "::std::string":$table_name);
    let assemblyFormat = "`<` $table_name `>`";
}

#endif // DB_Dialect
