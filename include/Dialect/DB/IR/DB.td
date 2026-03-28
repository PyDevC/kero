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

    let parameters = (ins StringRefParameter<"table name">:$table_name);
    let assemblyFormat = "`<` $table_name `>`";
}

def DB_Column_Type : DB_Dialect_Type<"Column", "column"> {
    let summary = "DB Column type to represent column associated to a table.";
    let description = [{
        DB Column type has three parts to it, one is table it's part of, second is name of the column, and third is data type of the column.
    }];

    let parameters = (ins StringRefParameter<"table name">:$table_name, StringRefParameter<"column name">:$column_name, TypeParameter<"::mlir::Type", "AnyType">:$dtype);
    let assemblyFormat = "`<` $table_name `,` $column_name `,` $dtype `>`";
}

def DB_Row_Type : DB_Dialect_Type<"Row", "row"> {
    let summary = "DB Row type to represent row associated to a column.";
    let description = [{
        DB Row type is like index of the current row being processed.
    }];
}

def DB_Result_Type : DB_Dialect_Type<"Result", "result"> {
    let summary = "DB Result Type to represent result of any operation.";
    let description = [{
        DB Result type is used to represent any kind of result since the result of any query should be of same type.
    }];
}

#endif // DB_Dialect
