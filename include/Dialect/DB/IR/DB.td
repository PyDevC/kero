#ifndef DB_Dialect
#define DB_Dialect

include "mlir/IR/DialectBase.td"

def DB_Dialect : Dialect {
    let summary = "Database Dialect for sql query representation.";
    let description = [{
        DB dialect represents various operations of sql query such as select, where, from, agg, etc.
    }];

    let name = "db";
    let cppNamespace = "::mlir::db";
    let useDefaultTypePrinterParser = 1;
}

#endif // DB_Dialect
