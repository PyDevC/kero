// RUN: kero-opt %s | FileCheck %s

// CHECK-LABEL: @test_db_types
module {
    func.func @test_db_types(%arg0: !db.table<"user">, %arg1: !db.column<"user", "id", i32>) {
        return
    }
}
