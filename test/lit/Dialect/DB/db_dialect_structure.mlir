// RUN: kero-opt %s | FileCheck %s

// CHECK-LABEL: @test_db_types
// CHECK-SAME: (%arg0: !db.table<"user">, %arg1: !db.column<"user", "id", i32>)
module {
    func.func @test_db_types (%arg0: !db.table<"user">, %arg1: !db.column<"user", "id", i32>) {
        return
    }
}

// CHECK-LABEL: @test_db_scan_op
module {
    func.func @test_db_scan_op (%arg0: !db.table<"user">) -> !db.result {
        // CHECK: db.scan
        %scan0 = db.scan %arg0 : !db.table<"user"> -> !db.result
        return %scan0 : !db.result
    }

}
