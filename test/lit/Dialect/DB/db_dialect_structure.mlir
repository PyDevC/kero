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

module {
    // CHECK-LABEL: @test_db_filter_op
    func.func @test_db_filter_op (%arg0: !db.table<"user">, %arg_age: !db.column<"user", "age", i32>) -> !db.result {
        %scan0 = db.scan %arg0 : !db.table<"user"> -> !db.result
        %filtered = db.filter %scan0 {
            ^bb0(%row : !db.row):
                %age_val = db.getcol %row , %arg_age : (!db.row, !db.column<"user", "age", i32>) -> i32
                %age_limit = arith.constant 10 : i32
                %cond = arith.cmpi sgt, %age_val, %age_limit : i32
                db.return %cond : i1
        } : (!db.result) -> !db.result
        return %filtered : !db.result
    }
    
    // CHECK-LABEL: @test_db_projection_op
    func.func @test_db_projection_op (%arg0: !db.table<"user">, %arg_age: !db.column<"user", "age", i32>) ->  !db.result {
        %filtered = func.call @test_db_filter_op(%arg0, %arg_age) : (!db.table<"user">, !db.column<"user", "age", i32>) -> !db.result
        %projected = db.project %filtered : !db.result -> !db.result
        return %projected : !db.result
    }
}
