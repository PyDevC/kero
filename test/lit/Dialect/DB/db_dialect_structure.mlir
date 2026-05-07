// RUN: kero-opt %s | FileCheck %s

// CHECK-LABEL: @test_db_types
// CHECK-SAME: (%arg0: !db.table<"user", 10, 100>, %arg1: !db.column<"user", "id", i32, 100>)
module {
    func.func @test_db_types (%arg0: !db.table<"user", 10, 100>, %arg1: !db.column<"user", "id", i32, 100>) {
        return
    }
}

// CHECK-LABEL: @test_db_scan_op
module {
    // CHECK-SAME: (%arg0: !db.table<"user", 10, 100>)
    func.func @test_db_scan_op (%arg0: !db.table<"user", 10, 100>) -> !db.result {
        // CHECK: db.scan
        %scan0 = db.scan %arg0 : !db.table<"user", 10, 100> -> !db.result
        return %scan0 : !db.result
    }

}

module {
    // CHECK-LABEL: @test_db_filter_op
    func.func @test_db_filter_op (%arg0: !db.table<"user", 10, 100>, %arg_age: !db.column<"user", "age", i32, 100>) -> !db.result {
        // CHECK-NEXT: %[[RESULT:.*]] = db.scan %arg0 : <"user", 10, 100>
        %scan0 = db.scan %arg0 : !db.table<"user", 10, 100> -> !db.result
        %filtered = db.filter %scan0 {
            ^bb0(%row : !db.row):
                %age_val = db.getcol %row , %arg_age : (!db.row, !db.column<"user", "age", i32, 100>) -> i32
                %age_limit = arith.constant 10 : i32
                %cond = arith.cmpi sgt, %age_val, %age_limit : i32
                db.return %cond : i1
        } : (!db.result) -> !db.result
        return %filtered : !db.result
    }
    
    // CHECK-LABEL: @test_db_projection_op
    func.func @test_db_projection_op (%arg0: !db.table<"user", 10, 100>, %arg_age: !db.column<"user", "age", i32, 100>) ->  !db.result {
        %filtered = func.call @test_db_filter_op(%arg0, %arg_age) : (!db.table<"user", 10, 100>, !db.column<"user", "age", i32, 100>) -> !db.result
        %projected = db.project %filtered : !db.result -> !db.result
        return %projected : !db.result
    }
}

// CHECK: @query
func.func @query (%arg0: !db.table<"PERSON", 5, 1000>, %arg_age: !db.column<"PERSON", "age", i32, 1000>) -> !db.result {
    %scan0 = db.scan %arg0 : !db.table<"PERSON", 5, 1000> -> !db.result
    %filter = db.filter %scan0 {
        ^bb0(%row : !db.row):
            %no_limit = arith.constant 1 : i32
            db.return %no_limit : i32
    } : (!db.result) -> !db.result
    return %filter : !db.result
}
