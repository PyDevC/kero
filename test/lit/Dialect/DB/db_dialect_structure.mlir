// RUN: kero-opt %s | FileCheck %s

module {
    //CHECK: func.func @test_db_types
    func.func @test_db_types(
        %arg0: !db.table<3, 100 : [
            #db.column<"age", i32, 100>,
            #db.column<"salary", i32, 100>,
            #db.column<"budget", i32, 100>
        ]>
    ) -> () {
        return
    }

    // CHECK: func.func @test_scan_op
    func.func @test_scan_op(
        %arg0: !db.table<3, 100 : [
            #db.column<"age", i32, 100>,
            #db.column<"salary", i32, 100>,
            #db.column<"budget", i32, 100>
        ]> ) -> (
        !db.table<3, 100 : [
            #db.column<"age", i32, 100>,
            #db.column<"salary", i32, 100>,
            #db.column<"budget", i32, 100>
        ]>) {

        %user = db.scan %arg0 
            : !db.table<3, 100 : [
                #db.column<"age", i32, 100>,
                #db.column<"salary", i32, 100>,
                #db.column<"budget", i32, 100>
            ]>
            -> !db.table<3, 100 : [
                #db.column<"age", i32, 100>,
                #db.column<"salary", i32, 100>,
                #db.column<"budget", i32, 100>
            ]>

        return %user : !db.table<3, 100 : [
                #db.column<"age", i32, 100>,
                #db.column<"salary", i32, 100>,
                #db.column<"budget", i32, 100>
            ]>
    }

    func.func @test_output_op(
        %arg0: !db.table<3, 100 : [
            #db.column<"age", i32, 100>,
            #db.column<"salary", i32, 100>,
            #db.column<"budget", i32, 100>
        ]> ) -> (
        !db.table<1, 100 : [
            #db.column<"age", i32, 100>
        ]>) {

        %selected = db.output { select = ["age"] } %arg0 
            : !db.table<3, 100 : [
                #db.column<"age", i32, 100>,
                #db.column<"salary", i32, 100>,
                #db.column<"budget", i32, 100>
            ]>
            -> !db.table<1, 100 : [#db.column<"age", i32, 100>]>
        return %selected : !db.table<1, 100 : [#db.column<"age", i32, 100>]>
    }
}

module {
    // This is not possible it's present here only for testing
    func.func @test_db_types(%salary: !db.column<i32>, %const: i32) -> (!db.column<i1>) {
        %0 = db.cmpi eq, %salary, %const : (!db.column<i32>, i32) -> !db.column<i1>
        %1 = db.cmpi neq, %salary, %const : (!db.column<i32>, i32) -> !db.column<i1>

        %and = db.and %0, %1 : (!db.column<i1>, !db.column<i1>) -> !db.column<i1>
        return %and : !db.column<i1>
    }
}
