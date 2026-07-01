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

    func.func @test_filter_op(
        %arg0: !db.table<3, 100 : [
            #db.column<"age", i32, 100>,
            #db.column<"salary", i32, 100>,
            #db.column<"budget", i32, 100>
        ]> ) -> (
        !db.table<3, -1 : [
            #db.column<"age", i32, -1>,
            #db.column<"salary", i32, -1>,
            #db.column<"budget", i32, -1>
        ]>) {

        %filtered = db.filter %arg0 
            : !db.table<3, 100 : [
                #db.column<"age", i32, 100>,
                #db.column<"salary", i32, 100>,
                #db.column<"budget", i32, 100>
            ]> {
            ^bb0(%age: !db.column<i32>, %salary: !db.column<i32>, %budget: !db.column<i32>):
                %c0 = arith.constant 10 : i32
                %0 = db.cmpi eq, %salary, %c0 : (!db.column<i32>, i32) -> !db.column<i1>
    
                db.filter_yield %0 : !db.column<i1>

            } -> (!db.table<3, -1: [
                #db.column<"age", i32, -1>,
                #db.column<"salary", i32, -1>,
                #db.column<"budget", i32, -1>
            ]>)

        return %filtered : !db.table<3, -1: [
                #db.column<"age", i32, -1>,
                #db.column<"salary", i32, -1>,
                #db.column<"budget", i32, -1>
            ]>
    }
}
