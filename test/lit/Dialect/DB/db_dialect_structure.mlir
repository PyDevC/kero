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
        !db.table<3, 100 : [
            #db.column<"age", i32, 100>,
            #db.column<"salary", i32, 100>,
            #db.column<"budget", i32, 100>
        ]> ) {

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

        %filtered = db.filter %user 
            : !db.table<3, 100 : [
                #db.column<"age", i32, 100>,
                #db.column<"salary", i32, 100>,
                #db.column<"budget", i32, 100>
            ]> -> !db.mask {
            ^bb0(%col0: tensor<1x100xi32>, %col1: tensor<1x100xi32>):
                %c0 = arith.constant 10 : i32
                %c_tensor = tensor.splat %c0 : tensor<1x100xi32>
                %cmp = arith.cmpi sle, %col0, %c_tensor : tensor<1x100xi32>
                db.return %user : !db.table<3, 100 : [
                        #db.column<"age", i32, 100>,
                        #db.column<"salary", i32, 100>,
                        #db.column<"budget", i32, 100>
                    ]>
                -> !db.table<3, 100 : [
                    #db.column<"age", i32, 100>,
                    #db.column<"salary", i32, 100>,
                    #db.column<"budget", i32, 100>
                ]>
            }
        }
}
