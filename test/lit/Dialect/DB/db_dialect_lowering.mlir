// RUN: kero-opt -db-to-tensor-and-linalg %s | FileCheck %s

module {
    func.func @query(
        %arg0: !db.table<3, 10000 : [
            #db.column<"age", i32, 10000>,
            #db.column<"salary", i32, 10000>,
            #db.column<"budget", i32, 10000>
        ]> ) -> ( !db.table<2, -1 : [ 
            #db.column<"age", i32, -1>,
            #db.column<"salary", i32, -1> 
        ]> ) {

        %user = db.scan %arg0 
            : !db.table<3, 10000 : [
                #db.column<"age", i32, 10000>,
                #db.column<"salary", i32, 10000>,
                #db.column<"budget", i32, 10000>
            ]>
            -> !db.table<3, 10000 : [
                #db.column<"age", i32, 10000>,
                #db.column<"salary", i32, 10000>,
                #db.column<"budget", i32, 10000>
            ]>

        %filtered = db.filter %user 
            : !db.table<3, 10000 : [
                #db.column<"age", i32, 10000>,
                #db.column<"salary", i32, 10000>,
                #db.column<"budget", i32, 10000>
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

        %selected = db.output { select = ["age", "salary"] } %filtered
            : !db.table<3, -1 : [
                #db.column<"age", i32, -1>,
                #db.column<"salary", i32, -1>,
                #db.column<"budget", i32, -1>
            ]>
            -> !db.table<2, -1 : [
                #db.column<"age", i32, -1>,
                #db.column<"salary", i32, -1>
            ]>

        return %selected : !db.table<2, -1 : [
                                #db.column<"age", i32, -1>,
                                #db.column<"salary", i32, -1>
                            ]>
    }                     
} 
