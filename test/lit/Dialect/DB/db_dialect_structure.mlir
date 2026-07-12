// RUN: kero-opt %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @test_db_types(
// CHECK-SAME:    %[[ARG0:.*]]: !db.table<2, 100 : [<"age", i32, 100>, <"salary", i32, 100>]>
// CHECK:         return
func.func @test_db_types(
    %arg0: !db.table<2, 100 : [
        #db.column<"age", i32, 100>,
        #db.column<"salary", i32, 100>
    ]>
) -> () {
    return
}


// -----
// CHECK-LABEL: func.func @test_scan_op(
// CHECK-SAME:    %[[ARG0:.*]]: !db.table<1, 100 : [<"age", i32, 100>]>
// CHECK-SAME:    -> !db.table<1, 100 : [<"age", i32, 100>]>
// CHECK:         %[[USER:.*]] = db.scan %[[ARG0]] : {{.*}} -> <1, 100 : [<"age", i32, 100>]>
// CHECK:         return %[[USER]] : !db.table<1, 100 : [<"age", i32, 100>]>
func.func @test_scan_op(%arg0: !db.table<1, 100 : [#db.column<"age", i32, 100>]>) 
    -> (!db.table<1, 100 : [#db.column<"age", i32, 100>]>) {

    %user = db.scan %arg0 : !db.table<1, 100 : [#db.column<"age", i32, 100>]>
        -> !db.table<1, 100 : [#db.column<"age", i32, 100>]>

    return %user : !db.table<1, 100 : [#db.column<"age", i32, 100>]>
}


// -----
// CHECK-LABEL: func.func @test_output_op(
// CHECK-SAME:    %[[ARG0]]: !db.table<3, 100 : [<"age", i32, 100>, <"salary", i32, 100>, <"budget", i32, 100>]>
// CHECK-SAME:    -> !db.table<1, 100 : [<"age", i32, 100>]>
// CHECK:         %[[SELECT:.*]] = db.output {select = ["age"]} %[[ARG0]] : {{.*}} -> <1, 100 : [<"age", i32, 100>]>
// CHECK-NEXT:    return %[[SELECT]] : {{.*}}
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


// -----
// CHECK-LABEL: func.func @test_filter_op(
// CHECK-SAME:    %[[ARG0]]: !db.table<3, 100 : [<"age", i32, 100>, <"salary", i32, 100>, <"budget", i32, 100>]>
// CHECK-SAME:    -> !db.table<3, -1 : [<"age", i32, -1>, <"salary", i32, -1>, <"budget", i32, -1>]>
// CHECK:         %[[FILTER:.*]] = db.filter %[[ARG0]] : {{.*}} {
// CHECK-NEXT:    ^bb0(
// CHECK-SAME:      %{{.*}}: !db.column<i32>, %[[SALARY:.*]]: !db.column<i32>, %{{.*}}: !db.column<i32>):
// CHECK:           %[[C0:.*]] = arith.constant 10 : i32
// CHECK:           %[[CMPI_EQ:.*]] = db.cmpi eq, %[[SALARY]], %[[C0]] : (<i32>, i32) -> <i1>
// CHECK:           db.filter_yield %[[CMPI_EQ]] : {{.*}}
// CHECK-NEXT:      } -> (<3, -1 : [<"age", i32, -1>, <"salary", i32, -1>, <"budget", i32, -1>]>)
// CHECK-NEXT:    return %[[FILTER]] : {{.*}}
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
