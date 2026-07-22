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


!table = !db.table<3, 100 : [#db.column<"age", i32, 100>, #db.column<"salary", i32, 100>, #db.column<"budget", f32, 100>]>
!new_table = !db.table<3, -1 : [#db.column<"age", i32, -1>, #db.column<"salary", i32, -1>, #db.column<"budget", f32, -1>]>

// -----
// CHECK-LABEL: func.func @test_output_op(
// CHECK-SAME:    %[[ARG0]]: !db.table<3, 100 : [<"age", i32, 100>, <"salary", i32, 100>, <"budget", f32, 100>]>
// CHECK-SAME:    -> !db.table<1, 100 : [<"age", i32, 100>]>
// CHECK:         %[[SELECT:.*]] = db.output {select = ["age"]} %[[ARG0]] : {{.*}} -> <1, 100 : [<"age", i32, 100>]>
// CHECK-NEXT:    return %[[SELECT]] : {{.*}}
func.func @test_output_op(%arg0: !table) -> (!db.table<1, 100 : [#db.column<"age", i32, 100>]>) {
    %selected = db.output { select = ["age"] } %arg0 : !table
        -> !db.table<1, 100 : [#db.column<"age", i32, 100>]>
    return %selected : !db.table<1, 100 : [#db.column<"age", i32, 100>]>
}


// -----
// CHECK-LABEL: func.func @test_filter_op(
// CHECK-SAME:    %[[ARG0]]: !db.table<3, 100 : [<"age", i32, 100>, <"salary", i32, 100>, <"budget", f32, 100>]>
// CHECK-SAME:    -> !db.table<3, -1 : [<"age", i32, -1>, <"salary", i32, -1>, <"budget", f32, -1>]>
// CHECK:         %[[FILTER:.*]] = db.filter %[[ARG0]] : {{.*}} {
// CHECK-NEXT:    ^bb0(
// CHECK-SAME:      %[[AGE:.*]]: !db.column<i32>, %[[SALARY:.*]]: !db.column<i32>, %[[BUDGET:.*]]: !db.column<f32>):
// CHECK:           %[[C1000:.*]] = arith.constant 1000 : i32
// CHECK:           %[[C18:.*]] = arith.constant 18 : i32
// CHECK:           %[[CMPI_LT0:.*]] = db.cmp lt, %[[SALARY]], %[[C1000]] : (<i32>, i32) -> <i1>
// CHECK:           %[[CMPI_EQ1:.*]] = db.cmp eq, %[[AGE]], %[[C18]] : (<i32>, i32) -> <i1>
// CHECK:           %[[CMPI_LT1:.*]] = db.cmp lt, %[[BUDGET]], %[[C1200_4]] : (<f32>, f32) -> <i1>
// CHECK:           %[[CMPI_AND:.*]] = db.and %[[CMPI_LT0]], %[[CMPI_EQ1]] : (<i1>, <i1>) -> <i1>
// CHECK:           %[[CMPI_OR:.*]] = db.or %[[CMPI_AND]], %[[CMPI_LT1]] : (<i1>, <i1>) -> <i1>
// CHECK:           db.filter_yield %[[CMPI_OR]] : {{.*}}
// CHECK-NEXT:      } -> (<3, -1 : [<"age", i32, -1>, <"salary", i32, -1>, <"budget", i32, -1>]>)
// CHECK-NEXT:    return %[[FILTER]] : {{.*}}
func.func @test_filter_op(%arg0: !table) -> !new_table {
    %filtered = db.filter %arg0 : !table {
        ^bb0(%age: !db.column<i32>, %salary: !db.column<i32>, %budget: !db.column<f32>):
            %c1000 = arith.constant 1000 : i32
            %c18 = arith.constant 18 : i32
            %c1200_4 = arith.constant 1200.4 : f32
            %0 = db.cmp lt, %salary, %c1000 : (!db.column<i32>, i32) -> !db.column<i1>
            %1 = db.cmp eq, %age, %c18 : (!db.column<i32>, i32) -> !db.column<i1>
            %2 = db.cmp lt, %budget, %c1200_4 : (!db.column<f32>, f32) -> !db.column<i1>

            %3 = db.and %0, %1 : (!db.column<i1>, !db.column<i1>) -> !db.column<i1>
            %4 = db.or %3, %2 : (!db.column<i1>, !db.column<i1>) -> !db.column<i1>

            db.filter_yield %4 : !db.column<i1>

        } -> (!new_table)

    return %filtered : !new_table
}
