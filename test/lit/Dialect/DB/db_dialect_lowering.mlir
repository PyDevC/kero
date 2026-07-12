// RUN: kero-opt -db-to-tensor-and-linalg %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @lower_scan_op(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<100xi32>, %[[ARG1:.*]]: tensor<100xi32>
// CHECK-SAME:    -> (tensor<100xi32>, tensor<100xi32>)
// CHECK-NEXT:    return %[[ARG0]], %[[ARG1]] : tensor<100xi32>, tensor<100xi32>
func.func @lower_scan_op(
    %arg0: !db.table<2, 100 : [
        #db.column<"age", i32, 100>,
        #db.column<"budget", i32, 100>
    ]>) -> (!db.table<2, 100 : [
        #db.column<"age", i32, 100>, 
        #db.column<"budget", i32, 100>
    ]>) {

    %user = db.scan %arg0 : !db.table<2, 100 : [
        #db.column<"age", i32, 100>,
        #db.column<"budget", i32, 100>]>
        -> !db.table<2, 100 : [
        #db.column<"age", i32, 100>,
        #db.column<"budget", i32, 100>]>

    return %user : !db.table<2, 100 : [
        #db.column<"age", i32, 100>,
        #db.column<"budget", i32, 100>]>
}

// -----
// CHECK-LABEL: func.func @lower_output_op(
// CHECK-SAME:    %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK-NEXT:    return %[[ARG0]] : {{.*}}
func.func @lower_output_op(
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
// CHECK-LABEL: func.func @lower_filter_op(
// CHECK-SAME:    %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}

// CHECK:         %[[MASK_EMPTY:.*]] = tensor.empty() : tensor<100xi1>
// CHECK:         %[[MASK:.*]] = linalg.generic {{{.*}}} 
// CHECK-SAME:      ins(%[[ARG0]], %[[ARG1]] : {{.*}}, {{.*}}) outs(%[[MASK_EMPTY]] : {{.*}})
// CHECK-NEXT:    ^bb0(%{{.*}}: i32, %[[SALARY:.*]]: i32, %{{.*}}: i1):
// CHECK:           %[[C10:.*]] = arith.constant 10 : i32
// CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[SALARY]], %[[C10]] : i32
// CHECK:           linalg.yield %[[CMP]] : i1

// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C100:.*]] = arith.constant 100 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index

// CHECK:         %[[SIZE:.*]] = scf.for %[[IDX1:.*]] = %[[C0]] to %[[C100]] step %[[C1]] 
// CHECK-SAME:      iter_args(%{{.*}} = {{.*}}) -> (index) {
// CHECK:           %[[BIT1:.*]] = tensor.extract %[[MASK]][%[[IDX1]]] : tensor<100xi1>
// CHECK:           scf.if %[[BIT1]]

// CHECK:         %[[OUT0:.*]] = tensor.empty(%{{.*}}) : tensor<?xi32>
// CHECK:         %[[OUT1:.*]] = tensor.empty(%{{.*}}) : tensor<?xi32>

// CHECK:         %[[RES:.*]]:3 = scf.for %[[IDX2:.*]] = %[[C0]] to %[[C100]] step %[[C1]] 
// CHECK-SAME:      iter_args(%[[A0:.*]] = %[[OUT0]], %[[A1:.*]] = %[[OUT1]], %{{.*}} = {{.*}})
// CHECK:           %[[BIT2:.*]] = tensor.extract %[[MASK]][%[[IDX2]]] : tensor<100xi1>
// CHECK:           scf.if %[[BIT2]]
// CHECK:             tensor.extract %[[ARG0]][%[[IDX2]]]
// CHECK:             tensor.insert %{{.*}} into %[[A0]]
// CHECK:             tensor.extract %[[ARG1]][%[[IDX2]]]
// CHECK:             tensor.insert %{{.*}} into %[[A1]]
// CHECK:           } else {
// CHECK:             scf.yield %[[A0]], %[[A1]]
// CHECK:         return %[[RES]]#0, %[[RES]]#1 : tensor<?xi32>, tensor<?xi32>
func.func @lower_filter_op(
    %arg0: !db.table<2, 100 : [
        #db.column<"age", i32, 100>,
        #db.column<"salary", i32, 100>
    ]> ) -> (
    !db.table<2, -1 : [
        #db.column<"age", i32, -1>,
        #db.column<"salary", i32, -1>
    ]>) {

    %filtered = db.filter %arg0 
        : !db.table<2, 100 : [
            #db.column<"age", i32, 100>,
            #db.column<"salary", i32, 100>
        ]> {
        ^bb0(%age: !db.column<i32>, %salary: !db.column<i32>):
            %c0 = arith.constant 10 : i32
            %0 = db.cmpi eq, %salary, %c0 : (!db.column<i32>, i32) -> !db.column<i1>

            db.filter_yield %0 : !db.column<i1>

        } -> (!db.table<2, -1: [
            #db.column<"age", i32, -1>,
            #db.column<"salary", i32, -1>
        ]>)

    return %filtered : !db.table<2, -1: [
            #db.column<"age", i32, -1>,
            #db.column<"salary", i32, -1>
        ]>
}
