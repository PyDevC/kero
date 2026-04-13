// RUN: kero-opt %s --db-to-tensor | FileCheck %s  
  
// CHECK-LABEL: @test_db_types  
// CHECK-SAME: (%arg0: tensor<100x1000xf32>, %arg1: tensor<1x1000xi32>)  
module {  
    func.func @test_db_types (%arg0: !db.table<"user">, %arg1: !db.column<"user", "id", i32>) {  
        return  
    }  
}  
  
// CHECK-LABEL: @test_db_scan_op  
// CHECK-SAME: (%arg0: tensor<100x1000xf32>) -> tensor<100x1000xf32>  
module {  
    func.func @test_db_scan_op (%arg0: !db.table<"user">) -> !db.result {  
        // CHECK: return %arg0 : tensor<100x1000xf32>  
        %scan0 = db.scan %arg0 : !db.table<"user"> -> !db.result  
        return %scan0 : !db.result  
    }  
}  
  
// CHECK-LABEL: @test_db_filter_op  
// CHECK-SAME: (%arg0: tensor<100x1000xf32>, %arg1: tensor<1x1000xi32>) -> tensor<100x1000xf32>  
func.func @test_db_filter_op (%arg0: !db.table<"user">, %arg_age: !db.column<"user", "age", i32>) -> !db.result {  
    // CHECK: %[[MASK:.+]] = linalg.generic
    // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<100x1000xf32>
    // CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32  
    // CHECK: %[[ZEROFILLED:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<100x1000xf32>) -> tensor<100x1000xf32>  
    // CHECK: %[[FILTERED:.+]] = linalg.generic  
    // CHECK: return %[[FILTERED]] : tensor<100x1000xf32> 
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
// CHECK-SAME: (%arg0: tensor<100x1000xf32>, %arg1: tensor<1x1000xi32>) -> tensor<100x1000xf32>  
func.func @test_db_projection_op (%arg0: !db.table<"user">, %arg_age: !db.column<"user", "age", i32>) ->  !db.result {  
    // CHECK: %[[FILTERED:.+]] = call @test_db_filter_op
    // CHECK: return %[[FILTERED]] : tensor<100x1000xf32> 
    %filtered = func.call @test_db_filter_op(%arg0, %arg_age) : (!db.table<"user">, !db.column<"user", "age", i32>) -> !db.result  
    %projected = db.project %filtered : !db.result -> !db.result  
    return %projected : !db.result  
}  
  
// CHECK-LABEL: @test_db_getcol_op  
// CHECK-SAME: (%arg0: index, %arg1: tensor<1x1000xi32>) -> i32  
func.func @test_db_getcol_op (%arg0: !db.row, %arg1: !db.column<"user", "age", i32>) -> i32 {  
    // CHECK: %[[ZERO:.+]] = arith.constant 0 : index  
    // CHECK: %[[EXTRACTED:.+]] = tensor.extract %arg1[%[[ZERO]], %arg0] : tensor<1x1000xi32>  
    // CHECK: return %[[EXTRACTED]] : i32  
    %age_val = db.getcol %arg0 , %arg1 : (!db.row, !db.column<"user", "age", i32>) -> i32  
    return %age_val : i32  
}  
  
// CHECK-LABEL: @query  
// CHECK-SAME: (%arg0: tensor<100x1000xf32>, %arg1: tensor<1x1000xi32>) -> tensor<100x1000xf32>  
func.func @query (%arg0: !db.table<"PERSON">, %arg_age: !db.column<"PERSON", "age", i32>) -> !db.result {  
    // CHECK: %[[MASK:.+]] = linalg.generic
    // CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<100x1000xf32>
    // CHECK: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32  
    // CHECK: %[[ZEROFILLED:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<100x1000xf32>) -> tensor<100x1000xf32>  
    // CHECK: %[[FILTERED:.+]] = linalg.generic  
    // CHECK: return %[[FILTERED]] : tensor<100x1000xf32>
    %scan0 = db.scan %arg0 : !db.table<"PERSON"> -> !db.result  
    %filter = db.filter %scan0 {  
        ^bb0(%row : !db.row):  
            %no_limit = arith.constant 1 : i1
            db.return %no_limit : i1  
    } : (!db.result) -> !db.result  
    return %filter : !db.result  
}
