// RUN: kero-opt %s --db-to-tensor | FileCheck %s

// CHECK-LABEL: @test_db_scan_to_tensor
module {
    func.func @test_db_scan_to_tensor (%arg0: !db.table<"user">) -> !db.result {
        // CHECK: tensor.empty
        %scan0 = db.scan %arg0 : !db.table<"user"> -> !db.result
        return %scan0 : !db.result
    }

}
