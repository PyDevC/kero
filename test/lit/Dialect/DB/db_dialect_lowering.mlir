// RUN: kero-opt -db-to-tensor-and-linalg %s | FileCheck %s

module {
    func.func @test_db_types(
        %arg0: !db.table<3, 100 : [
            #db.column<"age", i32, 100>,
            #db.column<"salary", i32, 100>,
            #db.column<"budget", i32, 100>
        ]>
    ) -> () {
        return
    }
}
