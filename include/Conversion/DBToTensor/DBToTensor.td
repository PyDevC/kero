#ifndef CONVERSION_DBTOTENSOR_TD_
#define CONVERSION_DBTOTENSOR_TD_

include "mlir/Pass/PassBase.td"

def DBToTensor : Pass<"db-to-tensor"> {
    let summary = "Lower `db-dialect` to tensor dialect.";
    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::db::DBDialect",
    ];
}

#endif // CONVERSION_DBTOTENSOR_TD_
