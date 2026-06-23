#ifndef CONVERSION_DB_TO_TENSOR_TD_
#define CONVERSION_DB_TO_TENSOR_TD_

include "mlir/Pass/PassBase.td"

def DBToTensor : Pass<"db-to-tensor-and-linalg"> {
    let summary = "lowers db dialect operations directly to Tensor and linalg dialects.";

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::arith::ArithDialect",
        "mlir::tensor::TensorDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::db::DBDialect",
    ];
}

#endif // CONVERSION_DB_TO_TENSOR_TD_
