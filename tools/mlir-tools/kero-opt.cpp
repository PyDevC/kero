#include "Conversion/DBToTensor/DBToTensor.h"
#include "Dialect/DB/IR/DBDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::db::DBDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::scf::SCFDialect>();
    registry.insert<mlir::tensor::TensorDialect>();

    mlir::db::registerDBToTensorPasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Kero-binary", registry));
}
