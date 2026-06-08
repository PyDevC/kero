#include "Dialect/DB/IR/DBDialect.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    
    mlir::registerAllDialects(registry);
    registry.insert<mlir::db::DBDialect>();

    mlir::registerAllPasses();
    mlir::registerConvertToLLVMDependentDialectLoading(registry);
    
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Kero-binary", registry));
}
