#include "Dialect/DB/IR/DBDialect.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Conversion/Passes.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    
    mlir::registerAllDialects(registry);
    mlir::registerAllExtensions(registry);
    registry.insert<mlir::db::DBDialect>();

    mlir::registerAllPasses();
    mlir::registerConvertToLLVMDependentDialectLoading(registry);

    mlir::db::registerDBToTensorPasses();
    
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Kero-binary", registry));
}
