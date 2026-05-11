#include "mlir/CAPI/Registration.h"
#include "kero-c/Registration.h"
#include "Conversion/DBToTensor/DBToTensor.h"
#include "Dialect/DB/IR/DBDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

void KeroRegisterAllDialects(MlirContext context) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<mlir::db::DBDialect>();
    unwrap(context)->appendDialectRegistry(registry);
}
void KeroRegisterAllPasses() {
    mlir::registerAllPasses();
    mlir::db::registerDBToTensorPasses();
}
