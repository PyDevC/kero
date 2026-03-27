#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Dialect/DB/IR/DBDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::db::DBDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Kero-binary", registry));
}
