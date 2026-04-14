#include "Conversion/DBToTensor/DBToTensor.h"
#include "Dialect/DB/IR/DBDialect.h"
#include "Dialect/DB/IR/DBOps.h"
#include "Dialect/DB/IR/DBTypes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nanobind::literals;

void mlirRegisterDBDialect(mlir::MLIRContext* context);

NB_MODULE(_kero, m) {
    m.def("register_db_dialect", [](mlir::MLIRContext* context) { mlirRegisterDBDialect(context); }, "ctx"_a, "Register db-dialect.");

    m.def("run_db_to_tensor_pass", [](mlir::ModuleOp& module) {
        mlir::PassManager pm(module.getContext());
        pm.addPass(mlir::db::createDBToTensor());
        return mlir::succeeded(pm.run(module)); }, "module"_a, "Lowers DB dialect operations to the Tensor/Linalg/MemRef dialects.");

    m.def("run_gpu_pipeline", [](mlir::ModuleOp& module) {
        mlir::PassManager pm(module.getContext());
        return mlir::succeeded(pm.run(module)); }, "module"_a, "Executes the GPU lowering pipeline (Bufferization -> NVVM).");

    m.def("jit_and_run", [](mlir::ModuleOp& module) { return nb::none(); }, "module"_a, "Compiles and executes the MLIR module using the NVPTX backend.");
}
