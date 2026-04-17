#include "llvm/Support/TargetSelect.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "Conversion/DBToTensor/DBToTensor.h"
#include "Dialect/DB/IR/DBDialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

struct CompiledQuery {
    std::function<void(void**)> jit_fn;
    std::string table_name;
    std::vector<std::string> referenced_columns;
    std::string ir_text;
    int64_t n_cols{0};
    std::shared_ptr<mlir::ExecutionEngine> engine_ref;
};

class KeroModule {
    public:
    KeroModule() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        mlir::DialectRegistry registry;

        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);

        mlir::registerAllToLLVMIRTranslations(registry);

        mlir::registerGPUDialectTranslation(registry);

        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);

        _ctx = std::make_unique<mlir::MLIRContext>(registry);

        _ctx->loadDialect<
            mlir::func::FuncDialect,
            mlir::arith::ArithDialect,
            mlir::linalg::LinalgDialect,
            mlir::tensor::TensorDialect,
            mlir::memref::MemRefDialect,
            mlir::bufferization::BufferizationDialect,
            mlir::scf::SCFDialect,
            mlir::cf::ControlFlowDialect,
            mlir::LLVM::LLVMDialect,
            mlir::db::DBDialect,
            mlir::gpu::GPUDialect>();

        _ctx->disableMultithreading();
    }

    CompiledQuery compile_cpu(
        const std::string& ir_text,
        const std::string& table_name,
        const std::vector<std::string>& referenced_columns,
        int64_t n_cols) {
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module) {
            throw std::runtime_error(
                "KeroModule.compile_cpu: IR parse failed.\nIR:\n" + ir_text);
        }

        mlir::PassManager pm(_ctx.get());
        pm.enableVerifier(true);
        _build_cpu_pipeline(pm);

        if (mlir::failed(pm.run(*module))) {
            throw std::runtime_error(
                "KeroModule.compile_cpu: pass pipeline failed for table '" +
                table_name + "'");
        }

        mlir::ExecutionEngineOptions options;
        auto maybeEngine = mlir::ExecutionEngine::create(*module, options);

        if (!maybeEngine) {
            std::string msg;
            llvm::handleAllErrors(
                maybeEngine.takeError(),
                [&](const llvm::ErrorInfoBase& e) { msg = e.message(); });
            throw std::runtime_error(
                "KeroModule.compile_cpu: ExecutionEngine creation failed: " +
                msg);
        }

        std::shared_ptr<mlir::ExecutionEngine> engine(
            std::move(*maybeEngine));

        auto fnSym = engine->lookup("query");
        if (!fnSym) {
            std::string msg;
            llvm::handleAllErrors(
                fnSym.takeError(),
                [&](const llvm::ErrorInfoBase& e) { msg = e.message(); });
            throw std::runtime_error(
                "KeroModule.compile_cpu: symbol 'query' not found: " + msg);
        }

        auto rawFn = reinterpret_cast<void (*)(void**)>(*fnSym);

        auto jit_fn = [rawFn, engine](void** args) { rawFn(args); };

        CompiledQuery cq;
        cq.jit_fn = std::move(jit_fn);
        cq.table_name = table_name;
        cq.referenced_columns = referenced_columns;
        cq.ir_text = ir_text;
        cq.n_cols = n_cols;
        cq.engine_ref = std::move(engine);
        return cq;
    }

    CompiledQuery compile_gpu(
        const std::string& ir_text,
        const std::string& table_name,
        const std::vector<std::string>& referenced_columns,
        int64_t n_cols,
        const std::string& target = "nvidia") {
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module) {
            throw std::runtime_error(
                "KeroModule.compile_gpu: IR parse failed.\nIR:\n" + ir_text);
        }

        mlir::PassManager pm(_ctx.get());
        pm.enableVerifier(true);

        if (target == "nvidia" || target == "cuda") {
            _build_gpu_pipeline_nvidia(pm);
        } else if (target == "amd" || target == "rocm") {
            _build_gpu_pipeline_amd(pm);
        } else {
            throw std::runtime_error(
                "KeroModule.compile_gpu: unknown target '" + target +
                "'. Supported: 'nvidia' (CUDA), 'amd' (ROCm)");
        }

        if (mlir::failed(pm.run(*module))) {
            throw std::runtime_error(
                "KeroModule.compile_gpu: GPU pass pipeline failed for table '" +
                table_name + "' (target='" + target + "')");
        }

        mlir::ExecutionEngineOptions options;
        auto maybeEngine = mlir::ExecutionEngine::create(*module, options);

        if (!maybeEngine) {
            std::string msg;
            llvm::handleAllErrors(
                maybeEngine.takeError(),
                [&](const llvm::ErrorInfoBase& e) { msg = e.message(); });
            throw std::runtime_error(
                "KeroModule.compile_gpu: ExecutionEngine creation failed: " +
                msg);
        }

        std::shared_ptr<mlir::ExecutionEngine> engine(
            std::move(*maybeEngine));

        auto fnSym = engine->lookup("query");
        if (!fnSym) {
            std::string msg;
            llvm::handleAllErrors(
                fnSym.takeError(),
                [&](const llvm::ErrorInfoBase& e) { msg = e.message(); });
            throw std::runtime_error(
                "KeroModule.compile_gpu: symbol 'query' not found: " + msg);
        }

        auto rawFn = reinterpret_cast<void (*)(void**)>(*fnSym);
        auto jit_fn = [rawFn, engine](void** args) { rawFn(args); };

        CompiledQuery cq;
        cq.jit_fn = std::move(jit_fn);
        cq.table_name = table_name;
        cq.referenced_columns = referenced_columns;
        cq.ir_text = ir_text;
        cq.n_cols = n_cols;
        cq.engine_ref = std::move(engine);
        return cq;
    }

    std::string verify_ir(const std::string& ir_text) {
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module)
            return "parse error";
        if (mlir::failed(module->verify()))
            return "verification error";
        return "";
    }

    std::string lower_to_llvm_ir(const std::string& ir_text) {
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module)
            throw std::runtime_error("lower_to_llvm_ir: IR parse failed");

        mlir::PassManager pm(_ctx.get());
        _build_cpu_pipeline(pm);

        if (mlir::failed(pm.run(*module)))
            throw std::runtime_error(
                "lower_to_llvm_ir: pass pipeline failed");

        std::string llvmIR;
        llvm::raw_string_ostream os(llvmIR);
        module->print(os);
        return llvmIR;
    }

    private:
    std::unique_ptr<mlir::MLIRContext> _ctx;

    static void _build_cpu_pipeline(mlir::PassManager& pm) {
        pm.addPass(mlir::db::createDBToTensor());
        pm.addPass(mlir::createCanonicalizerPass());

        mlir::bufferization::OneShotBufferizePassOptions bufOpts;
        bufOpts.allowReturnAllocsFromLoops = true;
        bufOpts.bufferizeFunctionBoundaries = true;
        pm.addPass(
            mlir::bufferization::createOneShotBufferizePass(bufOpts));
        pm.addPass(
            mlir::bufferization::createDropEquivalentBufferResultsPass());
        pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());

        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    }

    static void _build_gpu_pipeline_nvidia(mlir::PassManager& pm) {
        pm.addPass(mlir::db::createDBToTensor());
        pm.addPass(mlir::createCanonicalizerPass());

        mlir::bufferization::OneShotBufferizePassOptions bufOpts;
        bufOpts.allowReturnAllocsFromLoops = true;
        bufOpts.bufferizeFunctionBoundaries = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
        pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());
        pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());

        pm.addPass(mlir::createGpuKernelOutliningPass());
        pm.addPass(mlir::createConvertGpuOpsToNVVMOps());
        pm.addPass(mlir::createGpuToLLVMConversionPass());

        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    }

    static void _build_gpu_pipeline_amd(mlir::PassManager& pm) {
        pm.addPass(mlir::db::createDBToTensor());
        pm.addPass(mlir::createCanonicalizerPass());

        mlir::bufferization::OneShotBufferizePassOptions bufOpts;
        bufOpts.allowReturnAllocsFromLoops = true;
        bufOpts.bufferizeFunctionBoundaries = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
        pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());
        pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());
        pm.addPass(mlir::createGpuKernelOutliningPass());

#ifdef MLIR_ROCM_ENABLED
        pm.addPass(mlir::createConvertGpuOpsToROCmOps());
#else
        throw std::runtime_error(
            "KeroModule.compile_gpu: ROCm support not compiled. "
            "Rebuild MLIR with ROCm support enabled.");
#endif

        pm.addPass(mlir::createGpuToLLVMConversionPass());
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    }
};

NB_MODULE(_kero, m) {
    m.doc() = R"(
Kero MLIR compilation pipeline.

This module exposes KeroModule and CompiledQuery to Python.  A single
KeroModule instance is created per KeroEngine and shared across all
queries (Developer Guide §2.3: one MLIRContext per engine).

Typical usage (handled automatically by kero/engine/compiler.py)::

    km = _kero.KeroModule()
    cq = km.compile_cpu(ir_text, table_name, referenced_columns, n_cols)
    # cq.call_jit(list_of_buffer_addresses)
)";

    // -----------------------------------------------------------------------
    // CompiledQuery Python bindings
    // -----------------------------------------------------------------------
    nb::class_<CompiledQuery>(m, "CompiledQuery",
                              "Compiled query object returned by KeroModule.compile_cpu().\n\n"
                              "Carries the JIT function pointer and execution metadata.\n"
                              "Must be kept alive as long as call_jit() may be invoked.")
        .def_ro("table_name",
                &CompiledQuery::table_name,
                "Name of the table this query operates on.")
        .def_ro("referenced_columns",
                &CompiledQuery::referenced_columns,
                "Ordered column names matching function arg1..N.\n"
                "This is the argument-order contract (Developer Guide §10.1).")
        .def_ro("ir_text",
                &CompiledQuery::ir_text,
                "Original db-dialect MLIR text (for debugging).")
        .def_ro("n_cols",
                &CompiledQuery::n_cols,
                "Number of columns (static C dimension, from schema).")
        .def("call_jit", [](CompiledQuery& cq, const std::vector<uintptr_t>& ptrs) {
                 std::vector<void*> vptrs;
                 vptrs.reserve(ptrs.size());
                 for (auto p : ptrs)
                     vptrs.push_back(reinterpret_cast<void*>(p));
                 cq.jit_fn(vptrs.data()); }, "ptrs"_a, "Call the JIT-compiled @query function.\n\n"
                                                                                                  "ptrs must be a list of integer addresses of memref descriptor\n"
                                                                                                  "structs, in the argument order recorded in referenced_columns.\n"
                                                                                                  "The first entry is always the full-table memref (arg0).")
        .def("__repr__", [](const CompiledQuery& cq) { return "<CompiledQuery table='" + cq.table_name +
                                                           "' n_cols=" + std::to_string(cq.n_cols) + ">"; });

    nb::class_<KeroModule>(m, "KeroModule",
                           "MLIR compilation pipeline owner.\n\n"
                           "Create one instance per KeroEngine.  It owns the MLIRContext\n"
                           "for the engine lifetime (Developer Guide §2.3).")
        .def(nb::init<>(),
             "Create a KeroModule and initialise LLVM JIT targets.")
        .def("compile_cpu",
             &KeroModule::compile_cpu,
             "ir_text"_a,
             "table_name"_a,
             "referenced_columns"_a,
             "n_cols"_a,
             "Parse *ir_text* (db-dialect MLIR), run the CPU lowering\n"
             "pipeline (DBToTensorPass → bufferization → LLVM), and\n"
             "return a CompiledQuery with a live JIT function pointer.")
        .def("compile_gpu",
             &KeroModule::compile_gpu,
             "ir_text"_a,
             "table_name"_a,
             "referenced_columns"_a,
             "n_cols"_a,
             "target"_a = "nvidia",
             "GPU compilation pipeline.\n\n"
             "Compiles the query for NVIDIA (target='nvidia'/'cuda') or\n"
             "AMD (target='amd'/'rocm') GPU targets.\n"
             "Raises RuntimeError if the pipeline fails or the target is\n"
             "unsupported (ROCm requires an MLIR build with ROCm enabled).")
        .def("verify_ir",
             &KeroModule::verify_ir,
             "ir_text"_a,
             "Parse and verify *ir_text* without running the pass pipeline.\n"
             "Returns '' on success or an error message string on failure.\n"
             "Useful for testing the IR Emitter in isolation.")
        .def("lower_to_llvm_ir",
             &KeroModule::lower_to_llvm_ir,
             "ir_text"_a,
             "Run the full lowering pipeline on *ir_text* and return the\n"
             "resulting LLVM dialect IR as a string.  Diagnostic helper;\n"
             "production code uses compile_cpu() instead.")
        .def("__repr__", [](const KeroModule&) {
            return "<KeroModule>";
        });
}
