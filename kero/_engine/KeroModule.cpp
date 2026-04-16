// kero/_engine/KeroModule.cpp
//
// MLIR compilation pipeline for the Kero SQL query engine.
//
// This file provides the C++ side of the Python/C++ boundary for
// compilation.  It owns:
//   - The MLIRContext and all registered dialects
//   - The full lowering pipeline: db-dialect → tensor/linalg/arith
//                                            → bufferization (memref)
//                                            → LLVM dialect
//   - The mlir::ExecutionEngine JIT that produces callable function
//     pointers
//   - The nanobind Python bindings that expose compile_cpu() and
//     compile_gpu() to Python
//
// Architecture notes (Developer Guide §2.3, §5):
//   - One MLIRContext per engine lifetime.  ALL IR objects are interned
//     into this context.  Never create a second context.
//   - compile_cpu() accepts the textual IR string emitted by the Python
//     IREmitter, parses it into the shared context, runs the pipeline,
//     and returns a CompiledQuery with a live JIT function pointer.
//
// Pass pipeline (§5.1):
//   1. DBToTensorPass        (custom)  db-dialect → tensor/linalg/arith
//   2. Canonicalization
//   3. One-Shot Bufferization          tensor → memref
//      (bufferizeFunctionBoundaries=true — required for JIT ABI)
//   4. Buffer cleanup passes
//   5. convert-linalg-to-loops
//      lower-affine
//      convert-scf-to-cf
//      convert-memref-to-llvm  (§5.4)
//      convert-func-to-llvm    (§5.4)
//      arith-to-llvm
//      convert-cf-to-llvm
//      reconcile-unrealized-casts (§5.4)
//   6. mlir::ExecutionEngine::create() → JIT function pointer
//
// KEY DESIGN DECISIONS vs. the broken original:
//
//   FIX 1 — Translation interfaces registered at CONTEXT CONSTRUCTION TIME.
//     The original code called registerAllToLLVMIRTranslations() lazily
//     inside compile_cpu() via appendDialectRegistry().  By the time
//     ExecutionEngine::create() walks the module looking for translation
//     interfaces, the func dialect had no LLVMTranslationDialectInterface
//     registered, producing the error:
//       "cannot be converted to LLVM IR: missing
//        `LLVMTranslationDialectInterface` registration for dialect for
//        op: func.func"
//     The fix is to call registerAllToLLVMIRTranslations(registry) in the
//     constructor BEFORE the MLIRContext is created from that registry.
//     This mirrors torch-mlir's approach (lib/CAPI/TorchTypes.cpp) and
//     LingoDB's approach (lib/ExecutionBackends/ExecutionBackend.cpp).
//
//   FIX 2 — LLVMDialect must be explicitly loaded.
//     The LLVM dialect is the target of the entire lowering pipeline; it
//     must be in the context's loaded-dialect set or ops in the post-
//     pipeline module will not be recognised by the ExecutionEngine.
//
//   FIX 3 — bufferizeFunctionBoundaries = true.
//     The JIT calling convention passes memref descriptors across the
//     function boundary (void** ABI).  Setting this to false means the
//     function signature stays in tensor-land and the ExecutionEngine
//     cannot generate the right calling convention stubs.
//
//   FIX 4 — GPU pipeline must include GpuToLLVMConversionPass.
//     Without createGpuToLLVMConversionPass() the gpu.launch_func ops
//     that call into the CUDA/ROCm runtime (cudaLaunchKernel etc.) are
//     never lowered to LLVM calls, causing ExecutionEngine to fail.
//
//   FIX 5 — Removed unused SparseTensor includes.
//     SparseTensor headers are not needed and break builds on MLIR
//     installations compiled without sparse tensor support.
//
//   FIX 6 — Added missing GPU LLVM IR translation include.
//     registerGPUDialectTranslation() requires
//     mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h.

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
// FIX 5: Removed unused SparseTensor headers — they require MLIR to be
// built with sparse tensor support, breaking builds without it.
// #include "mlir/Dialect/SparseTensor/Pipelines/Passes.h"
// #include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

// FIX 1: These headers register the LLVMTranslationDialectInterface on
// each dialect.  They MUST be included so the interfaces can be
// registered into the DialectRegistry before the MLIRContext is built.
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
// FIX 6: Missing header — required by registerGPUDialectTranslation().
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "mlir/Transforms/Passes.h"

// DB dialect and its lowering pass.
#include "Conversion/DBToTensor/DBToTensor.h"
#include "Dialect/DB/IR/DBDialect.h"

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

// ===========================================================================
// CompiledQuery
//
// Returned by compile_cpu().  Carries the JIT function pointer and all
// metadata the Python Executor needs to call it correctly.
//
// Field contract (Developer Guide §5.6, §10.1):
//   jit_fn             — callable via call_jit(); takes void** ABI args
//   table_name         — used by Executor to find the Arrow Table
//   referenced_columns — ordered list matching arg1..N in @query
//   n_cols             — C dimension (static column count in the schema)
//   ir_text            — original db-dialect IR text for debugging
//   engine_ref         — keeps the JIT code alive; never drop this
// ===========================================================================

struct CompiledQuery {
    std::function<void(void**)> jit_fn;
    std::string table_name;
    std::vector<std::string> referenced_columns;
    std::string ir_text;
    int64_t n_cols{0};
    std::shared_ptr<mlir::ExecutionEngine> engine_ref;
};

// ===========================================================================
// KeroModule
//
// Owns the MLIRContext for one KeroEngine instance.  Python creates
// exactly one KeroModule per engine (Developer Guide §2.3: one context).
// ===========================================================================

class KeroModule {
    public:
    KeroModule() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        mlir::DialectRegistry registry;

        // ---- Bufferizable op interface registrations ----------------------
        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);

        // FIX 1: Register ALL LLVM IR translation interfaces at registry
        // construction time — before the MLIRContext is created.
        //
        // This is the canonical pattern used by both torch-mlir and LingoDB:
        //   torch-mlir: lib/CAPI/TorchTypes.cpp
        //     registerAllToLLVMIRTranslations(registry) in MLIRContext ctor.
        //   LingoDB: lib/ExecutionBackends/ExecutionBackend.cpp
        //     registerAllToLLVMIRTranslations(registry) before context init.
        //
        // registerAllToLLVMIRTranslations registers LLVMTranslationDialect-
        // Interface on every dialect that can be translated, including
        // func::FuncDialect (via mlir/Target/LLVMIR/Dialect/Func/...).
        // Without this, ExecutionEngine::create() fails with:
        //   "missing `LLVMTranslationDialectInterface` registration for
        //    dialect for op: func.func"
        mlir::registerAllToLLVMIRTranslations(registry);

        // FIX 6: GPU dialect translation for CUDA/ROCm support.
        // Requires mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h
        // (now included above).
        mlir::registerGPUDialectTranslation(registry);

        // The two explicit calls below are already included by
        // registerAllToLLVMIRTranslations, but are kept for clarity.
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);

        // ---- Construct context from fully-populated registry ---------------
        _ctx = std::make_unique<mlir::MLIRContext>(registry);

        // FIX 2: Load LLVMDialect explicitly.
        // The LLVM dialect is the destination of the entire lowering pipeline.
        // It must be in the loaded-dialect set so that post-pipeline ops
        // (llvm.func, llvm.return, llvm.getelementptr …) are recognised by
        // the ExecutionEngine translator.
        _ctx->loadDialect<
            mlir::func::FuncDialect,
            mlir::arith::ArithDialect,
            mlir::linalg::LinalgDialect,
            mlir::tensor::TensorDialect,
            mlir::memref::MemRefDialect,
            mlir::bufferization::BufferizationDialect,
            mlir::scf::SCFDialect,
            mlir::cf::ControlFlowDialect,
            mlir::LLVM::LLVMDialect, // FIX 2: was missing
            mlir::db::DBDialect,
            mlir::gpu::GPUDialect>();

        _ctx->disableMultithreading();
    }

    // -----------------------------------------------------------------------
    // compile_cpu
    //
    // Full CPU lowering pipeline (Developer Guide §5.1 – §5.5):
    //   1. Parse textual IR (db-dialect MLIR) → ModuleOp in shared context
    //   2. Run DBToTensorPass  (db → tensor/linalg/arith)
    //   3. One-Shot Bufferization (tensor → memref)
    //   4. LLVM conversion passes
    //   5. ExecutionEngine JIT → function pointer
    //   6. Return CompiledQuery
    //
    // Parameters
    // ----------
    // ir_text            : textual MLIR module emitted by IREmitter
    // table_name         : name of the queried table (for Executor lookup)
    // referenced_columns : column arg order contract (Developer Guide §10.1)
    // n_cols             : static column count (C dimension for the JIT ABI)
    // -----------------------------------------------------------------------
    CompiledQuery compile_cpu(
        const std::string& ir_text,
        const std::string& table_name,
        const std::vector<std::string>& referenced_columns,
        int64_t n_cols) {
        // ---- 1. Parse IR --------------------------------------------------
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module) {
            throw std::runtime_error(
                "KeroModule.compile_cpu: IR parse failed.\nIR:\n" + ir_text);
        }

        // ---- 2–4. Run pass pipeline ----------------------------------------
        mlir::PassManager pm(_ctx.get());
        pm.enableVerifier(true);
        _build_cpu_pipeline(pm);

        if (mlir::failed(pm.run(*module))) {
            throw std::runtime_error(
                "KeroModule.compile_cpu: pass pipeline failed for table '" +
                table_name + "'");
        }

        // ---- 5. JIT via ExecutionEngine ------------------------------------
        // Developer Guide §5.5: ExecutionEngine wraps LLVM ORC JIT.
        // It must stay alive as long as the function pointer is in use.
        //
        // NOTE: We no longer call appendDialectRegistry / registerAll* here.
        // Those registrations happened at context construction time (FIX 1),
        // so the context already has all translation interfaces loaded.
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

        // Wrap in shared_ptr so it survives the CompiledQuery lifetime.
        std::shared_ptr<mlir::ExecutionEngine> engine(
            std::move(*maybeEngine));

        // Look up the @query symbol produced by the IR Emitter.
        auto fnSym = engine->lookup("query");
        if (!fnSym) {
            std::string msg;
            llvm::handleAllErrors(
                fnSym.takeError(),
                [&](const llvm::ErrorInfoBase& e) { msg = e.message(); });
            throw std::runtime_error(
                "KeroModule.compile_cpu: symbol 'query' not found: " + msg);
        }

        // Cast to the void** ABI expected by ExecutionEngine (§5.5).
        auto rawFn = reinterpret_cast<void (*)(void**)>(*fnSym);

        // Capture engine so JIT code outlives any individual call.
        auto jit_fn = [rawFn, engine](void** args) { rawFn(args); };

        // ---- 6. Build and return CompiledQuery ----------------------------
        CompiledQuery cq;
        cq.jit_fn = std::move(jit_fn);
        cq.table_name = table_name;
        cq.referenced_columns = referenced_columns;
        cq.ir_text = ir_text;
        cq.n_cols = n_cols;
        cq.engine_ref = std::move(engine);
        return cq;
    }

    // -----------------------------------------------------------------------
    // compile_gpu
    //
    // Full GPU lowering pipeline for NVIDIA (CUDA) or AMD (ROCm):
    //   1. Parse textual IR (db-dialect MLIR) → ModuleOp in shared context
    //   2. Run DBToTensorPass  (db → tensor/linalg/arith)
    //   3. One-Shot Bufferization (tensor → memref)
    //   4. GPU lowering passes (convert to gpu dialect, emit gpu.launch_func)
    //   5. GPU to NVVM/ROCM conversion (gpu → NVVM/ROCm LLVM)
    //   6. GPU to LLVM conversion (gpu runtime calls → LLVM)  [FIX 4]
    //   7. ExecutionEngine JIT → function pointer
    //   8. Return CompiledQuery
    //
    // Parameters
    // ----------
    // ir_text            : textual MLIR module emitted by IREmitter
    // table_name         : name of the queried table (for Executor lookup)
    // referenced_columns : column arg order contract (Developer Guide §10.1)
    // n_cols             : static column count (C dimension for the JIT ABI)
    // target             : "nvidia" for CUDA, "amd" for ROCm
    // -----------------------------------------------------------------------
    CompiledQuery compile_gpu(
        const std::string& ir_text,
        const std::string& table_name,
        const std::vector<std::string>& referenced_columns,
        int64_t n_cols,
        const std::string& target = "nvidia") {
        // ---- 1. Parse IR --------------------------------------------------
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module) {
            throw std::runtime_error(
                "KeroModule.compile_gpu: IR parse failed.\nIR:\n" + ir_text);
        }

        // ---- 2–6. Run GPU pass pipeline -----------------------------------
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

        // ---- 7. JIT via ExecutionEngine ------------------------------------
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

    // -----------------------------------------------------------------------
    // verify_ir
    //
    // Parse and verify IR without running the pass pipeline.
    // Returns "" on success, error string on failure.
    // -----------------------------------------------------------------------
    std::string verify_ir(const std::string& ir_text) {
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module)
            return "parse error";
        if (mlir::failed(module->verify()))
            return "verification error";
        return "";
    }

    // -----------------------------------------------------------------------
    // lower_to_llvm_ir
    //
    // Run the full lowering pipeline and return the LLVM dialect IR as text.
    // Diagnostic helper; production code uses compile_cpu() instead.
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // _build_cpu_pipeline
    //
    // Populates a PassManager with the full lowering sequence.
    //
    // Stage 1 — DBToTensorPass (Developer Guide §5.2)
    //   db-dialect ops → tensor/linalg/arith.  No db ops remain after.
    //   Canonicalization cleans up trivially dead code.
    //
    // Stage 2 — One-Shot Bufferization (Developer Guide §5.3)
    //   tensor (value semantics) → memref (memory semantics).
    //
    //   FIX 3: bufferizeFunctionBoundaries = true  (was false).
    //   The JIT ABI passes memref descriptors across function boundaries
    //   (void** calling convention).  With false the function signature
    //   stays in tensor-land and the ExecutionEngine cannot generate the
    //   correct calling-convention stubs, causing silent mismatches or
    //   crashes.  torch-mlir and LingoDB both set this to true.
    //
    // Stage 3 — Loop and arithmetic lowering
    //   linalg.generic → SCF loops → ControlFlow CFG
    //
    // Stage 4 — LLVM dialect conversion (Developer Guide §5.4)
    //   memref, func, arith, cf → LLVM dialect.
    //   reconcile-unrealized-casts cleans up remaining cast ops.
    // -----------------------------------------------------------------------
    static void _build_cpu_pipeline(mlir::PassManager& pm) {
        // ---- Stage 1: db-dialect → tensor/linalg/arith -------------------
        pm.addPass(mlir::db::createDBToTensor());
        pm.addPass(mlir::createCanonicalizerPass());

        // ---- Stage 2: One-Shot Bufferization -----------------------------
        mlir::bufferization::OneShotBufferizePassOptions bufOpts;
        bufOpts.allowReturnAllocsFromLoops = true;
        // FIX 3: must be true for the JIT void** calling convention.
        bufOpts.bufferizeFunctionBoundaries = true;
        pm.addPass(
            mlir::bufferization::createOneShotBufferizePass(bufOpts));
        pm.addPass(
            mlir::bufferization::createDropEquivalentBufferResultsPass());
        pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // ---- Stage 3: linalg/SCF → ControlFlow --------------------------
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());

        // ---- Stage 4: LLVM conversion ------------------------------------
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    }

    // -----------------------------------------------------------------------
    // _build_gpu_pipeline_nvidia
    //
    // GPU pipeline for NVIDIA (CUDA) targets.
    // Uses gpu-kernel-outlining + convert-gpu-to-nvvm + gpu-to-llvm pattern.
    //
    // FIX 4: Added createGpuToLLVMConversionPass() (Stage 6).
    // Without it, gpu.launch_func ops that call into the CUDA runtime
    // (cudaLaunchKernel, cudaMemcpy etc.) are never lowered to LLVM calls.
    // The ExecutionEngine then cannot find those symbols and fails at JIT
    // time with "undefined symbol" errors.
    // -----------------------------------------------------------------------
    static void _build_gpu_pipeline_nvidia(mlir::PassManager& pm) {
        // ---- Stage 1: db-dialect → tensor/linalg/arith -------------------
        pm.addPass(mlir::db::createDBToTensor());
        pm.addPass(mlir::createCanonicalizerPass());

        // ---- Stage 2: One-Shot Bufferization -----------------------------
        mlir::bufferization::OneShotBufferizePassOptions bufOpts;
        bufOpts.allowReturnAllocsFromLoops = true;
        bufOpts.bufferizeFunctionBoundaries = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
        pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());
        pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // ---- Stage 3: linalg/SCF → ControlFlow --------------------------
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());

        // ---- Stage 4: GPU kernel outlining -------------------------------
        // Outline gpu.launch bodies to gpu kernels and wrap in gpu.module.
        pm.addPass(mlir::createGpuKernelOutliningPass());

        // ---- Stage 5: GPU → NVVM (CUDA) ----------------------------------
        pm.addPass(mlir::createConvertGpuOpsToNVVMOps());

        // ---- Stage 6: GPU runtime calls → LLVM  (FIX 4) -----------------
        // Lower gpu.launch_func / cudaLaunchKernel / cudaMemcpy etc. to
        // LLVM function calls so the ExecutionEngine can link them.
        pm.addPass(mlir::createGpuToLLVMConversionPass());

        // ---- Stage 7: Final LLVM conversion -----------------------------
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    }

    // -----------------------------------------------------------------------
    // _build_gpu_pipeline_amd
    //
    // GPU pipeline for AMD (ROCm) targets.
    // Uses gpu-kernel-outlining + convert-gpu-to-rocm + gpu-to-llvm pattern.
    // Note: ROCm support depends on MLIR build configuration.
    //
    // FIX 4: Added createGpuToLLVMConversionPass() (Stage 6) — same
    // reasoning as for the NVIDIA pipeline above.
    // -----------------------------------------------------------------------
    static void _build_gpu_pipeline_amd(mlir::PassManager& pm) {
        // ---- Stage 1: db-dialect → tensor/linalg/arith -------------------
        pm.addPass(mlir::db::createDBToTensor());
        pm.addPass(mlir::createCanonicalizerPass());

        // ---- Stage 2: One-Shot Bufferization -----------------------------
        mlir::bufferization::OneShotBufferizePassOptions bufOpts;
        bufOpts.allowReturnAllocsFromLoops = true;
        bufOpts.bufferizeFunctionBoundaries = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
        pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());
        pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
        pm.addPass(mlir::createCanonicalizerPass());

        // ---- Stage 3: linalg/SCF → ControlFlow --------------------------
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createSCFToControlFlowPass());

        // ---- Stage 4: GPU kernel outlining --------------------------------
        pm.addPass(mlir::createGpuKernelOutliningPass());

        // ---- Stage 5: GPU → ROCm (AMD) -----------------------------------
#ifdef MLIR_ROCM_ENABLED
        pm.addPass(mlir::createConvertGpuOpsToROCmOps());
#else
        throw std::runtime_error(
            "KeroModule.compile_gpu: ROCm support not compiled. "
            "Rebuild MLIR with ROCm support enabled.");
#endif

        // ---- Stage 6: GPU runtime calls → LLVM  (FIX 4) -----------------
        // Lower gpu.launch_func / hipLaunchKernel etc. to LLVM calls.
        pm.addPass(mlir::createGpuToLLVMConversionPass());

        // ---- Stage 7: Final LLVM conversion ------------------------------
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    }
};

// ===========================================================================
// nanobind Python bindings
// ===========================================================================

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
                 // Developer Guide §5.5: JIT ABI is void** — a pointer to
                 // an array of void* pointers, each pointing to a memref
                 // descriptor struct allocated by executor.cpp.
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

    // -----------------------------------------------------------------------
    // KeroModule Python bindings
    // -----------------------------------------------------------------------
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
