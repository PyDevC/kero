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
//     compile_gpu() (stub) to Python
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
//   4. Buffer cleanup passes
//   5. convert-linalg-to-loops
//      lower-affine
//      convert-scf-to-cf
//      convert-memref-to-llvm  (§5.4: convert-memref-to-llvm)
//      convert-func-to-llvm    (§5.4: convert-func-to-llvm)
//      arith-to-llvm
//      convert-cf-to-llvm
//      reconcile-unrealized-casts (§5.4)
//   6. mlir::ExecutionEngine::create() → JIT function pointer
//
// FIX: The createDBToTensor() function is in namespace mlir::db per
// DBToTensor.cpp.  The call site must use that namespace.
// FIX: createDBToTensorPass() is the exported symbol name used in the
// header — kept consistent with DBToTensor.h.inc generated registration.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "llvm/Support/TargetSelect.h"

// DB dialect and its lowering pass.
#include "Dialect/DB/IR/DBDialect.h"
#include "Conversion/DBToTensor/DBToTensor.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/function.h>

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
    std::function<void(void **)> jit_fn;
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
        // Initialise LLVM backend targets (idempotent across multiple calls).
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();

        // Create the shared context and load every dialect the pipeline uses.
        _ctx = std::make_unique<mlir::MLIRContext>();
        _ctx->loadDialect<
            mlir::func::FuncDialect,
            mlir::arith::ArithDialect,
            mlir::linalg::LinalgDialect,
            mlir::tensor::TensorDialect,
            mlir::memref::MemRefDialect,
            mlir::bufferization::BufferizationDialect,
            mlir::scf::SCFDialect,
            mlir::cf::ControlFlowDialect,
            mlir::db::DBDialect       // custom db dialect
        >();

        // Enable multi-threading for the context if desired.  For now keep
        // single-threaded so pass-manager diagnostics are sequential.
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
        const std::string &ir_text,
        const std::string &table_name,
        const std::vector<std::string> &referenced_columns,
        int64_t n_cols
    ) {
        // ---- 1. Parse IR ------------------------------------------------
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module) {
            throw std::runtime_error(
                "KeroModule.compile_cpu: IR parse failed.\nIR:\n" + ir_text
            );
        }

        // ---- 2–4. Run pass pipeline -------------------------------------
        mlir::PassManager pm(_ctx.get());
        pm.enableVerifier(true);
        _build_cpu_pipeline(pm);

        if (mlir::failed(pm.run(*module))) {
            throw std::runtime_error(
                "KeroModule.compile_cpu: pass pipeline failed for table '"
                + table_name + "'"
            );
        }

        // ---- 5. JIT via ExecutionEngine ----------------------------------
        // Developer Guide §5.5: ExecutionEngine wraps LLVM ORC JIT.
        // It must stay alive as long as the function pointer is in use.
        auto maybeEngine = mlir::ExecutionEngine::create(
            *module,
            /*llvmModuleBuilder=*/nullptr,
            /*transformer=*/nullptr
        );
        if (!maybeEngine) {
            std::string msg;
            llvm::handleAllErrors(
                maybeEngine.takeError(),
                [&](const llvm::ErrorInfoBase &e) { msg = e.message(); }
            );
            throw std::runtime_error(
                "KeroModule.compile_cpu: ExecutionEngine creation failed: " + msg
            );
        }

        // Wrap in shared_ptr so it survives the CompiledQuery lifetime.
        auto engine = std::make_shared<mlir::ExecutionEngine>(
            std::move(*maybeEngine)
        );

        // Look up the @query symbol produced by the IR Emitter.
        auto fnSym = engine->lookup("query");
        if (!fnSym) {
            std::string msg;
            llvm::handleAllErrors(
                fnSym.takeError(),
                [&](const llvm::ErrorInfoBase &e) { msg = e.message(); }
            );
            throw std::runtime_error(
                "KeroModule.compile_cpu: symbol 'query' not found: " + msg
            );
        }

        // Cast to the void** ABI expected by ExecutionEngine (§5.5).
        auto rawFn = reinterpret_cast<void (*)(void **)>(*fnSym);

        // Capture engine so JIT code outlives any individual call.
        auto jit_fn = [rawFn, engine](void **args) { rawFn(args); };

        // ---- 6. Build and return CompiledQuery ---------------------------
        CompiledQuery cq;
        cq.jit_fn             = std::move(jit_fn);
        cq.table_name         = table_name;
        cq.referenced_columns = referenced_columns;
        cq.ir_text            = ir_text;
        cq.n_cols             = n_cols;
        cq.engine_ref         = std::move(engine);
        return cq;
    }

    // -----------------------------------------------------------------------
    // compile_gpu — stub (Developer Guide §9)
    //
    // The GPU path branches off after bufferization (§9.1).  The CPU
    // pipeline is fully isolated; no GPU code touches it.
    // -----------------------------------------------------------------------
    CompiledQuery compile_gpu(
        const std::string & /*ir_text*/,
        const std::string & /*table_name*/,
        const std::vector<std::string> & /*referenced_columns*/,
        int64_t /*n_cols*/,
        const std::string &target = "nvidia"
    ) {
        throw std::runtime_error(
            "KeroModule.compile_gpu: not yet implemented (target='" + target
            + "').  See Developer Guide §9."
        );
    }

    // -----------------------------------------------------------------------
    // verify_ir
    //
    // Parse and verify IR without running the pass pipeline.
    // Returns "" on success, error string on failure.
    // Useful for testing the IR Emitter in isolation (Developer Guide §11,
    // Phase 1 milestone).
    // -----------------------------------------------------------------------
    std::string verify_ir(const std::string &ir_text) {
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module)
            return "parse error";
        if (mlir::failed(mlir::verify(*module)))
            return "verification error";
        return "";
    }

    // -----------------------------------------------------------------------
    // lower_to_llvm_ir
    //
    // Run the full lowering pipeline and return the LLVM IR text.
    // This is a diagnostic/debugging helper — the normal path uses
    // compile_cpu() which goes all the way to a JIT function pointer.
    //
    // Returns the LLVM IR as a string, or throws on failure.
    // -----------------------------------------------------------------------
    std::string lower_to_llvm_ir(const std::string &ir_text) {
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module)
            throw std::runtime_error("lower_to_llvm_ir: IR parse failed");

        mlir::PassManager pm(_ctx.get());
        _build_cpu_pipeline(pm);

        if (mlir::failed(pm.run(*module)))
            throw std::runtime_error("lower_to_llvm_ir: pass pipeline failed");

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
    //   bufferizeFunctionBoundaries=true is required because the JIT ABI
    //   passes memref descriptors across the function boundary.
    //
    // Stage 3 — Loop and arithmetic lowering
    //   linalg.generic → SCF loops → ControlFlow
    //
    // Stage 4 — LLVM dialect conversion (Developer Guide §5.4)
    //   memref, func, arith, cf → LLVM dialect.
    //   reconcile-unrealized-casts cleans up any remaining cast ops.
    // -----------------------------------------------------------------------
    static void _build_cpu_pipeline(mlir::PassManager &pm) {
        // ---- Stage 1: db-dialect → tensor/linalg/arith ------------------
        // FIX: createDBToTensorPass() is the exported name in DBToTensor.h
        // (generated from the .td pass definition via GEN_PASS_REGISTRATION).
        // The implementation lives in namespace mlir::db, but the header
        // exposes it through the mlir namespace via the generated glue.
        pm.addPass(mlir::db::createDBToTensorPass());
        pm.addPass(mlir::createCanonicalizationPass());

        // ---- Stage 2: One-Shot Bufferization ----------------------------
        mlir::bufferization::OneShotBufferizationOptions bufOpts;
        bufOpts.allowReturnAllocsFromLoops  = true;
        bufOpts.bufferizeFunctionBoundaries = true;
        // Allow ops not covered by a specific bufferization pattern to be
        // treated as unknown (safe for our tensor/linalg dialect mix).
        bufOpts.defaultMemorySpaceFn =
            [](mlir::TensorType) -> std::optional<mlir::Attribute> {
                return std::nullopt;
            };
        pm.addPass(
            mlir::bufferization::createOneShotBufferizePass(bufOpts)
        );
        pm.addPass(
            mlir::bufferization::createDropEquivalentBufferResultsPass()
        );
        pm.addPass(mlir::bufferization::createBufferDeallocationPass());
        pm.addPass(mlir::createCanonicalizationPass());

        // ---- Stage 3: linalg/SCF → ControlFlow -------------------------
        // 3a. linalg.generic → SCF loops
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        // 3b. Affine maps → standard SCF
        pm.addPass(mlir::createLowerAffinePass());
        // 3c. SCF → ControlFlow CFG
        pm.addPass(mlir::createConvertSCFToControlFlowPass());

        // ---- Stage 4: LLVM conversion -----------------------------------
        // 4a. MemRef → LLVM struct descriptors (Developer Guide §5.4)
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        // 4b. func.func / func.return → llvm.func / llvm.return
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        // 4c. arith → LLVM arithmetic ops
        pm.addPass(mlir::createArithToLLVMConversionPass());
        // 4d. ControlFlow → LLVM branch ops
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        // 4e. Reconcile any remaining unrealized casts (Developer Guide §5.4)
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
        .def("call_jit",
             [](CompiledQuery &cq, const std::vector<uintptr_t> &ptrs) {
                 // Developer Guide §5.5: the JIT ABI is void** — a pointer to
                 // an array of void* pointers, each pointing to a memref
                 // descriptor struct allocated by executor.cpp.
                 std::vector<void *> vptrs;
                 vptrs.reserve(ptrs.size());
                 for (auto p : ptrs)
                     vptrs.push_back(reinterpret_cast<void *>(p));
                 cq.jit_fn(vptrs.data());
             },
             "ptrs"_a,
             "Call the JIT-compiled @query function.\n\n"
             "ptrs must be a list of integer addresses of memref descriptor\n"
             "structs, in the argument order recorded in referenced_columns.\n"
             "The first entry is always the full-table memref (arg0).")
        .def("__repr__", [](const CompiledQuery &cq) {
            return "<CompiledQuery table='" + cq.table_name
                 + "' n_cols=" + std::to_string(cq.n_cols) + ">";
        });

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
             "GPU compilation — raises RuntimeError (not yet implemented).\n"
             "See Developer Guide §9 for the intended design.")
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
        .def("__repr__", [](const KeroModule &) {
            return "<KeroModule>";
        });
}
