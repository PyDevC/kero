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
//   2. One-Shot Bufferization          tensor → memref
//   3. convert-memref-to-llvm  }
//      convert-linalg-to-loops }       memref/func → LLVM dialect
//      lower-affine            }
//      convert-scf-to-cf       }
//      convert-func-to-llvm    }
//      arith-to-llvm           }
//      reconcile-unrealized-casts
//   4. mlir::ExecutionEngine::create() → JIT function pointer

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

#include "Dialect/DBDialect.h"
#include "Conversion/DBToTensorPass.h"

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
//   n_cols             — C dimension (static column count)
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
// exactly one KeroModule per engine.
// ===========================================================================

class KeroModule {
public:
    KeroModule() {
        // Initialise LLVM backend targets (idempotent).
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
            kero::db::DBDialect
        >();
    }

    // -----------------------------------------------------------------------
    // compile_cpu
    //
    // Steps (§5.1 – §5.5):
    //   1. Parse textual IR → ModuleOp in shared context
    //   2. Run DBToTensorPass (custom, §5.2)
    //   3. One-Shot Bufferization (§5.3)
    //   4. LLVM conversion passes (§5.4)
    //   5. ExecutionEngine JIT (§5.5)
    //   6. Return CompiledQuery (§5.6)
    // -----------------------------------------------------------------------

    CompiledQuery compile_cpu(
        const std::string &ir_text,
        const std::string &table_name,
        const std::vector<std::string> &referenced_columns,
        int64_t n_cols
    ) {
        // -- 1. Parse --------------------------------------------------
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ir_text, _ctx.get());
        if (!module) {
            throw std::runtime_error(
                "KeroModule.compile_cpu: IR parse failed.\nIR:\n" + ir_text
            );
        }

        // -- 2–4. Run the pass pipeline --------------------------------
        mlir::PassManager pm(_ctx.get());
        _build_cpu_pipeline(pm);

        if (mlir::failed(pm.run(*module))) {
            throw std::runtime_error(
                "KeroModule.compile_cpu: pass pipeline failed for table '"
                + table_name + "'"
            );
        }

        // -- 5. JIT via ExecutionEngine --------------------------------
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
                "KeroModule.compile_cpu: ExecutionEngine creation failed: "
                + msg
            );
        }

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

        // Capture engine_ref so the JIT code outlives any individual call.
        auto jit_fn = [rawFn, engine](void **args) { rawFn(args); };

        // -- 6. Build CompiledQuery ------------------------------------
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
    // verify_ir — parse + verify without running the pipeline.
    // Useful for testing the IR Emitter in isolation.
    // Returns "" on success, error string on failure.
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

private:
    std::unique_ptr<mlir::MLIRContext> _ctx;

    // -----------------------------------------------------------------------
    // _build_cpu_pipeline
    //
    // Populates a PassManager with the lowering sequence described in §5.1.
    //
    // Stage 1 — DBToTensorPass
    //   db-dialect → tensor/linalg/arith.  No db-dialect ops remain after.
    //
    // Stage 2 — One-Shot Bufferization (§5.3)
    //   tensor (value semantics) → memref (memory semantics).
    //   bufferizeFunctionBoundaries=true ensures function arguments and
    //   return values are also bufferized, which is required because the
    //   JIT ABI passes memref descriptors directly.
    //
    // Stage 3 — LLVM conversion (§5.4)
    //   linalg → loops → standard CF → LLVM.
    //   Three mandatory passes + reconcile-unrealized-casts.
    // -----------------------------------------------------------------------

    static void _build_cpu_pipeline(mlir::PassManager &pm) {
        // ---- Stage 1: db-dialect → tensor/linalg/arith ---------------
        pm.addPass(kero::createDBToTensorPass());
        pm.addPass(mlir::createCanonicalizationPass());

        // ---- Stage 2: One-Shot Bufferization -------------------------
        mlir::bufferization::OneShotBufferizationOptions bufOpts;
        bufOpts.allowReturnAllocsFromLoops  = true;
        bufOpts.bufferizeFunctionBoundaries = true;
        // Allow unknown ops to be bufferized (needed for linalg.generic
        // with dynamic shapes before all patterns are registered).
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

        // ---- Stage 3: linalg/tensor/memref → LLVM -------------------
        // 3a. Linalg on tensors → loops
        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        // 3b. Affine → standard SCF
        pm.addPass(mlir::createLowerAffinePass());
        // 3c. SCF → ControlFlow
        pm.addPass(mlir::createConvertSCFToControlFlowPass());
        // 3d. MemRef → LLVM  (Developer Guide §5.4: convert-memref-to-llvm)
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        // 3e. Func → LLVM   (Developer Guide §5.4: convert-func-to-llvm)
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        // 3f. Arith → LLVM
        pm.addPass(mlir::createArithToLLVMConversionPass());
        // 3g. ControlFlow → LLVM
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        // 3h. Reconcile any unrealized casts (Developer Guide §5.4)
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
queries (§2.3: one MLIRContext per engine).

Typical usage (handled automatically by kero/engine/compiler.py)::

    km = _kero.KeroModule()
    cq = km.compile_cpu(ir_text, table_name, referenced_columns, n_cols)
    # cq.call_jit(list_of_buffer_addresses)
)";

    // -----------------------------------------------------------------------
    // CompiledQuery
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
                "Ordered column names matching function arg1..N.")
        .def_ro("ir_text",
                &CompiledQuery::ir_text,
                "Textual db-dialect MLIR (for debugging).")
        .def_ro("n_cols",
                &CompiledQuery::n_cols,
                "Number of columns (static C dimension).")
        .def("call_jit",
             [](CompiledQuery &cq, const std::vector<uintptr_t> &ptrs) {
                 // Developer Guide §5.5: args are void** — a pointer to
                 // an array of pointers, each pointing to a memref
                 // descriptor struct (allocated by executor.cpp).
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
    // KeroModule
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
        .def("__repr__", [](const KeroModule &) {
            return "<KeroModule>";
        });
}
