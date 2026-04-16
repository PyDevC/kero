set(DEV_MODULE Development.Module)

find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(BUILD_SHARED_LIBS OFF)

nanobind_add_module(_kero KeroModule.cpp)

target_link_libraries(_kero
    PRIVATE
    DB
    DBToTensorPass

    MLIRArithDialect
    MLIRArithTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRSCFDialect
    MLIRSCFTransforms
    MLIRTensorDialect

    MLIRAffineToStandard
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRConvertToLLVMPass
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRToLLVMIRTranslationRegistration
    MLIRSCFToControlFlow

    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms

    MLIRExecutionEngine
    MLIRMemRefUtils

    LLVMCore
    LLVMSupport
    LLVMTarget
    LLVMX86CodeGen
    LLVMX86AsmParser
)

target_include_directories(_kero PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(_kero PUBLIC cxx_std_17)
install(TARGETS _kero LIBRARY DESTINATION kero)

find_package(Arrow REQUIRED)

nanobind_add_module(_arrow_bridge ArrowBridge.cpp)

target_link_libraries(_arrow_bridge
    PRIVATE
    Arrow::arrow_shared
)

target_include_directories(_arrow_bridge PRIVATE ${PROJECT_SOURCE_DIR}/include)

install(TARGETS _arrow_bridge DESTINATION kero)
