get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRIR

    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect

    MLIROptLib
    MLIRPass
    MLIRTransforms
)

add_llvm_executable(kero-opt kero-opt.cpp)
llvm_update_compile_flags(kero-opt)
target_link_libraries(kero-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(kero-opt)
