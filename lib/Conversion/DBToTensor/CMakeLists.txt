add_mlir_dialect_library(DBToTensorPass
    DBToTensor.cpp
    ${PROJECT_SOURCE_DIR}/lib/Conversion/DBToTensor/

    DEPENDS
    DBToTensorIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect

    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRFuncTransforms
)
