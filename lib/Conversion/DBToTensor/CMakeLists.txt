add_mlir_dialect_library(DBToTensorPass
    DBToTensor.cpp
    ${PROJECT_SOURCE_DIR}/lib/Conversion/DBToTensor/

    DEPENDS
    DBToTensorIncGen
    DBDialectIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    MLIRTensorDialect
    DB

    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRFuncTransforms
)
