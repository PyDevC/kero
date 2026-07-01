add_mlir_dialect_library(DBToTensorPass 
    DBToTensor.cpp
    ${PROJECT_SOURCE_DIR}/lib/Conversion/DBToTensor/

    DEPENDS
    DBDialectIncGen
    DBToTensorIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    DB

    MLIRIR
    MLIRPass
    MLIRSupport
)
