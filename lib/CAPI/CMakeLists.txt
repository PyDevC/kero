add_mlir_public_c_api_library(KeroCAPI
    Dialect.cpp
    Registration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/kero-c/

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRCAPIIR
    MLIRRegisterAllDialects
    MLIRPass
    MLIRTransforms
    MLIRRegisterAllPasses
    MLIRSupport
    DB
)
