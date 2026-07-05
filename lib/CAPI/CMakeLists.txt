get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_public_c_api_library(KeroCAPI
    Dialect.cpp
    Registration.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/kero-c/

    LINK_LIBS PUBLIC
    ${dialect_libs}
    ${conversion_libs}
    MLIRIR
    MLIRCAPIIR
    MLIRRegisterAllDialects
    MLIRPass
    MLIRTransforms
    MLIRRegisterAllPasses
    MLIRSupport
    DB
)
