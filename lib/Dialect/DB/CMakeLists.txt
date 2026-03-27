add_mlir_dialect_library(DB
    DBDialect.cpp

    DEPENDS
    DBDialectIncGen

    LINK_LIBS PUBLIC
    MLIRDialect
    MLIRIR
    MLIRSupport
)
