set(LLVM_TARGET_DEFINITIONS DB.td)
mlir_tablegen(DBDialect.h.inc -gen-dialect-decls -name=DB)
mlir_tablegen(DBDialect.cpp.inc -gen-dialect-defs -name=DB)
add_public_tablegen_target(DBDialectIncGen)
