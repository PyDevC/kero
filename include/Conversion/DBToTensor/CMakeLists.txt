set(LLVM_TARGET_DEFINITIONS DBToTensor.td)
mlir_tablegen(DBToTensor.h.inc -gen-pass-decls -name=DBToTensor)
add_public_tablegen_target(DBToTensorIncGen)
