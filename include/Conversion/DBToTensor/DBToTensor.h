#ifndef CONVERSION_DB_TO_TENSOR_H_
#define CONVERSION_DB_TO_TENSOR_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace db {

#define GEN_PASS_DECL
#include "Conversion/DBToTensor/DBToTensor.h.inc"

#define GEN_PASS_REGISTRATION
#include "Conversion/DBToTensor/DBToTensor.h.inc"

} // namespace db
} // namespace mlir

#endif // CONVERSION_DB_TO_TENSOR_H_
