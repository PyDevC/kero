#ifndef CONVERSION_DBTOTENSOR_H_
#define CONVERSION_DBTOTENSOR_H_

#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace db {

#define GEN_PASS_DECL
#include "Conversion/DBToTensor/DBToTensor.h.inc"

#define GEN_PASS_REGISTRATION
#include "Conversion/DBToTensor/DBToTensor.h.inc"

} // namespace db
} // namespace mlir

#endif // CONVERSION_DBTOTENSOR_H_
